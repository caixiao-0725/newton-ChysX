// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::TetFEMConstraint
//
// Four-particle ("N=4") tetrahedral FEM element implementing the stable
// Neo-Hookean model of Smith et al. 2018.  This is a direct port of
// Newton's VBD kernel `evaluate_volumetric_neo_hookean_force_and_hessian`
// into the ChysX constraint framework, producing identical numerical
// results.
//
// Each instance is a tetrahedron with vertices (v0, v1, v2, v3):
//
//   v0 : reference vertex (negative row-sum of Dm_inv)
//   v1, v2, v3 : edges defined by columns of Ds = [x1-x0, x2-x0, x3-x0]
//
// Rest geometry
// -------------
//   Dm      = [x1-x0, x2-x0, x3-x0]  (columns are rest edges from v0)
//   Dm_inv  = inverse(Dm)             (stored per tet as Mat3f)
//   V_rest  = 1 / (det(Dm_inv) * 6)  (rest volume, computed on-the-fly)
//
// Deformation gradient
// --------------------
//   Ds = [x1-x0, x2-x0, x3-x0]       (current edges)
//   F  = Ds * Dm_inv
//
// Stable Neo-Hookean energy (Smith et al. 2018)
// ----------------------------------------------
//   mu_nh     = mu
//   lambda_nh = lambda + mu
//   alpha     = 1 + mu_nh / lambda_nh
//   J         = det(F)
//   psi(F)    = (mu_nh/2)(||F||^2 - 3) + (lambda_nh/2)(J - alpha)^2
//   E_tet     = V_rest * psi(F)
//
// First Piola-Kirchhoff stress
// ----------------------------
//   P_vec = V_rest * (mu_nh * vec(F) + lambda_nh * (J - alpha) * vec(cof(F)))
//
// 9x9 Hessian (d^2 J / dF^2 term dropped — vanishes in per-vertex blocks)
// ----------------------------
//   H_9x9 = V_rest * (mu_nh * I_9 + lambda_nh * cof_vec outer cof_vec)
//
// Per-vertex assembly
// -------------------
//   For vertex order k, let m = (m0, m1, m2) from rows of Dm_inv
//   (v0 gets negative column sums).
//
//   G = [m0*I3; m1*I3; m2*I3]   (9x3 Jacobian of vec(F) w.r.t. x_k)
//   f_k = -G^T * P_vec
//   H_k = G^T * H_9x9 * G       (3x3 per-vertex Hessian)
//
// Damping (Rayleigh, stiffness-proportional)
// -------------------------------------------
//   F_dot    = Ds_dot * Dm_inv    (with Ds_dot from velocity differences)
//   P_damp   = k_damp * H_9x9 * vec(F_dot)
//   f_damp_k = -G^T * P_damp
//   H_k     *= (1 + k_damp / dt)
//
// Storage
// -------
//   ConstraintN<4>::indices_ : per-tet (v0, v1, v2, v3) as Vec4i
//   Dm_inv_     : per-tet inverse rest shape matrix (Mat3f)
//   materials_  : per-tet (mu, lambda, k_damp) as Vec3f

#pragma once

#include <cstdint>

#include "../../math/matrix.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"
#include "../../memory/device_span.h"
#include "../core/constraint_n.h"

namespace chysx {
namespace constraint {

class TetFEMConstraint : public ConstraintN<4> {
public:
    TetFEMConstraint() = default;
    ~TetFEMConstraint() override = default;

    // Upload `n` tetrahedra.  `host_tets[c]` is the (v0, v1, v2, v3) tuple
    // of tet c.  Dm_inv is computed from the current device-side positions
    // (treated as rest pose).  `host_materials[c]` is (mu, lambda, k_damp).
    void set_tets_from_positions(
        const math::Vec4i* host_tets,
        const math::Vec3f* host_materials,
        int n,
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0);

    // ---- per-step state (for damping) --------------------------------
    // Must be called before accumulate_gradient / accumulate_hessian
    // if any tet has k_damp > 0.
    void set_previous_positions(DeviceSpan<math::Vec3f> pos_prev) noexcept {
        pos_prev_ = pos_prev;
    }
    void set_dt(float dt) noexcept { dt_ = dt; }

    // ---- buffer access -----------------------------------------------
    CudaArray<math::Mat3f>& Dm_inv() noexcept { return Dm_inv_; }
    const CudaArray<math::Mat3f>& Dm_inv() const noexcept { return Dm_inv_; }
    CudaArray<math::Vec3f>& materials() noexcept { return materials_; }
    const CudaArray<math::Vec3f>& materials() const noexcept { return materials_; }

    // ---- Constraint overrides ----------------------------------------

    float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const override;

private:
    CudaArray<math::Mat3f> Dm_inv_;
    CudaArray<math::Vec3f> materials_;  // (mu, lambda, k_damp) per tet
    DeviceSpan<math::Vec3f> pos_prev_;  // non-owning, set each step
    float dt_ = 0.01f;

    mutable CudaArray<float> energy_buffer_;
};

}  // namespace constraint
}  // namespace chysx
