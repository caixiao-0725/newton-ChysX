// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::SdfContact
//
// Penalty contact between cloth particles and an *animated* signed-
// distance-field body (`SdfVolume`).  Same per-step pipeline as
// `StaticContactSet`:
//
//   1. `detect(positions, n, stream, velocities, dt)`
//        -- one thread per particle, trilinear-sample the bound
//           `SdfVolume`, cache (normal, depth) and tangential slip
//           in per-particle Vec4f buffers.
//   2. `accumulate_gradient(rhs, n, stream)`
//        -- rhs[p] += -k · depth_p · n_p  (+ friction RHS).
//   3. `bake_diag(A_diag, n, dt, stream)`
//        -- A.diag[p] += k · (n n^T)  +  α_p · (I - n n^T).
//   4. `apply_coulomb_friction(rhs, n, stream)`
//        -- post-projection of the assembled Newton residual onto the
//           Coulomb cone (additive, on top of the implicit IPC block).
//
// Output and downstream contracts are IDENTICAL to `StaticContactSet`;
// the only thing that changes is where the (normal, depth) come from
// (SDF trilinear sample of an animated body vs. analytic plane/box
// distance functions).  This means a `ClothSimulator` can run both
// detectors in parallel and their contributions just sum into the
// same rhs/diag — penalty additivity makes that physically correct.
//
// Body motion + friction
// ----------------------
//
// The underlying `SdfVolume` carries a per-frame world<-local pose;
// when the body moves, the contact frame moves with it.  To get the
// CORRECT slip cache for friction, we subtract the SDF body's
// linear velocity from each particle's velocity before projecting
// onto the contact tangent:
//
//     u_rel    = v_particle - v_body                            (point on body)
//     u_t^lag = (u_rel - (u_rel · n) · n) · dt
//
// Without this correction a particle sitting still on a slowly
// rising box would see u_rel = -v_box and experience spurious
// Coulomb friction "dragging it down"; with it, the relative slip
// is zero and the friction integrator stays inert (correct).
//
// Set the body's linear velocity via `set_body_velocity(...)`; the
// default (0) reduces to the static-body behaviour.  Angular
// velocity is not modelled (rigid translation only); to extend,
// switch to per-particle `v_body(x_world) = v_lin + ω × (x - pos)`.

#pragma once

#include <cstdint>

#include "../../math/matrix.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"
#include "sdf_volume.h"

namespace chysx {
namespace collision {

class SdfContact {
public:
    SdfContact() = default;

    SdfContact(const SdfContact&)            = delete;
    SdfContact& operator=(const SdfContact&) = delete;
    SdfContact(SdfContact&&) noexcept            = default;
    SdfContact& operator=(SdfContact&&) noexcept = default;

    // ---- volume binding (non-owning) --------------------------------

    // Bind the `SdfVolume` this detector will sample.  The volume
    // must outlive this `SdfContact` (no shared_ptr; the cloth
    // simulator owns both).  Passing `nullptr` disables the detector.
    void bind_volume(const SdfVolume* volume) noexcept { volume_ = volume; }

    const SdfVolume* volume() const noexcept { return volume_; }

    // ---- material parameters (mirror static_contact) ----------------

    void set_thickness(float t) noexcept { thickness_ = t; }
    float thickness() const noexcept { return thickness_; }

    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    void set_friction(float mu) noexcept { friction_ = mu; }
    float friction() const noexcept { return friction_; }

    void set_friction_epsilon(float eps_u) noexcept { friction_epsilon_ = eps_u; }
    float friction_epsilon() const noexcept { return friction_epsilon_; }

    // SDF body's linear velocity in world frame [m/s].  Subtracted
    // from each particle's velocity before projecting onto the
    // contact tangent so friction sees the correct *relative* slip
    // (a particle riding a translating SDF body experiences zero
    // tangential slip when stationary in the body's frame).
    //
    // Internally the latest value is async-copied into a 1-Vec3f
    // device buffer the detect kernel reads through a pointer, so
    // updates inside a CUDA Graph capture survive replay (a
    // by-value kernel argument would have been snapshotted at
    // capture time and silently frozen).
    void set_body_velocity(const math::Vec3f& v,
                           std::uintptr_t cuda_stream = 0);
    const math::Vec3f& body_velocity() const noexcept { return body_velocity_; }

    bool active() const noexcept {
        return stiffness_ > 0.0f && volume_ != nullptr && volume_->active();
    }

    // ---- per-step pipeline ------------------------------------------

    // Per-particle DCD against the bound SDF volume.  After the
    // call, the internal cache holds the (depth, normal) at each
    // particle (depth = 0 means no contact) plus -- when both
    // `velocities` and `dt > 0` are provided -- the lagged
    // tangential slip `u_t = (v_particle - v_body - n (n · (...))) * dt`.
    //
    // `positions` and `velocities` are device pointers to
    // `n_particles` Vec3f.  When `velocities` is null or `dt <= 0`,
    // the slip cache is zeroed and friction collapses to the
    // sticking-band branch.  No-op when `!active()`.
    void detect(const math::Vec3f* positions,
                int                n_particles,
                std::uintptr_t     cuda_stream = 0,
                const math::Vec3f* velocities  = nullptr,
                float              dt          = 0.0f);

    // rhs[p] += -k · depth_p · n_p  (+ optional friction RHS).
    //
    // Must be called on the same `n_particles` that was last passed
    // to `detect()`; otherwise this is a silent no-op (defensive).
    void accumulate_gradient(math::Vec3f*    rhs,
                             int             n_particles,
                             std::uintptr_t  cuda_stream = 0) const;

    // A.diag[p] += k · (n n^T)  +  α_p · (I - n n^T).
    //
    // The friction block is skipped when `friction() == 0`, when no
    // slip cache was populated by the last `detect()`, or when
    // `depth_p <= 0`.  `dt` is unused by the Coulomb block (kept as
    // an argument so the caller in `ClothSimulator` doesn't need a
    // separate code path from `static_contact.bake_diag`).
    void bake_diag(math::Mat3f*    A_diag,
                   int             n_particles,
                   float           dt,
                   std::uintptr_t  cuda_stream = 0) const;

    // Coulomb-cone post-projection of the assembled Newton residual
    // (see `static_contact.h` for the algebraic derivation -- this is
    // the same operator on the SDF contact set).  No-op when
    // `friction() <= 0` or the particle is not in active contact.
    void apply_coulomb_friction(math::Vec3f*    rhs,
                                int             n_particles,
                                std::uintptr_t  cuda_stream = 0) const;

private:
    // Non-owning pointer (the volume's storage outlives us).
    const SdfVolume* volume_ = nullptr;

    float thickness_        = 0.0f;
    float stiffness_        = 0.0f;
    float friction_         = 0.0f;
    float friction_epsilon_ = 1.0e-4f;

    math::Vec3f body_velocity_ = math::Vec3f(0.0f, 0.0f, 0.0f);

    int  cached_n_particles_ = 0;
    bool cached_has_slip_    = false;

    // Per-particle output caches, identical layout to
    // `StaticContactSet::contacts_` / `slips_` so the scatter /
    // diag / cone-projection kernels are mirror images.
    CudaArray<math::Vec4f> contacts_;
    CudaArray<math::Vec4f> slips_;
    // Device mirror of `body_velocity_`, one Vec3f.  See the comment
    // above `set_body_velocity` for why this lives on device.
    CudaArray<math::Vec3f> body_velocity_dev_;
};

}  // namespace collision
}  // namespace chysx
