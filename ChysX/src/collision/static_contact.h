// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::StaticContactSet
//
// Penalty contact between cloth particles and a small set of *static*
// rigid primitives (oriented planes + oriented boxes).  Output flows
// directly into the implicit-Euler linear system the cloth simulator
// solves: only the per-particle diagonal block of the Hessian and
// the right-hand side are touched.
//
// Mathematical model
// ------------------
//
// For a particle p with current position x_p and a primitive Σ with
// signed distance function `d(x_p, Σ)` (positive outside, negative
// inside) we use the smooth penalty energy
//
//     E_c = (1/2) * k * max(0, h - d)^2,
//
// where `h = thickness_` and `k = stiffness_`.  Picking the deepest
// active contact per particle (the one with smallest d below `h`)
// gives, at first order in dx,
//
//     ∂E/∂x_p     = -k * (h - d) * n
//     ∂²E/∂x_p²  ≈  k * (n n^T)
//
// where `n = ∇d(x_p, Σ)` is the unit outward normal (computed at the
// closest surface point, or pointing toward the closest face for
// points strictly inside the primitive).  We drop the curvature
// of `n` from the Hessian — this is the standard Gauss-Newton
// approximation and keeps the per-particle 3x3 block SPD regardless
// of penetration depth.
//
// Plugging into chysx's linear system
//
//     A x = b,    A = M/dt² + H_E + k δ_pp (n n^T)
//                 b = M/dt² (x̃ - x_n) - ∇E(x_n) - (-k * depth * n)
//                                                  ^^^^^^^^^^^^^^^^^^
//                                            penalty gradient at p
//
// the contact contribution touches A only at the diagonal block
// `A_pp` and b only at row p.  No cross-particle (i,j) coupling is
// produced by static contacts, so unlike `SelfCollisionConstraint`
// there is no off-diagonal SpMV sidecar (and no contact-aware
// re-capture of the PCG graph either).
//
// Pipeline (one shape set, many frames)
// -------------------------------------
//
//   1.  add_plane(...) / add_box(...) at setup time.  Shapes are
//       host-built and pushed to a small device-side table on first
//       detect() after a change.
//   2.  detect(positions, n) per step — fills a per-particle
//       (depth, normal) cache.  Particles outside every primitive
//       get depth = 0 and a zero normal; the grad / diag passes
//       skip them.
//   3.  accumulate_gradient(rhs)  — chysx step block 3,
//                                   *before* assemble_rhs_kernel.
//   4.  bake_diag(A_diag)         — chysx step block 5,
//                                   *after*  H_.set_zero().
//
// Runtime cost is O(n_particles * (n_planes + n_boxes)); we expect
// each scene to have a handful of primitives, so the whole pipeline
// is essentially three small kernels.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace collision {

// Plane equation: dot(n, x) + d == 0  ⇔  x lies on the plane.
//
// `n` must be a unit vector pointing into the half-space the cloth
// should stay in (the "outside").  The signed distance of an
// arbitrary point x to the plane is `dot(n, x) + d`; positive means
// above the plane (free), negative means penetrating.
struct PlaneShape {
    math::Vec3f n;
    float       d;
};

// Oriented box.  `center` is the world-space centre, `(ex, ey, ez)`
// are the three orthonormal column vectors of the box's local frame
// (so the box-local axes pulled into world space), and `half_ext`
// stores the half-extents along (ex, ey, ez).  An axis-aligned box
// uses `ex = (1,0,0)`, `ey = (0,1,0)`, `ez = (0,0,1)`.
//
// Signed distance for a world-space point x:
//
//     q  = (dot(x - c, ex), dot(x - c, ey), dot(x - c, ez))   [local]
//     dᵢ = |qᵢ| - hᵢ
//     outside  (any dᵢ > 0):  signed_dist = ‖max(d, 0)‖₂        > 0
//     inside  (all dᵢ ≤ 0):  signed_dist = max(d) (≤ 0)
//
// The outward normal is `R · (clamped_local / signed_dist)` outside,
// or the column of R corresponding to the closest face when inside.
struct BoxShape {
    math::Vec3f center;
    math::Vec3f half_ext;
    math::Vec3f ex;
    math::Vec3f ey;
    math::Vec3f ez;
};

class StaticContactSet {
public:
    StaticContactSet() = default;

    StaticContactSet(const StaticContactSet&)            = delete;
    StaticContactSet& operator=(const StaticContactSet&) = delete;
    StaticContactSet(StaticContactSet&&) noexcept            = default;
    StaticContactSet& operator=(StaticContactSet&&) noexcept = default;

    // ---- shape registration -----------------------------------------

    // Drop every previously added plane / box.  Use this between
    // scenes; for steady-state simulation the shape set stays fixed.
    void clear();

    void add_plane(const PlaneShape& p);
    void add_box(const BoxShape& b);

    int n_planes() const noexcept { return n_planes_; }
    int n_boxes()  const noexcept { return n_boxes_; }

    // ---- material parameters ----------------------------------------

    void set_thickness(float t) noexcept { thickness_ = t; }
    float thickness() const noexcept { return thickness_; }

    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // Coulomb friction coefficient `μ` (dimensionless).  When
    // non-zero, every particle in active contact gets a Lagged-Newton
    // linearisation of the IPC-style isotropic Coulomb friction
    // (Li et al. 2020) baked onto the implicit-Euler Hessian:
    //
    //     α_p           = μ · f_n,p · f1_SF_over_x(‖u_t,p^lag‖)
    //     A_diag[p]    += α_p · (I - n n^T)
    //
    // where `f_n,p = k · depth_p` is the normal load, `u_t,p^lag` is
    // the tangential part of the previous step's particle displacement
    // (`v · dt`) projected onto the contact tangent plane, and
    // `f1_SF_over_x` smoothly ramps from `2/ε_u` at `‖u_t‖=0` to
    // `1/‖u_t‖` once `‖u_t‖ > ε_u`.  The resulting tangential force at
    // the solution `dx_t` is `f_t = -α_p · dx_t`, automatically bounded
    // by `‖f_t‖ ≤ μ · f_n` (the Coulomb cone) without an explicit
    // projection step.  Zero disables friction (default).
    //
    // Replaces the earlier viscous tangential damping (`μ_v` [N·s/m]):
    // the previous formulation scaled like `μ_v / dt` on the diagonal
    // and was unstable whenever `μ_v / dt` dwarfed `M / dt^2`.  The
    // Coulomb model self-caps because the diagonal contribution decays
    // proportionally to `1 / ‖u_t‖`, so a large `μ` no longer creates
    // an over-stiff tangential block at standstill.
    void set_friction(float mu) noexcept { friction_ = mu; }
    float friction() const noexcept { return friction_; }

    // Tangential slip regularisation distance `ε_u` [m].  Sets the
    // sticking band: tangential displacements smaller than `ε_u`
    // produce a force linear in `dx_t` (with stiffness `μ·f_n / ε_u`),
    // beyond `ε_u` the force saturates at the Coulomb limit `μ·f_n`.
    // `1e-4` m is a reasonable default for cloth-on-table contact at
    // millimetre cell sizes.
    void set_friction_epsilon(float eps_u) noexcept { friction_epsilon_ = eps_u; }
    float friction_epsilon() const noexcept { return friction_epsilon_; }

    // True if there is any work to do this step.
    bool active() const noexcept {
        return stiffness_ > 0.0f && (n_planes_ > 0 || n_boxes_ > 0);
    }

    // ---- per-step contact pipeline ----------------------------------

    // Per-particle DCD against every plane + box.  After the call,
    // the internal cache holds the (depth, normal) of the *deepest*
    // primitive penetration for each particle, plus -- when both
    // `velocities` and `dt` are provided -- the lagged tangential
    // slip `u_t = (v - n (n·v)) * dt` baked into a separate slip
    // cache (consumed by `bake_diag` to evaluate the IPC-style
    // friction block).  Particles outside every primitive get
    // depth = 0; the grad / diag passes skip them in O(1) per
    // particle.
    //
    // `positions` and `velocities` are device pointers to
    // `n_particles` Vec3f, e.g. Newton's `state.particle_q.ptr` /
    // `state.particle_qd.ptr` cast through `reinterpret_cast`.  When
    // `velocities` is null or `dt <= 0`, the slip cache is zeroed
    // and friction degenerates to the static-only branch (force
    // linear in `dx_t` with stiffness `μ·f_n / ε_u`, capped at
    // `μ·f_n`).  Throws if `n_particles <= 0` or `positions` is null.
    void detect(const math::Vec3f* positions,
                int                n_particles,
                std::uintptr_t     cuda_stream = 0,
                const math::Vec3f* velocities  = nullptr,
                float              dt          = 0.0f);

    // rhs[p] += -k * depth_p * n_p   (chysx's "+grad E" sign).
    //
    // Must be called on the same `n_particles` that was last passed
    // to `detect()`; otherwise this is a silent no-op (defensive).
    void accumulate_gradient(math::Vec3f*    rhs,
                             int             n_particles,
                             std::uintptr_t  cuda_stream = 0) const;

    // A_diag[p] += k * (n_p n_p^T)  +  α_p * (I - n_p n_p^T).
    //
    // The first term is the normal-force Gauss-Newton block; the
    // second is the IPC-style Coulomb friction tangent block, with
    //
    //     α_p = μ · k · depth_p · f1_SF_over_x(‖u_t,p^lag‖)
    //
    // and `f1_SF_over_x` smoothly ramping from `2/ε_u` at zero slip
    // to `1/‖u_t‖` past the regularisation band.  The friction block
    // is skipped when `friction() == 0`, when no slip cache was
    // populated by the last `detect()`, or when `depth_p <= 0`.
    // ``dt`` is the simulation step size and is unused by the
    // Coulomb block (kept as an argument so the call sites in
    // ClothSimulator do not need a separate code path).
    //
    // Same precondition as `accumulate_gradient` — call with the same
    // `n_particles` that `detect()` last saw.
    void bake_diag(math::Mat3f*   A_diag,
                   int             n_particles,
                   float           dt,
                   std::uintptr_t  cuda_stream = 0) const;

private:
    void upload_shapes_();

    int   n_planes_ = 0;
    int   n_boxes_  = 0;
    float thickness_ = 0.0f;
    float stiffness_ = 0.0f;
    float friction_         = 0.0f;
    float friction_epsilon_ = 1.0e-4f;
    bool  shapes_dirty_ = false;
    int   cached_n_particles_ = 0;
    bool  cached_has_slip_    = false;  // last detect() saw velocities

    // Host + device staging.  Tiny shape counts mean reallocating on
    // every add_*() is fine (cloth scenes have a handful of primitives).
    CudaArray<PlaneShape> planes_;
    CudaArray<BoxShape>   boxes_;

    // Per-particle (nx, ny, nz, depth) packed into a Vec4f so the
    // detect / scatter / bake kernels can load it as a single 16-byte
    // vector and skip-if-inactive in one branch.
    CudaArray<math::Vec4f> contacts_;

    // Per-particle (ux, uy, uz, ‖u_t‖) lagged tangential slip,
    // computed inside `detect_kernel` when velocities are provided.
    // Zero (and ignored) for the velocity-less code path.
    CudaArray<math::Vec4f> slips_;
};

}  // namespace collision
}  // namespace chysx
