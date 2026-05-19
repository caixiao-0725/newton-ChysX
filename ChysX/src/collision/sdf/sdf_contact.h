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
//        -- rhs[p] += -k * depth_p * n_p   (penalty only, no friction).
//   3. `bake_diag(A_diag, n, dt, stream)`
//        -- A.diag[p] += k * (n n^T)       (penalty only, no friction).
//   4. `apply_coulomb_friction(rhs, n, mass, gravity, inv_dt2, stream)`
//        -- Coulomb-cone post-projection: STICK hard-pins dx_t to
//           v_body_t * dt; SLIDE projects F0_t onto the cone boundary.
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

    void set_friction_epsilon(float eps) noexcept { friction_epsilon_ = eps; }
    float friction_epsilon() const noexcept { return friction_epsilon_; }

    void set_contact_kd(float kd) noexcept { contact_kd_ = kd; }
    float contact_kd() const noexcept { return contact_kd_; }

    /// When true, friction is baked implicitly into gradient/Hessian
    /// (IPC style, matching VBD).  When false (default), friction is
    /// applied as a Coulomb-cone post-projection on the assembled rhs.
    void set_ipc_friction_enabled(bool v) noexcept { ipc_friction_ = v; }
    bool ipc_friction_enabled() const noexcept { return ipc_friction_; }

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
    // particle (depth = 0 means no contact).
    //
    // `positions` is a device pointer to `n_particles` Vec3f.
    // `velocities` and `dt` are accepted for API compatibility but
    // are no longer used (friction is handled entirely by
    // `apply_coulomb_friction`).  No-op when `!active()`.
    void detect(const math::Vec3f* positions,
                int                n_particles,
                std::uintptr_t     cuda_stream = 0,
                const math::Vec3f* velocities  = nullptr,
                float              dt          = 0.0f);

    // Penalty gradient (+ optional IPC friction).
    //
    // When `ipc_friction_enabled()`, injects both penalty AND friction
    // forces using the VBD-style IPC model (relative_translation based).
    // `prev_positions` is the x_n array cached by ClothSimulator.
    //
    // When `!ipc_friction_enabled()`, injects penalty only; friction
    // is handled by `apply_coulomb_friction` as a post-projection.
    void accumulate_gradient(math::Vec3f*    rhs,
                             int             n_particles,
                             std::uintptr_t  cuda_stream = 0,
                             const math::Vec3f* positions = nullptr,
                             const math::Vec3f* prev_positions = nullptr,
                             float           dt = 0.0f) const;

    // Penalty Hessian diagonal (+ optional IPC friction Hessian).
    // Same IPC/Coulomb routing as `accumulate_gradient`.
    void bake_diag(math::Mat3f*    A_diag,
                   int             n_particles,
                   float           dt,
                   std::uintptr_t  cuda_stream = 0,
                   const math::Vec3f* positions = nullptr,
                   const math::Vec3f* prev_positions = nullptr) const;

    // Coulomb-cone post-projection of the assembled Newton residual.
    // STICK: hard-pin rhs_t so dx_t = v_body_t * dt.
    // SLIDE: project F0_t onto cone boundary mu * f_n.
    // No-op when `friction() <= 0` or the particle is not in contact.
    void apply_coulomb_friction(math::Vec3f*    rhs,
                                int             n_particles,
                                const float*    mass,
                                const math::Mat3f* diag,
                                const math::Vec3f& gravity,
                                float           inv_dt2,
                                std::uintptr_t  cuda_stream = 0) const;

    /// Inject tangential penalty for STICK particles into diag and rhs
    /// so PCG drives dx_t toward v_body_t * dt.  Must be called AFTER
    /// bake_diag (which bakes penalty nn^T) and AFTER apply_coulomb_friction.
    void bake_stick_constraint(math::Vec3f*    rhs,
                               math::Mat3f*    diag,
                               int             n_particles,
                               const float*    mass,
                               std::uintptr_t  cuda_stream = 0) const;

private:
    // Non-owning pointer (the volume's storage outlives us).
    const SdfVolume* volume_ = nullptr;

    float thickness_        = 0.0f;
    float stiffness_        = 0.0f;
    float friction_         = 0.0f;
    float friction_epsilon_  = 0.01f;
    float contact_kd_       = 1.0e-2f;
    bool  ipc_friction_     = false;

    math::Vec3f body_velocity_ = math::Vec3f(0.0f, 0.0f, 0.0f);

    int          cached_n_particles_ = 0;
    float        cached_dt_          = 0.0f;
    const math::Vec3f* cached_velocities_ = nullptr;

    // Per-particle (normal, depth) cache.
    CudaArray<math::Vec4f> contacts_;
    // Device mirror of `body_velocity_`, one Vec3f.  See the comment
    // above `set_body_velocity` for why this lives on device.
    CudaArray<math::Vec3f> body_velocity_dev_;
};

}  // namespace collision
}  // namespace chysx
