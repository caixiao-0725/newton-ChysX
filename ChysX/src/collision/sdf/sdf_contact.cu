// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of `SdfContact`.
//
// One thread per particle; the per-particle work is to trilinear-sample
// the bound `SdfVolume`, cache (depth, normal) and -- if velocities
// were provided -- the lagged tangential slip
// `u_t = ((v - v_body) - n ((v - v_body)·n)) * dt`, then have follow-up
// scatter passes push the resulting penalty + IPC-friction contributions
// into the cloth simulator's `rhs` and `H_.diag` arrays.
//
// The scatter / diag / cone-projection kernels here are line-for-line
// the same as `static_contact.cu`'s -- the contact data format
// (per-particle Vec4f `(nx, ny, nz, depth)` + per-particle Vec4f slip
// `(ux, uy, uz, ‖u‖)`) is shared, so the friction algebra is identical.
// We duplicate (rather than expose) the kernels so each detector
// stays self-contained and the build graph has no cross-collision-
// detector linkage.

#include "sdf_contact.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <stdexcept>
#include <string>

#include "../friction.cuh"
#include "sdf_volume.cuh"

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::SdfContact: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// One thread per particle.  Trilinear-samples the SDF, picks the
// (signed) distance and normal, computes the active depth (= thickness
// - sd) and the lagged tangential slip in the body's instantaneous
// rest frame.  depth ≤ 0 means "no active contact" and the downstream
// scatter passes skip it.
__global__ void detect_kernel(
    const float3* __restrict__       positions,
    const float3* __restrict__       velocities,        // may be nullptr
    int                              n_particles,
    SdfVolumeView                    view,
    const math::Vec3f* __restrict__  body_velocity_dev, // 1 Vec3f on device
    float                            thickness,
    float                            dt,
    math::Vec4f* __restrict__        contacts,
    math::Vec4f* __restrict__        slips) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const float3 pf = positions[p];
    const math::Vec3f x(pf.x, pf.y, pf.z);

    float       sd;
    math::Vec3f grad;
    view.sample(x, sd, grad);

    // Out-of-grid sentinel (~1e30) or non-penetrating queries: no contact.
    float depth = thickness - sd;

    math::Vec3f n(0.0f, 0.0f, 0.0f);
    if (depth > 0.0f) {
        // Normalise the SDF gradient to a unit normal.  On the surface
        // the trilinear gradient is already ~unit, but inside the body
        // (or near voxel boundaries) it can drift; normalising keeps
        // `n n^T` projectors well-conditioned.
        const float g2 = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
        const float inv_g = rsqrtf(fmaxf(g2, 1.0e-30f));
        n.x = grad.x * inv_g;
        n.y = grad.y * inv_g;
        n.z = grad.z * inv_g;
    } else {
        // Force depth to exactly 0 for the "no contact" branch the
        // scatter kernels test against (`depth <= 0.0f`).
        depth = 0.0f;
    }

    contacts[p] = math::Vec4f(n.x, n.y, n.z, depth);

    if (slips != nullptr) {
        math::Vec3f u_t(0.0f, 0.0f, 0.0f);
        float       u_norm = 0.0f;
        if (depth > 0.0f && velocities != nullptr && dt > 0.0f) {
            const float3 vf = velocities[p];
            const math::Vec3f vb = body_velocity_dev[0];
            // Relative velocity in the body frame: subtract body
            // linear velocity so a particle riding a translating body
            // sees zero slip.
            const math::Vec3f v_rel(vf.x - vb.x,
                                    vf.y - vb.y,
                                    vf.z - vb.z);
            const math::Vec3f dxv = v_rel * dt;
            const float vn = dot(n, dxv);
            u_t = dxv - n * vn;
            u_norm = sqrtf(u_t.x * u_t.x + u_t.y * u_t.y + u_t.z * u_t.z);
        }
        slips[p] = math::Vec4f(u_t.x, u_t.y, u_t.z, u_norm);
    }
}

// Local alias so the IPC `f1_SF_over_x` call sites stay short.
__device__ inline float f1_sf_over_x(float u_norm, float eps_u) {
    return ipc_f1_sf_over_x(u_norm, eps_u);
}

// rhs[p] += -k · depth · n  +  (-α · u_t^lag).
// Mirrors static_contact's scatter_gradient_kernel exactly.
__global__ void scatter_gradient_kernel(
    const math::Vec4f* __restrict__ contacts,
    const math::Vec4f* __restrict__ slips,
    int                             n_particles,
    float                           stiffness,
    float                           friction_mu,
    float                           friction_epsilon,
    float                           thickness,
    math::Vec3f* __restrict__       rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float kd = -stiffness * depth;
    rhs[p].x += kd * c.x;
    rhs[p].y += kd * c.y;
    rhs[p].z += kd * c.z;

    if (friction_mu > 0.0f && slips != nullptr) {
        // Use a depth floor for friction alpha so that particles at
        // shallow penetration still experience meaningful friction.
        const float depth_fric = fmaxf(depth, 0.1f * thickness);
        const float f_n = stiffness * depth_fric;
        const math::Vec4f slip = slips[p];
        const float u_norm = slip.w;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        rhs[p].x += alpha * slip.x;
        rhs[p].y += alpha * slip.y;
        rhs[p].z += alpha * slip.z;
    }
}

// diag[p] += k · (n n^T)  +  α · (I - n n^T).
// Mirrors static_contact's bake_diag_kernel exactly.
__global__ void bake_diag_kernel(
    const math::Vec4f* __restrict__ contacts,
    const math::Vec4f* __restrict__ slips,
    int                             n_particles,
    float                           stiffness,
    float                           friction_mu,
    float                           friction_epsilon,
    float                           thickness,
    math::Mat3f* __restrict__       diag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float k  = stiffness;
    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    float a00 = k * nx * nx;
    float a01 = k * nx * ny;
    float a02 = k * nx * nz;
    float a11 = k * ny * ny;
    float a12 = k * ny * nz;
    float a22 = k * nz * nz;

    if (friction_mu > 0.0f) {
        const float depth_fric = fmaxf(depth, 0.1f * thickness);
        const float f_n = k * depth_fric;
        const float u_norm = (slips != nullptr) ? slips[p].w : 0.0f;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        a00 += alpha * (1.0f - nx * nx);
        a01 += alpha * (-nx * ny);
        a02 += alpha * (-nx * nz);
        a11 += alpha * (1.0f - ny * ny);
        a12 += alpha * (-ny * nz);
        a22 += alpha * (1.0f - nz * nz);
    }

    math::Mat3f& A = diag[p];
    A.data[0] += a00;
    A.data[1] += a01;
    A.data[2] += a02;
    A.data[3] += a01;
    A.data[4] += a11;
    A.data[5] += a12;
    A.data[6] += a02;
    A.data[7] += a12;
    A.data[8] += a22;
}

// Coulomb-cone post-projection on the assembled Newton residual.
//
// Unified logic for both static and moving bodies.  The key insight:
// `scatter_gradient_kernel` already injected `+α · slip` into the
// gradient, where `slip = (v_particle - v_body) · dt` projected onto
// the tangent plane.  After `assemble_rhs` flips the sign, the
// friction contribution in the final rhs is  `-α · slip`.
//
// When computing F0 (the force excluding the penalty normal push),
// we subtract out the friction force arising from the *body's own
// velocity* component of the slip.  This ensures the Coulomb-cone
// check measures the particle's *relative* tangential force with
// respect to the body, not the absolute force.  A particle riding
// the body at the same velocity will then have F0_t ≈ 0, landing
// in the STICK branch, which zeroes the residual tangential force —
// correctly pinning the particle to the moving body.
//
// Coulomb-cone post-projection on the assembled Newton residual.
//
// STICK branch: replace the tangential rhs so the PCG solves
// dx_t = v_body_t · dt, making the particle ride the body exactly.
//
//   target rhs_t = (M/dt² + alpha) · v_body_t · dt + M · g_t
//                = M · v_body_t / dt + alpha · v_body_t · dt + M · g_t
//
// SLIDING branch: project F0_t onto the Coulomb-cone boundary.
//
// Cone uses  max(depth, 0.1·thickness)  so shallow-penetration
// particles still experience meaningful friction.
__global__ void apply_coulomb_friction_kernel(
    const math::Vec4f* __restrict__  contacts,
    const math::Vec4f* __restrict__  slips,
    int                              n_particles,
    float                            stiffness,
    float                            friction_mu,
    float                            friction_epsilon,
    float                            thickness,
    const math::Vec3f* __restrict__  body_velocity_dev,
    const float* __restrict__        mass,
    float3                           gravity,
    float                            inv_dt2,
    float                            dt,
    math::Vec3f* __restrict__        rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;
    if (friction_mu <= 0.0f) return;

    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    // --- Compute the body-velocity part of the friction force that
    //     scatter_gradient injected into the rhs ---
    //
    // scatter_gradient added:  grad += alpha * slip_vec
    // where slip_vec = (v_p - v_body)*dt projected tangentially.
    // After assemble_rhs flips sign, rhs gets:
    //   rhs += -alpha * slip_vec = alpha*v_body_t*dt - alpha*v_p_t*dt
    //
    // The body-velocity part is  alpha * v_body_t * dt.  But we must
    // compute it using the SAME alpha that scatter_gradient used, and
    // crucially: when slip_vec = 0 (v_p = v_body), scatter injected
    // NOTHING, so the body part is also 0.
    //
    // To stay consistent, we decompose the actual scatter contribution:
    //   scatter_force_in_rhs = -alpha * slip_vec   (after sign flip)
    //   body_part            = -alpha * (-v_body_t*dt)  [tangential only]
    //
    // But since slip_vec might be 0 even though v_body ≠ 0, we scale
    // by the ratio (slip coming from body) / (total slip).  The cleanest
    // approach: body_part = scatter_total + particle_part, where
    // particle_part = -alpha * v_p_t * dt.  But we don't know v_p_t.
    //
    // Simplest correct fix: recompute body_part as  alpha * v_body_t * dt
    // but use the CLAMPED f1 that avoids the 1/eps singularity when
    // slip → 0.  When slip=0 the particle is already riding the body,
    // so f_body should equal the rhs contribution (which is 0).
    // We achieve this by scaling f_body by (slip_norm / max(slip_norm, eps)):
    // at slip=0 the factor is 0 → f_body=0; at slip >> eps the factor → 1.
    float f_body_x = 0.0f, f_body_y = 0.0f, f_body_z = 0.0f;
    if (slips != nullptr && dt > 0.0f) {
        const math::Vec3f vb = body_velocity_dev[0];
        const float vb_n = vb.x * nx + vb.y * ny + vb.z * nz;
        const float vbt_x = vb.x - vb_n * nx;
        const float vbt_y = vb.y - vb_n * ny;
        const float vbt_z = vb.z - vb_n * nz;

        const float depth_fric = fmaxf(depth, 0.1f * thickness);
        const float f_n = stiffness * depth_fric;
        const math::Vec4f slip = slips[p];
        const float u_norm = slip.w;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;

        // Scale factor: 0 when slip=0 (no force was injected), 1 when
        // slip >> eps (full body contribution exists in rhs).
        const float scale = u_norm / fmaxf(u_norm, friction_epsilon);

        f_body_x = scale * alpha * vbt_x * dt;
        f_body_y = scale * alpha * vbt_y * dt;
        f_body_z = scale * alpha * vbt_z * dt;
    }

    const float fn_push = stiffness * depth;
    const math::Vec3f r = rhs[p];
    const math::Vec3f F0(r.x - fn_push * nx - f_body_x,
                         r.y - fn_push * ny - f_body_y,
                         r.z - fn_push * nz - f_body_z);

    const float F0_n_scalar = F0.x * nx + F0.y * ny + F0.z * nz;

    const float depth_for_cone = fmaxf(depth, 0.1f * thickness);
    const float cone = friction_mu * stiffness * depth_for_cone;

    const math::Vec3f F0_t(F0.x - F0_n_scalar * nx,
                           F0.y - F0_n_scalar * ny,
                           F0.z - F0_n_scalar * nz);
    const float F_T_sq = F0_t.x * F0_t.x + F0_t.y * F0_t.y + F0_t.z * F0_t.z;
    if (F_T_sq <= 0.0f) return;
    const float F_T = sqrtf(F_T_sq);

    if (F_T <= cone) {
        // STICK: hard-pin the tangential rhs to produce dx_t = v_body·dt.
        //
        // The Newton system is  H · dx = rhs,  where
        //   H_t ≈ M/dt² + alpha   (tangential diagonal from inertia + friction)
        //
        // We want  dx_t = v_body_t · dt,  so we need
        //   rhs_t = H_t · v_body_t · dt
        //         = (M/dt² + alpha) · v_body_t · dt
        //         = M · v_body_t / dt  +  alpha · v_body_t · dt
        //
        // Also add tangential gravity:  + M · g_t.
        const math::Vec3f vb = body_velocity_dev[0];
        const float vb_n = vb.x * nx + vb.y * ny + vb.z * nz;
        const float vbt_x = vb.x - vb_n * nx;
        const float vbt_y = vb.y - vb_n * ny;
        const float vbt_z = vb.z - vb_n * nz;

        const float m = mass[p];
        const float m_over_dt = m / fmaxf(dt, 1.0e-30f);

        // alpha for friction Hessian (same as bake_diag_kernel uses)
        const float depth_fric = fmaxf(depth, 0.1f * thickness);
        const float f_n_fric = stiffness * depth_fric;
        float alpha_h = 0.0f;
        if (slips != nullptr) {
            const float u_norm = slips[p].w;
            const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
            alpha_h = friction_mu * f_n_fric * f1;
        }

        const float gn = gravity.x * nx + gravity.y * ny + gravity.z * nz;
        const float gt_x = gravity.x - gn * nx;
        const float gt_y = gravity.y - gn * ny;
        const float gt_z = gravity.z - gn * nz;

        // target rhs_t = M * v_body_t / dt  +  alpha * v_body_t * dt  +  M * g_t
        const float target_rhs_t_x = m_over_dt * vbt_x + alpha_h * vbt_x * dt + m * gt_x;
        const float target_rhs_t_y = m_over_dt * vbt_y + alpha_h * vbt_y * dt + m * gt_y;
        const float target_rhs_t_z = m_over_dt * vbt_z + alpha_h * vbt_z * dt + m * gt_z;

        // Current tangential rhs
        const float rhs_n = r.x * nx + r.y * ny + r.z * nz;
        const float rhs_t_x = r.x - rhs_n * nx;
        const float rhs_t_y = r.y - rhs_n * ny;
        const float rhs_t_z = r.z - rhs_n * nz;

        // Replace tangential rhs
        rhs[p].x += (target_rhs_t_x - rhs_t_x);
        rhs[p].y += (target_rhs_t_y - rhs_t_y);
        rhs[p].z += (target_rhs_t_z - rhs_t_z);
    } else {
        // SLIDING: project onto cone boundary.
        const float shrink = cone / F_T;
        rhs[p].x -= shrink * F0_t.x;
        rhs[p].y -= shrink * F0_t.y;
        rhs[p].z -= shrink * F0_t.z;
    }
}

}  // namespace

void SdfContact::detect(const math::Vec3f* positions,
                        int                n_particles,
                        std::uintptr_t     cuda_stream,
                        const math::Vec3f* velocities,
                        float              dt) {
    if (!active() || n_particles <= 0) return;
    if (positions == nullptr) {
        throw std::invalid_argument(
            "chysx::collision::SdfContact::detect: positions must be non-null");
    }

    if (cached_n_particles_ != n_particles) {
        contacts_.allocate_device(static_cast<std::size_t>(n_particles));
        cached_n_particles_ = n_particles;
    }

    const bool need_slip =
        (friction_ > 0.0f) && (velocities != nullptr) && (dt > 0.0f);
    if (need_slip &&
        slips_.gpu_size() != static_cast<std::size_t>(n_particles)) {
        slips_.allocate_device(static_cast<std::size_t>(n_particles));
    }
    cached_has_slip_ = need_slip;
    cached_dt_ = dt;

    SdfVolumeView view = volume_->make_view();

    // Lazy-allocate the 1-Vec3f device buffer for body velocity.
    // Synchronously primed to the current host cache so the very
    // first detect() after construction sees zero (the default)
    // rather than uninitialised memory.
    if (body_velocity_dev_.gpu_size() != 1u) {
        body_velocity_dev_.resize(1u);
        body_velocity_dev_[0] = body_velocity_;
        body_velocity_dev_.copy_to_device(cuda_stream);
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    detect_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        reinterpret_cast<const float3*>(positions),
        reinterpret_cast<const float3*>(velocities),
        n_particles,
        view,
        body_velocity_dev_.gpu_data(),
        thickness_,
        dt,
        contacts_.gpu_data(),
        need_slip ? slips_.gpu_data() : nullptr);
    check_cuda(cudaGetLastError(), "detect_kernel launch");
}

void SdfContact::set_body_velocity(const math::Vec3f& v,
                                   std::uintptr_t cuda_stream) {
    body_velocity_ = v;
    if (body_velocity_dev_.gpu_size() != 1u) {
        body_velocity_dev_.resize(1u);
    }
    body_velocity_dev_[0] = v;
    body_velocity_dev_.copy_to_device(cuda_stream);
}

void SdfContact::accumulate_gradient(math::Vec3f*    rhs,
                                     int             n_particles,
                                     std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const math::Vec4f* slip_ptr =
        (friction_ > 0.0f && cached_has_slip_) ? slips_.gpu_data() : nullptr;
    scatter_gradient_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        slip_ptr,
        n_particles,
        stiffness_,
        friction_,
        friction_epsilon_,
        thickness_,
        rhs);
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

void SdfContact::bake_diag(math::Mat3f*   diag,
                           int             n_particles,
                           float           dt,
                           std::uintptr_t  cuda_stream) const {
    (void)dt;  // Coulomb friction is dt-free (lagged slip).
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const math::Vec4f* slip_ptr =
        (friction_ > 0.0f && cached_has_slip_) ? slips_.gpu_data() : nullptr;
    bake_diag_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        slip_ptr,
        n_particles,
        stiffness_,
        friction_,
        friction_epsilon_,
        thickness_,
        diag);
    check_cuda(cudaGetLastError(), "bake_diag_kernel launch");
}

void SdfContact::apply_coulomb_friction(math::Vec3f*       rhs,
                                        int                n_particles,
                                        const float*       mass,
                                        const math::Vec3f& gravity,
                                        float              inv_dt2,
                                        std::uintptr_t     cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;
    if (friction_ <= 0.0f) return;

    const math::Vec4f* slip_ptr =
        (cached_has_slip_) ? slips_.gpu_data() : nullptr;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const float3 g3 = make_float3(gravity.x, gravity.y, gravity.z);
    apply_coulomb_friction_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        slip_ptr,
        n_particles,
        stiffness_,
        friction_,
        friction_epsilon_,
        thickness_,
        body_velocity_dev_.gpu_data(),
        mass,
        g3,
        inv_dt2,
        cached_dt_,
        rhs);
    check_cuda(cudaGetLastError(), "apply_coulomb_friction_kernel launch");
}

}  // namespace collision
}  // namespace chysx
