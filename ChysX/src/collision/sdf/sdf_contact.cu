// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of `SdfContact`.
//
// One thread per particle; the per-particle work is to trilinear-sample
// the bound `SdfVolume`, cache (depth, normal) in a per-particle Vec4f
// `(nx, ny, nz, depth)`, then have follow-up scatter passes push the
// resulting penalty contributions into the cloth simulator's `rhs` and
// `H_.diag` arrays.
//
// The scatter / diag kernels inject penalty-only forces.  All friction
// is handled by the Coulomb-cone post-projection kernel
// (`apply_coulomb_friction_kernel`) which directly pins/projects the
// assembled Newton residual -- no implicit friction stiffness needed.

#include "sdf_contact.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <stdexcept>
#include <string>

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
// - sd).  depth <= 0 means "no active contact" and the downstream
// scatter passes skip it.
__global__ void detect_kernel(
    const float3* __restrict__       positions,
    int                              n_particles,
    SdfVolumeView                    view,
    float                            thickness,
    math::Vec4f* __restrict__        contacts) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const float3 pf = positions[p];
    const math::Vec3f x(pf.x, pf.y, pf.z);

    float       sd;
    math::Vec3f grad;
    view.sample(x, sd, grad);

    float depth = thickness - sd;

    math::Vec3f n(0.0f, 0.0f, 0.0f);
    if (depth > 0.0f) {
        const float g2 = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
        const float inv_g = rsqrtf(fmaxf(g2, 1.0e-30f));
        n.x = grad.x * inv_g;
        n.y = grad.y * inv_g;
        n.z = grad.z * inv_g;
    } else {
        depth = 0.0f;
    }

    contacts[p] = math::Vec4f(n.x, n.y, n.z, depth);
}

// =====================================================================
// IPC-style implicit friction kernels.
//
// Matches VBD's `_compute_body_particle_contact_force`:
//   relative_translation = (pos - prev_pos) - v_body * dt
//   u_t = tangential projection of relative_translation
//   f_friction, K_friction = compute_projected_isotropic_friction(
//       mu, f_n, n, u_t, eps_u * dt)
//
// These kernels inject BOTH penalty + friction into rhs and diag,
// so when IPC mode is active the Coulomb post-projection is skipped.
// =====================================================================

// IPC gradient: rhs[p] += -k * depth * n  +  f_friction(u_t)
// u_t is computed from lagged velocity: (v_particle - v_body) * dt
// projected onto the contact tangent plane.
__global__ void scatter_gradient_ipc_kernel(
    const math::Vec4f* __restrict__  contacts,
    int                              n_particles,
    float                            stiffness,
    float                            friction_mu,
    float                            friction_epsilon,
    float                            dt,
    const float3* __restrict__       velocities,
    const math::Vec3f* __restrict__  body_velocity_dev,
    float                            contact_kd,
    math::Vec3f* __restrict__        rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float nx = c.x, ny = c.y, nz = c.z;

    const float f_n = stiffness * depth;
    rhs[p].x += -f_n * nx;
    rhs[p].y += -f_n * ny;
    rhs[p].z += -f_n * nz;

    // Lagged relative displacement: (v_particle - v_body) * dt
    const float3 vp = velocities[p];
    const math::Vec3f vb = body_velocity_dev[0];
    const float rel_x = (vp.x - vb.x) * dt;
    const float rel_y = (vp.y - vb.y) * dt;
    const float rel_z = (vp.z - vb.z) * dt;

    const float rel_dot_n = rel_x * nx + rel_y * ny + rel_z * nz;

    // Contact damping
    if (contact_kd > 0.0f && rel_dot_n < 0.0f) {
        const float damp = contact_kd * stiffness;
        rhs[p].x -= damp * rel_dot_n * nx;
        rhs[p].y -= damp * rel_dot_n * ny;
        rhs[p].z -= damp * rel_dot_n * nz;
    }

    // Tangential slip
    const float ut_x = rel_x - rel_dot_n * nx;
    const float ut_y = rel_y - rel_dot_n * ny;
    const float ut_z = rel_z - rel_dot_n * nz;
    const float u_norm = sqrtf(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z);

    if (friction_mu > 0.0f && f_n > 0.0f && u_norm > 0.0f) {
        const float eps_u = friction_epsilon * dt;
        float f1_sf_over_x;
        if (u_norm > eps_u) {
            f1_sf_over_x = 1.0f / u_norm;
        } else {
            f1_sf_over_x = (-u_norm / eps_u + 2.0f) / eps_u;
        }
        const float scale = friction_mu * f_n * f1_sf_over_x;
        rhs[p].x -= scale * ut_x;
        rhs[p].y -= scale * ut_y;
        rhs[p].z -= scale * ut_z;
    }
}

// IPC Hessian diagonal: diag[p] += k*(nn^T) + damping Hessian + friction Hessian
__global__ void bake_diag_ipc_kernel(
    const math::Vec4f* __restrict__  contacts,
    int                              n_particles,
    float                            stiffness,
    float                            friction_mu,
    float                            friction_epsilon,
    float                            dt,
    const float3* __restrict__       velocities,
    const math::Vec3f* __restrict__  body_velocity_dev,
    float                            contact_kd,
    math::Mat3f* __restrict__        diag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float k = stiffness;
    const float nx = c.x, ny = c.y, nz = c.z;

    math::Mat3f& A = diag[p];
    A.data[0] += k * nx * nx;
    A.data[1] += k * nx * ny;
    A.data[2] += k * nx * nz;
    A.data[3] += k * nx * ny;
    A.data[4] += k * ny * ny;
    A.data[5] += k * ny * nz;
    A.data[6] += k * nx * nz;
    A.data[7] += k * ny * nz;
    A.data[8] += k * nz * nz;

    const float3 vp = velocities[p];
    const math::Vec3f vb = body_velocity_dev[0];
    const float rel_x = (vp.x - vb.x) * dt;
    const float rel_y = (vp.y - vb.y) * dt;
    const float rel_z = (vp.z - vb.z) * dt;
    const float rel_dot_n = rel_x * nx + rel_y * ny + rel_z * nz;

    if (contact_kd > 0.0f && rel_dot_n < 0.0f) {
        const float damp_coeff = contact_kd * k / fmaxf(dt, 1e-30f);
        A.data[0] += damp_coeff * nx * nx;
        A.data[1] += damp_coeff * nx * ny;
        A.data[2] += damp_coeff * nx * nz;
        A.data[3] += damp_coeff * nx * ny;
        A.data[4] += damp_coeff * ny * ny;
        A.data[5] += damp_coeff * ny * nz;
        A.data[6] += damp_coeff * nx * nz;
        A.data[7] += damp_coeff * ny * nz;
        A.data[8] += damp_coeff * nz * nz;
    }

    const float f_n = k * depth;
    if (friction_mu > 0.0f && f_n > 0.0f) {
        const float ut_x = rel_x - rel_dot_n * nx;
        const float ut_y = rel_y - rel_dot_n * ny;
        const float ut_z = rel_z - rel_dot_n * nz;
        const float u_norm = sqrtf(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z);

        if (u_norm > 0.0f) {
            const float eps_u = friction_epsilon * dt;
            float f1_sf_over_x;
            if (u_norm > eps_u) {
                f1_sf_over_x = 1.0f / u_norm;
            } else {
                f1_sf_over_x = (-u_norm / eps_u + 2.0f) / eps_u;
            }
            const float scale = friction_mu * f_n * f1_sf_over_x;
            A.data[0] += scale * (1.0f - nx * nx);
            A.data[1] += scale * (0.0f - nx * ny);
            A.data[2] += scale * (0.0f - nx * nz);
            A.data[3] += scale * (0.0f - nx * ny);
            A.data[4] += scale * (1.0f - ny * ny);
            A.data[5] += scale * (0.0f - ny * nz);
            A.data[6] += scale * (0.0f - nx * nz);
            A.data[7] += scale * (0.0f - ny * nz);
            A.data[8] += scale * (1.0f - nz * nz);
        }
    }
}

// =====================================================================
// Original penalty-only kernels (used by Coulomb post-projection mode).
// =====================================================================

// rhs[p] += -k * depth * n.
// Penalty-only gradient (no implicit friction -- all friction is handled
// by the Coulomb post-projection in apply_coulomb_friction_kernel).
__global__ void scatter_gradient_kernel(
    const math::Vec4f* __restrict__ contacts,
    int                             n_particles,
    float                           stiffness,
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
}

// diag[p] += k * (n n^T).
// Penalty-only Hessian diagonal (no implicit friction stiffness).
__global__ void bake_diag_kernel(
    const math::Vec4f* __restrict__ contacts,
    int                             n_particles,
    float                           stiffness,
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

    math::Mat3f& A = diag[p];
    A.data[0] += k * nx * nx;
    A.data[1] += k * nx * ny;
    A.data[2] += k * nx * nz;
    A.data[3] += k * nx * ny;
    A.data[4] += k * ny * ny;
    A.data[5] += k * ny * nz;
    A.data[6] += k * nx * nz;
    A.data[7] += k * ny * nz;
    A.data[8] += k * nz * nz;
}

// Coulomb-cone post-projection on the assembled Newton residual.
//
// F0 = rhs - penalty_push  (inertia + elastic, no friction).
// F0_t = tangential component of F0.
//
// STICK:   add impulse  M * (v_body_t - v_particle_t) / dt  to rhs.
//          This redirects the inertia term from v_particle to v_body
//          while preserving all elastic and gravity forces.
// SLIDING: project F0_t onto Coulomb-cone boundary  mu * f_n.
//
// Cone uses  max(depth, 0.1*thickness)  so shallow-penetration
// particles still experience meaningful friction.
__global__ void apply_coulomb_friction_kernel(
    const math::Vec4f* __restrict__  contacts,
    int                              n_particles,
    float                            stiffness,
    float                            friction_mu,
    float                            thickness,
    const math::Vec3f* __restrict__  body_velocity_dev,
    const float* __restrict__        mass,
    const float3* __restrict__       velocities,
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

    // F0 = rhs - penalty push.  No friction was injected by scatter.
    const float fn_push = stiffness * depth;
    const math::Vec3f r = rhs[p];
    const math::Vec3f F0(r.x - fn_push * nx,
                         r.y - fn_push * ny,
                         r.z - fn_push * nz);

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
        // STICK: add an inertia-redirect impulse so the solver drives
        // this particle's tangential velocity toward v_body_t.
        //   delta = M * (v_body_t - v_n_t) / dt
        const math::Vec3f vb = body_velocity_dev[0];
        const float3 vp = velocities[p];

        const float vb_n = vb.x * nx + vb.y * ny + vb.z * nz;
        const float vp_n = vp.x * nx + vp.y * ny + vp.z * nz;
        const float dvt_x = (vb.x - vb_n * nx) - (vp.x - vp_n * nx);
        const float dvt_y = (vb.y - vb_n * ny) - (vp.y - vp_n * ny);
        const float dvt_z = (vb.z - vb_n * nz) - (vp.z - vp_n * nz);

        const float m = mass[p];
        const float m_over_dt = m / fmaxf(dt, 1.0e-30f);

        rhs[p].x += m_over_dt * dvt_x;
        rhs[p].y += m_over_dt * dvt_y;
        rhs[p].z += m_over_dt * dvt_z;
    } else {
        // SLIDING: project onto cone boundary.
        const float shrink = cone / F_T;
        rhs[p].x -= shrink * F0_t.x;
        rhs[p].y -= shrink * F0_t.y;
        rhs[p].z -= shrink * F0_t.z;
    }
}

// Inject a large tangential penalty for STICK particles into both
// diag (Hessian diagonal) and rhs, acting as a soft Dirichlet BC:
//
//   diag_t += k_stick * (I - nn^T)
//   rhs_t  += k_stick * v_body_t * dt
//
// The STICK/SLIDE decision reuses the same contacts / cone logic.
// `k_stick` is set to M/dt^2 so it roughly doubles the inertia weight
// in the tangential direction.
__global__ void bake_stick_constraint_kernel(
    const math::Vec4f* __restrict__  contacts,
    int                              n_particles,
    float                            stiffness,
    float                            friction_mu,
    float                            thickness,
    const math::Vec3f* __restrict__  body_velocity_dev,
    const float* __restrict__        mass,
    const float3* __restrict__       velocities,
    float                            dt,
    math::Vec3f* __restrict__        rhs,
    math::Mat3f* __restrict__        diag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;
    if (friction_mu <= 0.0f) return;

    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    // Recompute F0 / cone to decide STICK vs SLIDE.
    const float fn_push = stiffness * depth;
    const math::Vec3f r = rhs[p];
    const math::Vec3f F0(r.x - fn_push * nx,
                         r.y - fn_push * ny,
                         r.z - fn_push * nz);
    const float F0_n_scalar = F0.x * nx + F0.y * ny + F0.z * nz;
    const float depth_for_cone = fmaxf(depth, 0.1f * thickness);
    const float cone = friction_mu * stiffness * depth_for_cone;

    const math::Vec3f F0_t(F0.x - F0_n_scalar * nx,
                           F0.y - F0_n_scalar * ny,
                           F0.z - F0_n_scalar * nz);
    const float F_T_sq = F0_t.x * F0_t.x + F0_t.y * F0_t.y + F0_t.z * F0_t.z;
    if (F_T_sq <= 0.0f) return;
    const float F_T = sqrtf(F_T_sq);

    if (F_T > cone) return;  // SLIDING: handled by apply_coulomb_friction

    // STICK: inject a penalty spring driving dx_t toward v_body_t * dt.
    const float m = mass[p];
    const float inv_dt = 1.0f / fmaxf(dt, 1.0e-30f);
    const float k_stick = 1000.0f * m * inv_dt * inv_dt;

    const math::Vec3f vb = body_velocity_dev[0];
    const float vb_n = vb.x * nx + vb.y * ny + vb.z * nz;
    const float dx_target_x = (vb.x - vb_n * nx) * dt;
    const float dx_target_y = (vb.y - vb_n * ny) * dt;
    const float dx_target_z = (vb.z - vb_n * nz) * dt;

    // rhs += k_stick * dx_target  (tangential only)
    rhs[p].x += k_stick * dx_target_x;
    rhs[p].y += k_stick * dx_target_y;
    rhs[p].z += k_stick * dx_target_z;

    // diag += k_stick * (I - nn^T)  (tangential projector)
    math::Mat3f& A = diag[p];
    A.data[0] += k_stick * (1.0f - nx * nx);
    A.data[1] += k_stick * (0.0f - nx * ny);
    A.data[2] += k_stick * (0.0f - nx * nz);
    A.data[3] += k_stick * (0.0f - nx * ny);
    A.data[4] += k_stick * (1.0f - ny * ny);
    A.data[5] += k_stick * (0.0f - ny * nz);
    A.data[6] += k_stick * (0.0f - nx * nz);
    A.data[7] += k_stick * (0.0f - ny * nz);
    A.data[8] += k_stick * (1.0f - nz * nz);
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

    cached_dt_ = dt;
    cached_velocities_ = velocities;

    SdfVolumeView view = volume_->make_view();

    if (body_velocity_dev_.gpu_size() != 1u) {
        body_velocity_dev_.resize(1u);
        body_velocity_dev_[0] = body_velocity_;
        body_velocity_dev_.copy_to_device(cuda_stream);
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    detect_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        reinterpret_cast<const float3*>(positions),
        n_particles,
        view,
        thickness_,
        contacts_.gpu_data());
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
                                     std::uintptr_t  cuda_stream,
                                     const math::Vec3f* /*positions*/,
                                     const math::Vec3f* /*prev_positions*/,
                                     float           dt) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    if (ipc_friction_ && cached_velocities_ && dt > 0.0f) {
        scatter_gradient_ipc_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(),
            n_particles,
            stiffness_,
            friction_,
            friction_epsilon_,
            dt,
            reinterpret_cast<const float3*>(cached_velocities_),
            body_velocity_dev_.gpu_data(),
            contact_kd_,
            rhs);
        check_cuda(cudaGetLastError(), "scatter_gradient_ipc_kernel launch");
    } else {
        scatter_gradient_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(),
            n_particles,
            stiffness_,
            rhs);
        check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
    }
}

void SdfContact::bake_diag(math::Mat3f*   diag,
                           int             n_particles,
                           float           dt,
                           std::uintptr_t  cuda_stream,
                           const math::Vec3f* /*positions*/,
                           const math::Vec3f* /*prev_positions*/) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    if (ipc_friction_ && cached_velocities_ && dt > 0.0f) {
        bake_diag_ipc_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(),
            n_particles,
            stiffness_,
            friction_,
            friction_epsilon_,
            dt,
            reinterpret_cast<const float3*>(cached_velocities_),
            body_velocity_dev_.gpu_data(),
            contact_kd_,
            diag);
        check_cuda(cudaGetLastError(), "bake_diag_ipc_kernel launch");
    } else {
        bake_diag_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(),
            n_particles,
            stiffness_,
            diag);
        check_cuda(cudaGetLastError(), "bake_diag_kernel launch");
    }
}

void SdfContact::apply_coulomb_friction(math::Vec3f*       rhs,
                                        int                n_particles,
                                        const float*       mass,
                                        const math::Mat3f* diag,
                                        const math::Vec3f& gravity,
                                        float              inv_dt2,
                                        std::uintptr_t     cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;
    if (friction_ <= 0.0f) return;
    if (cached_velocities_ == nullptr) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    apply_coulomb_friction_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
        friction_,
        thickness_,
        body_velocity_dev_.gpu_data(),
        mass,
        reinterpret_cast<const float3*>(cached_velocities_),
        cached_dt_,
        rhs);
    check_cuda(cudaGetLastError(), "apply_coulomb_friction_kernel launch");
}

void SdfContact::bake_stick_constraint(math::Vec3f*       rhs,
                                       math::Mat3f*       diag,
                                       int                n_particles,
                                       const float*       mass,
                                       std::uintptr_t     cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;
    if (friction_ <= 0.0f) return;
    if (cached_velocities_ == nullptr) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bake_stick_constraint_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
        friction_,
        thickness_,
        body_velocity_dev_.gpu_data(),
        mass,
        reinterpret_cast<const float3*>(cached_velocities_),
        cached_dt_,
        rhs,
        diag);
    check_cuda(cudaGetLastError(), "bake_stick_constraint_kernel launch");
}

}  // namespace collision
}  // namespace chysx
