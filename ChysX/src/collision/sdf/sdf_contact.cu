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
// scatter_gradient injects penalty only (no implicit friction),
// so rhs = M/dt^2 * (x_tilde - x_n) - grad_elastic - k*depth*n.
//
// F0 = rhs - penalty_push  (inertia + elastic, no friction).
// F0_t = tangential component of F0.
//
// STICK:   replace rhs_t so  dx_t = v_body_t * dt.
//          target rhs_t = M * v_body_t / dt + M * g_t.
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
    float3                           gravity,
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
        // STICK: hard-pin tangential rhs so  dx_t = v_body_t * dt.
        //
        // H_t ~ M/dt^2  (penalty-only diagonal, no friction alpha).
        // target rhs_t = M * v_body / dt + M * g_t.
        const math::Vec3f vb = body_velocity_dev[0];
        const float vb_n = vb.x * nx + vb.y * ny + vb.z * nz;
        const float vbt_x = vb.x - vb_n * nx;
        const float vbt_y = vb.y - vb_n * ny;
        const float vbt_z = vb.z - vb_n * nz;

        const float m = mass[p];
        const float m_over_dt = m / fmaxf(dt, 1.0e-30f);

        const float gn = gravity.x * nx + gravity.y * ny + gravity.z * nz;
        const float gt_x = gravity.x - gn * nx;
        const float gt_y = gravity.y - gn * ny;
        const float gt_z = gravity.z - gn * nz;

        const float target_rhs_t_x = m_over_dt * vbt_x + m * gt_x;
        const float target_rhs_t_y = m_over_dt * vbt_y + m * gt_y;
        const float target_rhs_t_z = m_over_dt * vbt_z + m * gt_z;

        // Current tangential rhs -- preserve normal component.
        const float rhs_n = r.x * nx + r.y * ny + r.z * nz;
        const float rhs_t_x = r.x - rhs_n * nx;
        const float rhs_t_y = r.y - rhs_n * ny;
        const float rhs_t_z = r.z - rhs_n * nz;

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

    cached_dt_ = dt;

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
                                     std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    scatter_gradient_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
        rhs);
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

void SdfContact::bake_diag(math::Mat3f*   diag,
                           int             n_particles,
                           float           dt,
                           std::uintptr_t  cuda_stream) const {
    (void)dt;
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    bake_diag_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const float3 g3 = make_float3(gravity.x, gravity.y, gravity.z);
    apply_coulomb_friction_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
        friction_,
        thickness_,
        body_velocity_dev_.gpu_data(),
        mass,
        g3,
        cached_dt_,
        rhs);
    check_cuda(cudaGetLastError(), "apply_coulomb_friction_kernel launch");
}

}  // namespace collision
}  // namespace chysx
