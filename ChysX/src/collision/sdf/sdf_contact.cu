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
        const float f_n = stiffness * depth;
        const math::Vec4f slip = slips[p];
        const float u_norm = slip.w;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        // IPC lagged-Newton friction.  `slip = (v_p - v_body) · dt`
        // projected onto the tangent plane, i.e. the world-frame
        // tangential displacement the cloth particle WOULD undergo
        // this step *relative to the body* if no friction acted.
        // The linearised friction reaction wants to cancel that
        // relative motion, so it adds  +α · slip  to the *gradient*
        // (the negative of the friction force).  `assemble_rhs`
        // later flips the sign of the gradient when folding it into
        // the Newton residual; the net effect on the final RHS is
        //   rhs +=  -α · slip = +α · (v_body − v_p) · dt_t
        // which pulls the particle toward the body in the tangent
        // plane — exactly the direction needed to "stick" the cloth
        // to a moving SDF jaw.
        //
        // The previous version had this with the opposite sign,
        // which (for a stationary particle on an upward-moving jaw)
        // dragged the cloth *downward* and made the gripper unable
        // to lift its cargo.  See `example_chysx_sdf_gripper.py`
        // for the regression that surfaced this.
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
        const float f_n = k * depth;
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
// Mirrors static_contact's apply_coulomb_friction_kernel exactly.
__global__ void apply_coulomb_friction_kernel(
    const math::Vec4f* __restrict__ contacts,
    int                             n_particles,
    float                           stiffness,
    float                           friction_mu,
    math::Vec3f* __restrict__       rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;
    if (friction_mu <= 0.0f) return;

    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    const float fn_push = stiffness * depth;
    const math::Vec3f r = rhs[p];
    const math::Vec3f F0(r.x - fn_push * nx,
                         r.y - fn_push * ny,
                         r.z - fn_push * nz);

    const float F0_n_scalar = F0.x * nx + F0.y * ny + F0.z * nz;
    const float f_n_mag     = fmaxf(0.0f, -F0_n_scalar);
    if (f_n_mag <= 0.0f) return;

    const math::Vec3f F0_t(F0.x - F0_n_scalar * nx,
                           F0.y - F0_n_scalar * ny,
                           F0.z - F0_n_scalar * nz);
    const float F_T_sq = F0_t.x * F0_t.x + F0_t.y * F0_t.y + F0_t.z * F0_t.z;
    if (F_T_sq <= 0.0f) return;
    const float F_T = sqrtf(F_T_sq);

    const float cone = friction_mu * f_n_mag;
    // if (F_T > cone) {
    //     // SLIDING branch: cap the tangential force at the cone radius.
    //     // The IPC implicit friction baked in `scatter_gradient` /
    //     // `bake_diag` already drives sticking through an α·(I - nnᵀ)
    //     // tangential stiffness; capping here brings it back inside the
    //     // Coulomb cone when it overshoots.
    //     //
    //     // We deliberately do NOT zero the tangent in the STICK branch
    //     // (|F_T| ≤ cone).  Doing so was safe for the static-body
    //     // `StaticContactSet` (zero tangent == zero tangential
    //     // acceleration == particle pinned in place), but for an SDF
    //     // body with non-zero `body_velocity` the cloth is supposed to
    //     // accelerate tangentially WITH the body — that's the whole
    //     // point of friction in a moving-jaw scenario.  Zeroing rhs's
    //     // tangent component would silently kill the IPC friction force
    //     // that is doing the lifting, so the cloth never follows the
    //     // jaw upward.  Letting the IPC term flow through unchanged
    //     // when inside the cone gives the correct moving-body stick.
    //     const float reduce = 1.0f - cone / F_T;
    //     rhs[p].x -= reduce * F0_t.x;
    //     rhs[p].y -= reduce * F0_t.y;
    //     rhs[p].z -= reduce * F0_t.z;
    // }
    if (F_T <= cone) {
        // Legacy STICK branch: remove all tangential force.
        rhs[p].x -= F0_t.x;
        rhs[p].y -= F0_t.y;
        rhs[p].z -= F0_t.z;
    } else {
        // Legacy SLIDING branch: project onto cone boundary.
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
        diag);
    check_cuda(cudaGetLastError(), "bake_diag_kernel launch");
}

void SdfContact::apply_coulomb_friction(math::Vec3f*    rhs,
                                        int             n_particles,
                                        std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;
    if (friction_ <= 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    apply_coulomb_friction_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        n_particles,
        stiffness_,
        friction_,
        rhs);
    check_cuda(cudaGetLastError(), "apply_coulomb_friction_kernel launch");
}

}  // namespace collision
}  // namespace chysx
