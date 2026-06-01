// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::apply_contact_spmv.

#include "contact_spmv.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "friction.cuh"
#include "zero_count.h"

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::apply_contact_spmv: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// One thread per contact; each thread does up to 4 atomicAdd-mat3
// updates into `diag_blocks`.  Mirrors cuda-cloth's
// `KernelComputeCollisionHessianAndForce_4` MINUS the force-side write
// (chysx folds the force into the RHS via the gradient pathway --
// `SelfCollisionConstraint::accumulate_gradient` -- so we only bake
// the Hessian-diagonal half here).
//
// IPC friction (when `slips != nullptr && friction_mu > 0`) is merged
// in here on top of the normal-stiffness block:
//
//   H_ii  +=  w_i^2 · ( k · (n n^T)  +  α_c · (I - n n^T) )
//
// with `α_c = μ · k · depth_c · f1_SF_over_x(‖u_t,c^lag‖)`.  This
// reuses the same `pairs[c]` / `weights[c]` loads -- only one extra
// `Vec4f` load per contact (the slip cache).
__global__ void bake_contact_diag_kernel(
    const math::Vec4i* __restrict__ pairs,
    const ContactWeights* __restrict__ weights,
    const math::Vec4f* __restrict__ slips,           // may be nullptr
    const int* __restrict__ count_ptr,
    int max_contacts,
    float k_alpha,
    float friction_mu,                                // 0 disables
    float friction_epsilon,
    math::Mat3f* __restrict__ diag_blocks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const ContactWeights w = weights[c];
    const float nx = w.nx, ny = w.ny, nz = w.nz;
    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    // IPC friction stiffness alpha_c = mu · k · depth · f1_SF_over_x(‖u_t^lag‖).
    // Zero when friction is off or no slip cache; the friction branch
    // then collapses into the normal-only block (no extra arithmetic
    // beyond a couple of FMAs against zero, no extra atomics).
    float alpha_friction = 0.0f;
    if (friction_mu > 0.0f && slips != nullptr) {
        const float f_n = k_alpha * w.depth;
        const float u_norm = slips[c].w;
        const float f1 = ipc_f1_sf_over_x(u_norm, friction_epsilon);
        alpha_friction = friction_mu * f_n * f1;
    }

    // Build the per-pair 3x3 block H_c = k_alpha · (n n^T) + alpha_friction
    // · (I - n n^T) once in registers.  Only the six upper-triangle
    // entries are unique (symmetric).
    //
    //   k_alpha · (n n^T)_ii  + alpha · (1 - n_i^2)
    //   k_alpha · (n n^T)_ij  + alpha · (- n_i n_j)         (i != j)
    //
    // i.e. each entry is `(k_alpha - alpha) · n_i n_j` off-diagonal,
    // and `(k_alpha - alpha) · n_i^2 + alpha` on the diagonal.
    const float diff   = k_alpha - alpha_friction;
    const float h00 = diff * nx * nx + alpha_friction;
    const float h11 = diff * ny * ny + alpha_friction;
    const float h22 = diff * nz * nz + alpha_friction;
    const float h01 = diff * nx * ny;
    const float h02 = diff * nx * nz;
    const float h12 = diff * ny * nz;

    // Per-particle diagonal contribution: w_i^2 · H_c.  Accumulate the
    // nine 3x3 entries entirely in registers so each of the four
    // particle slots costs exactly 9 atomicAdds (vs. 18 when normal
    // and friction were emitted as separate scatter passes).
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float ww = ws[i] * ws[i];
        if (ww == 0.0f) continue;
        const int idx = idxs[i];
        float* dst = diag_blocks[idx].data;
        const float m00 = ww * h00;
        const float m11 = ww * h11;
        const float m22 = ww * h22;
        const float m01 = ww * h01;
        const float m02 = ww * h02;
        const float m12 = ww * h12;
        atomicAdd(&dst[0], m00);
        atomicAdd(&dst[1], m01);
        atomicAdd(&dst[2], m02);
        atomicAdd(&dst[3], m01);  // symmetric
        atomicAdd(&dst[4], m11);
        atomicAdd(&dst[5], m12);
        atomicAdd(&dst[6], m02);  // symmetric
        atomicAdd(&dst[7], m12);  // symmetric
        atomicAdd(&dst[8], m22);
    }
}

// One thread per contact; each thread does the FULL 4x4 minus its
// diagonal.  Layout mirrors cuda-cloth's `CollisionSpmv_4`
// (SolverUtils.cu): for every i in 0..3 sum_{j != i} w_i*w_j*x[id_j],
// then accumulate `H * temp` into `y[id_i]` with `H = k * (n n^T) +
// alpha_friction * (I - n n^T)`.  The two blocks share the same
// `temp` so the friction add is essentially "free" once we've already
// done the gather.
//
// The diagonal `i == j` term is INTENTIONALLY dropped because the
// caller already baked it into the BlockCSR3's `diag` array via
// `bake_contact_diag` -- it's covered by the regular CSR SpMV.
__global__ void apply_contact_spmv_kernel(
    const math::Vec4i* __restrict__ pairs,
    const ContactWeights* __restrict__ weights,
    const math::Vec4f* __restrict__ slips,         // may be nullptr
    const int* __restrict__ count_ptr,
    int max_contacts,
    float k_alpha,
    float friction_mu,                              // 0 disables
    float friction_epsilon,
    const math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const ContactWeights w = weights[c];
    const math::Vec3f normal(w.nx, w.ny, w.nz);
    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    // Per-contact friction stiffness (same formula as
    // `bake_contact_diag_kernel`).  Zero when slips are unavailable or
    // mu is off, which collapses the kernel back to the original
    // normal-only SpMV (no extra atomics emitted).
    float alpha_friction = 0.0f;
    if (friction_mu > 0.0f && slips != nullptr) {
        const float f_n = k_alpha * w.depth;
        const float u_norm = slips[c].w;
        const float f1 = ipc_f1_sf_over_x(u_norm, friction_epsilon);
        alpha_friction = friction_mu * f_n * f1;
    }

    // Cache neighbour positions once -- four reads, then four reuses.
    const math::Vec3f xs[4] = {
        x[idxs[0]], x[idxs[1]], x[idxs[2]], x[idxs[3]],
    };

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float wi = ws[i];
        if (wi == 0.0f) continue;
        // temp = sum_{j != i} w_j * x_j
        float tx = 0.0f, ty = 0.0f, tz = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j == i) continue;
            const float wj = ws[j];
            tx += wj * xs[j].x;
            ty += wj * xs[j].y;
            tz += wj * xs[j].z;
        }
        // Normal: H_n * temp = k * (n . temp) * n
        // Friction: H_f * temp = alpha * (temp - (n . temp) * n)
        // Combined: (k_alpha - alpha) * (n . temp) * n + alpha * temp
        const float dn = normal.x * tx + normal.y * ty + normal.z * tz;
        const float n_coef = wi * (k_alpha - alpha_friction) * dn;
        const float wi_alpha = wi * alpha_friction;
        const float ax = n_coef * normal.x + wi_alpha * tx;
        const float ay = n_coef * normal.y + wi_alpha * ty;
        const float az = n_coef * normal.z + wi_alpha * tz;
        atomicAdd(&y[idxs[i]].x, ax);
        atomicAdd(&y[idxs[i]].y, ay);
        atomicAdd(&y[idxs[i]].z, az);
    }
}

}  // namespace

void bake_contact_diag(math::Mat3f* diag_blocks,
                       int /*n_particles*/,
                       const ContactSpMVOp& op,
                       float alpha,
                       std::uintptr_t cuda_stream) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const int launch_size = (op.max_contacts > 0) ? op.max_contacts : 1;
    const int* count_ptr  = op.count_dev ? op.count_dev : zero_count_ptr();

    const float k_alpha = alpha * op.stiffness;
    const float fric_mu =
        op.friction_active() ? op.friction_mu : 0.0f;
    const math::Vec4f* slip_ptr =
        op.friction_active() ? op.slips : nullptr;
    bake_contact_diag_kernel<<<grid_for(launch_size), kBlockDim, 0, stream>>>(
        op.pairs,
        op.weights,
        slip_ptr,
        count_ptr,
        op.max_contacts,
        k_alpha,
        fric_mu,
        op.friction_epsilon,
        diag_blocks);
    check_cuda(cudaGetLastError(), "bake_contact_diag_kernel launch");
}

void apply_contact_spmv(const ContactSpMVOp& op,
                        const math::Vec3f* x,
                        math::Vec3f* y,
                        int /*n_particles*/,
                        float alpha,
                        std::uintptr_t cuda_stream) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    // Always launch at least 1 block so the kernel sequence is
    // identical with or without active contacts — keeps a surrounding
    // CUDA Graph capture valid across frames.  When `max_contacts` is
    // 0 (or count_dev reads 0) every thread exits immediately via the
    // `c >= n` guard inside the kernel.
    const int launch_size = (op.max_contacts > 0) ? op.max_contacts : 1;
    const int* count_ptr  = op.count_dev ? op.count_dev : zero_count_ptr();

    const float k_alpha = alpha * op.stiffness;
    const float fric_mu =
        op.friction_active() ? op.friction_mu : 0.0f;
    const math::Vec4f* slip_ptr =
        op.friction_active() ? op.slips : nullptr;
    apply_contact_spmv_kernel<<<grid_for(launch_size), kBlockDim, 0, stream>>>(
        op.pairs,
        op.weights,
        slip_ptr,
        count_ptr,
        op.max_contacts,
        k_alpha,
        fric_mu,
        op.friction_epsilon,
        x,
        y);
    check_cuda(cudaGetLastError(), "apply_contact_spmv_kernel launch");
}

}  // namespace collision
}  // namespace chysx
