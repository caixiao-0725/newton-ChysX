// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::SelfCollisionConstraint.

#include "self_collision_constraint.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../../collision/friction.cuh"

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::SelfCollisionConstraint: ") +
            what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Sign convention.  Penalty energy
//
//     E_c  =  (k/2) * (thickness - dot(g(x), n))^2,
//
// where `g(x) = sum_i w_i * x_i` is the contact's signed-distance
// constraint vector (== `x_v - x_face_cp` for a VF contact).  Then
//
//     d E_c / d x_j  =  -k * (thickness - dot(g, n)) * w_j * n
//                    =  -k * depth * w_j * n.
//
// chysx accumulates `+grad E` into `out_grad` and subtracts at the end
// (`assemble_rhs_kernel`), so we add `-k * depth * w_j * n` here.  The
// sign flips relative to cuda-cloth's `KernelComputeCollisionHessianAndForce_4`
// because cuda-cloth stores `+force = -grad E` directly in its `f`
// buffer.
__global__ void scatter_gradient_kernel(
    const math::Vec4i* __restrict__ pairs,
    const collision::ContactWeights* __restrict__ weights,
    const math::Vec4f* __restrict__ slips,        // may be nullptr
    const int* __restrict__ count_ptr,
    int max_contacts,
    float stiffness,
    float friction_mu,                            // 0 disables
    float friction_epsilon,
    math::Vec3f* __restrict__ out_grad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const collision::ContactWeights w = weights[c];

    // Normal gradient: -k * depth * n  (chysx stores +grad E and
    // assemble_rhs flips it at the end).
    const float kd = -stiffness * w.depth;
    const math::Vec3f g_normal(kd * w.nx, kd * w.ny, kd * w.nz);

    // Friction RHS contribution: -alpha_c * u_t^lag  per particle slot,
    // scaled by w_i below.  We compute alpha_c with the same formula
    // `bake_contact_diag_kernel` / `apply_contact_spmv_kernel` use so
    // the per-contact stiffness is consistent across all three passes.
    math::Vec3f g_friction(0.0f, 0.0f, 0.0f);
    if (friction_mu > 0.0f && slips != nullptr) {
        const float f_n = stiffness * w.depth;
        const math::Vec4f slip = slips[c];
        const float f1 = collision::ipc_f1_sf_over_x(slip.w, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        g_friction.x = -alpha * slip.x;
        g_friction.y = -alpha * slip.y;
        g_friction.z = -alpha * slip.z;
    }

    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float s = ws[i];
        if (s == 0.0f) continue;
        const int idx = idxs[i];
        atomicAdd(&out_grad[idx].x, s * (g_normal.x + g_friction.x));
        atomicAdd(&out_grad[idx].y, s * (g_normal.y + g_friction.y));
        atomicAdd(&out_grad[idx].z, s * (g_normal.z + g_friction.z));
    }
}

}  // namespace

collision::ContactSpMVOp SelfCollisionConstraint::make_spmv_op(
    const collision::SelfCollisionDetector& detector) const noexcept {
    collision::ContactSpMVOp op;
    op.pairs            = detector.pairs().gpu_data();
    op.weights          = detector.weights().gpu_data();
    op.count_dev        = detector.count_device_ptr();
    op.max_contacts     = detector.max_contacts();
    op.stiffness        = stiffness_;
    // Friction is "armed" only when the detector has actually run
    // `accumulate_slips()` this step (i.e. its slip cache is sized).
    // We hand the pointer through unconditionally; the consumer-side
    // `friction_active()` check guards against the slipless path.
    const bool slips_ready =
        detector.slips().gpu_size() == static_cast<std::size_t>(
            detector.max_contacts());
    op.slips            = slips_ready ? detector.slips().gpu_data() : nullptr;
    op.friction_mu      = friction_;
    op.friction_epsilon = friction_epsilon_;
    return op;
}

void SelfCollisionConstraint::accumulate_gradient(
    const collision::SelfCollisionDetector& detector,
    DeviceSpan<math::Vec3f> out_grad,
    std::uintptr_t cuda_stream) const {
    const int cap = detector.max_contacts();
    if (cap <= 0 || stiffness_ == 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const bool slips_ready =
        detector.slips().gpu_size() == static_cast<std::size_t>(cap);
    const math::Vec4f* slip_ptr =
        (friction_ > 0.0f && slips_ready) ? detector.slips().gpu_data() : nullptr;
    scatter_gradient_kernel<<<grid_for(cap), kBlockDim, 0, stream>>>(
        detector.pairs().gpu_data(),
        detector.weights().gpu_data(),
        slip_ptr,
        detector.count_device_ptr(),
        cap,
        stiffness_,
        friction_,
        friction_epsilon_,
        out_grad.data());
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

}  // namespace constraint
}  // namespace chysx
