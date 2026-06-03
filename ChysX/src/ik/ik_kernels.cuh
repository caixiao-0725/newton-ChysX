// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// IK utility CUDA kernels: batched FK, motion subspace, cost computation,
// sampling strategies, seed selection, and integration.

#pragma once

#include "../rigid/featherstone/joint_types.cuh"
#include "../rigid/featherstone/spatial_math.cuh"
#include <curand_kernel.h>

namespace chysx {
namespace ik {

using rigid::Transform7;
using rigid::SpatialVector;
using rigid::Vec3f;
using rigid::Quatf;
using rigid::jcalc_transform;
using rigid::jcalc_motion;
using rigid::jcalc_integrate;
using rigid::tf_identity;
using rigid::tf_inverse;
using rigid::tf_multiply;
using rigid::tf_point;
using rigid::tf_vector;
using rigid::transform_twist;

static constexpr int kBlockSize = 128;

// ============================================================================
// Batched FK Pass 1: compute local relative transforms per joint
// dim = [n_batch * joint_count], grid-stride
// ============================================================================

__global__ void ik_fk_local_kernel(
    int n_batch, int joint_count,
    const int* __restrict__ joint_type,
    const float* __restrict__ joint_q,       // [n_batch, n_coords]
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,   // [joint_count * 2]
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ joint_X_c,
    int n_coords,
    float* __restrict__ X_local_out          // [n_batch * joint_count * 7]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * joint_count;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / joint_count;
        int j   = idx % joint_count;

        int t          = joint_type[j];
        int q_start    = joint_q_start[j];
        int axis_start = joint_qd_start[j];
        int lin_axes   = joint_dof_dim[j * 2 + 0];
        int ang_axes   = joint_dof_dim[j * 2 + 1];

        const float* q_row = joint_q + row * n_coords;

        Transform7 X_j = jcalc_transform(t, joint_axis, axis_start, lin_axes, ang_axes, q_row, q_start);
        Transform7 X_rel = tf_multiply(tf_multiply(joint_X_p[j], X_j), tf_inverse(joint_X_c[j]));

        float* out = X_local_out + (row * joint_count + j) * 7;
        out[0] = X_rel.p.x; out[1] = X_rel.p.y; out[2] = X_rel.p.z;
        out[3] = X_rel.q.x; out[4] = X_rel.q.y; out[5] = X_rel.q.z; out[6] = X_rel.q.w;
    }
}

// ============================================================================
// Batched FK Pass 2: accumulate parent chain to get world transforms
// dim = [n_batch * joint_count]
// ============================================================================

__global__ void ik_fk_accum_kernel(
    int n_batch, int joint_count,
    const int* __restrict__ joint_parent,
    const float* __restrict__ X_local,       // [n_batch, joint_count, 7]
    float* __restrict__ body_q               // [n_batch, body_count, 7]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * joint_count;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / joint_count;
        int j   = idx % joint_count;

        const float* xl_base = X_local + row * joint_count * 7;

        const float* xl = xl_base + j * 7;
        Transform7 Xw(Vec3f(xl[0], xl[1], xl[2]), Quatf(xl[3], xl[4], xl[5], xl[6]));

        int parent = joint_parent[j];
        while (parent >= 0) {
            const float* xp = xl_base + parent * 7;
            Transform7 Xp(Vec3f(xp[0], xp[1], xp[2]), Quatf(xp[3], xp[4], xp[5], xp[6]));
            Xw = tf_multiply(Xp, Xw);
            parent = joint_parent[parent];
        }

        float* out = body_q + (row * joint_count + j) * 7;
        out[0] = Xw.p.x; out[1] = Xw.p.y; out[2] = Xw.p.z;
        out[3] = Xw.q.x; out[4] = Xw.q.y; out[5] = Xw.q.z; out[6] = Xw.q.w;
    }
}

// ============================================================================
// Batched motion subspace computation
// dim = [n_batch * joint_count]
// ============================================================================

__global__ void ik_motion_subspace_kernel(
    int n_batch, int joint_count, int n_dofs,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_qd_start,
    const float* __restrict__ joint_qd,      // [n_batch, n_dofs] — typically all-zero for IK
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,   // [joint_count * 2]
    const float* __restrict__ body_q,        // [n_batch, joint_count, 7]
    const Transform7* __restrict__ joint_X_p,
    float* __restrict__ joint_S_s            // [n_batch, n_dofs, 6]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * joint_count;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / joint_count;
        int j   = idx % joint_count;

        int type   = joint_type[j];
        int parent = joint_parent[j];
        int qd_start = joint_qd_start[j];
        int lin_axis_count = joint_dof_dim[j * 2 + 0];
        int ang_axis_count = joint_dof_dim[j * 2 + 1];

        Transform7 X_pj = joint_X_p[j];
        Transform7 X_wpj = X_pj;
        if (parent >= 0) {
            const float* bq = body_q + (row * joint_count + parent) * 7;
            Transform7 parent_tf(Vec3f(bq[0], bq[1], bq[2]), Quatf(bq[3], bq[4], bq[5], bq[6]));
            X_wpj = tf_multiply(parent_tf, X_pj);
        }

        const float* qd_row = joint_qd + row * n_dofs;
        SpatialVector* S_s_out = reinterpret_cast<SpatialVector*>(joint_S_s + row * n_dofs * 6);

        jcalc_motion(type, joint_axis, lin_axis_count, ang_axis_count,
                     X_wpj, qd_row, qd_start, S_s_out);
    }
}

// ============================================================================
// Compute costs: cost[row] = sum(residuals[row, :]^2)
// dim = n_batch
// ============================================================================

__global__ void ik_compute_costs_kernel(
    int n_batch, int n_residuals,
    const float* __restrict__ residuals,     // [n_batch, n_residuals]
    float* __restrict__ costs                // [n_batch]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float cost = 0.0f;
    const float* r = residuals + row * n_residuals;
    for (int i = 0; i < n_residuals; ++i) {
        cost += r[i] * r[i];
    }
    costs[row] = cost;
}

// ============================================================================
// Integration: apply dq_dof to joint_q via jcalc_integrate
// dim = [n_batch * joint_count]
// ============================================================================

__global__ void ik_integrate_dq_kernel(
    int n_batch, int joint_count, int n_coords, int n_dofs,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const int* __restrict__ joint_dof_dim,   // [joint_count * 2]
    const Transform7* __restrict__ joint_X_c,
    const Vec3f* __restrict__ body_com,
    const float* __restrict__ joint_q_curr,  // [n_batch, n_coords]
    const float* __restrict__ joint_qd_curr, // [n_batch, n_dofs]
    const float* __restrict__ dq_dof,        // [n_batch, n_dofs]
    float dt,
    float* __restrict__ joint_q_out,         // [n_batch, n_coords]
    float* __restrict__ joint_qd_out         // [n_batch, n_dofs]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * joint_count;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / joint_count;
        int j   = idx % joint_count;

        int t          = joint_type[j];
        int parent     = joint_parent[j];
        int child      = joint_child[j];
        int coord_start = joint_q_start[j];
        int dof_start  = joint_qd_start[j];
        int lin_axes   = joint_dof_dim[j * 2 + 0];
        int ang_axes   = joint_dof_dim[j * 2 + 1];

        const float* q_row   = joint_q_curr + row * n_coords;
        const float* qd_row  = joint_qd_curr + row * n_dofs;
        const float* dq_row  = dq_dof + row * n_dofs;
        float* q_out_row     = joint_q_out + row * n_coords;
        float* qd_out_row    = joint_qd_out + row * n_dofs;

        jcalc_integrate(parent, joint_X_c[j], body_com[child], t,
                        q_row, qd_row, dq_row,
                        coord_start, dof_start, lin_axes, ang_axes, dt,
                        q_out_row, qd_out_row);
    }
}

// ============================================================================
// Sampling kernels
// ============================================================================

__global__ void ik_sample_none_kernel(
    int n_expanded, int n_coords, int n_seeds,
    const float* __restrict__ joint_q_in,    // [n_problems, n_coords]
    float* __restrict__ joint_q_out          // [n_expanded, n_coords]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_expanded) return;
    int problem = tid / n_seeds;
    for (int c = 0; c < n_coords; ++c) {
        joint_q_out[tid * n_coords + c] = joint_q_in[problem * n_coords + c];
    }
}

__global__ void ik_sample_gauss_kernel(
    int n_expanded, int n_coords, int n_seeds,
    const float* __restrict__ joint_q_in,    // [n_problems, n_coords]
    const float* __restrict__ joint_lower,
    const float* __restrict__ joint_upper,
    const int* __restrict__ joint_bounded,
    float noise_std, unsigned int base_seed,
    float* __restrict__ joint_q_out          // [n_expanded, n_coords]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_expanded) return;
    int problem = tid / n_seeds;
    int seed_idx = tid % n_seeds;

    if (seed_idx == 0) {
        for (int c = 0; c < n_coords; ++c)
            joint_q_out[tid * n_coords + c] = joint_q_in[problem * n_coords + c];
        return;
    }

    curandState state;
    curand_init(base_seed, tid, 0, &state);

    for (int c = 0; c < n_coords; ++c) {
        float q = joint_q_in[problem * n_coords + c] + curand_normal(&state) * noise_std;
        if (joint_bounded[c]) {
            q = fminf(fmaxf(q, joint_lower[c]), joint_upper[c]);
        }
        joint_q_out[tid * n_coords + c] = q;
    }
}

__global__ void ik_sample_uniform_kernel(
    int n_expanded, int n_coords,
    const float* __restrict__ joint_lower,
    const float* __restrict__ joint_upper,
    const int* __restrict__ joint_bounded,
    unsigned int base_seed,
    float* __restrict__ joint_q_out          // [n_expanded, n_coords]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_expanded) return;

    curandState state;
    curand_init(base_seed, tid, 0, &state);

    for (int c = 0; c < n_coords; ++c) {
        if (joint_bounded[c]) {
            float u = curand_uniform(&state);
            joint_q_out[tid * n_coords + c] = joint_lower[c] + u * (joint_upper[c] - joint_lower[c]);
        } else {
            joint_q_out[tid * n_coords + c] = 0.0f;
        }
    }
}

__global__ void ik_sample_roberts_kernel(
    int n_expanded, int n_coords, int n_seeds,
    const float* __restrict__ joint_q_in,    // [n_problems, n_coords]
    const float* __restrict__ roberts_basis,
    const float* __restrict__ joint_lower,
    const float* __restrict__ joint_upper,
    const int* __restrict__ joint_bounded,
    float* __restrict__ joint_q_out          // [n_expanded, n_coords]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_expanded) return;
    int problem = tid / n_seeds;
    int seed_idx = tid % n_seeds;

    if (seed_idx == 0) {
        for (int c = 0; c < n_coords; ++c)
            joint_q_out[tid * n_coords + c] = joint_q_in[problem * n_coords + c];
        return;
    }

    for (int c = 0; c < n_coords; ++c) {
        if (joint_bounded[c]) {
            float lo = joint_lower[c];
            float hi = joint_upper[c];
            float r = fmodf(float(seed_idx) * roberts_basis[c], 1.0f);
            if (r < 0.0f) r += 1.0f;
            joint_q_out[tid * n_coords + c] = lo + r * (hi - lo);
        } else {
            joint_q_out[tid * n_coords + c] = joint_q_in[problem * n_coords + c];
        }
    }
}

// ============================================================================
// Select best seed per problem
// dim = n_problems
// ============================================================================

__global__ void ik_select_best_seed_kernel(
    int n_problems, int n_seeds,
    const float* __restrict__ costs,         // [n_expanded]
    int* __restrict__ best_indices           // [n_problems]
)
{
    int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= n_problems) return;

    int base = problem * n_seeds;
    float best_cost = costs[base];
    int best_idx = 0;
    for (int s = 1; s < n_seeds; ++s) {
        float c = costs[base + s];
        if (c < best_cost) {
            best_cost = c;
            best_idx = s;
        }
    }
    best_indices[problem] = best_idx;
}

__global__ void ik_gather_best_seed_kernel(
    int n_problems, int n_seeds, int n_coords,
    const float* __restrict__ joint_q_expanded, // [n_expanded, n_coords]
    const int* __restrict__ best_indices,       // [n_problems]
    float* __restrict__ joint_q_out             // [n_problems, n_coords]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_problems * n_coords;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int problem = idx / n_coords;
        int c = idx % n_coords;
        int src_row = problem * n_seeds + best_indices[problem];
        joint_q_out[problem * n_coords + c] = joint_q_expanded[src_row * n_coords + c];
    }
}

// ============================================================================
// Copy 2D batch row to flat array or vice versa
// ============================================================================

__global__ void ik_fill_problem_idx_kernel(
    int n_expanded, int n_seeds,
    int* __restrict__ problem_idx             // [n_expanded]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_expanded) return;
    problem_idx[tid] = tid / n_seeds;
}

}  // namespace ik
}  // namespace chysx
