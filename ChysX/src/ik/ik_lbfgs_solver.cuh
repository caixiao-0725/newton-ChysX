// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// L-BFGS IK solver CUDA kernels.
// Implements two-loop recursion for search direction,
// parallel Wolfe line search, and history management.

#pragma once

#include "ik_lm_solver.cuh"   // for kMaxDofs

namespace chysx {
namespace ik {

// ============================================================================
// Compute gradient: g = J^T * r (Gauss-Newton gradient)
// One thread per batch row.
// ============================================================================

__global__ void ik_compute_gradient_kernel(
    int n_batch, int n_dofs, int n_residuals,
    const float* __restrict__ jacobian,      // [n_batch, n_residuals, n_dofs]
    const float* __restrict__ residuals,     // [n_batch, n_residuals]
    float* __restrict__ gradient             // [n_batch, n_dofs]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    const float* J = jacobian + row * n_residuals * n_dofs;
    const float* r = residuals + row * n_residuals;
    float* g = gradient + row * n_dofs;

    for (int i = 0; i < n_dofs; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < n_residuals; ++k) {
            sum += J[k * n_dofs + i] * r[k];
        }
        g[i] = sum;
    }
}

// ============================================================================
// L-BFGS two-loop recursion: compute search direction from gradient + history
// One thread per batch row.
// ============================================================================

__global__ void ik_lbfgs_search_direction_kernel(
    int n_batch, int n_dofs, int history_len,
    const float* __restrict__ gradient,        // [n_batch, n_dofs]
    const float* __restrict__ s_history,       // [n_batch, history_len, n_dofs]
    const float* __restrict__ y_history,       // [n_batch, history_len, n_dofs]
    const float* __restrict__ rho_history,     // [n_batch, history_len]
    const int* __restrict__ history_count,     // [n_batch]
    const int* __restrict__ history_start,     // [n_batch]
    float h0_scale,
    float* __restrict__ alpha_scratch,         // [n_batch, history_len]
    float* __restrict__ search_direction       // [n_batch, n_dofs]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float q[kMaxDofs];
    int count = history_count[row];
    int start = history_start[row];

    const float* g = gradient + row * n_dofs;
    for (int j = 0; j < n_dofs; ++j) q[j] = g[j];

    float* alpha = alpha_scratch + row * history_len;

    // First loop: backward through history
    for (int i = 0; i < count; ++i) {
        int idx = (start + count - 1 - i) % history_len;
        const float* s_i = s_history + (row * history_len + idx) * n_dofs;
        const float* y_i = y_history + (row * history_len + idx) * n_dofs;
        float rho_i = rho_history[row * history_len + idx];

        float s_dot_q = 0.0f;
        for (int j = 0; j < n_dofs; ++j) s_dot_q += s_i[j] * q[j];
        float alpha_i = rho_i * s_dot_q;
        alpha[idx] = alpha_i;

        for (int j = 0; j < n_dofs; ++j) q[j] -= alpha_i * y_i[j];
    }

    // Apply initial Hessian approximation
    for (int j = 0; j < n_dofs; ++j) q[j] *= h0_scale;

    // Second loop: forward through history
    for (int i = 0; i < count; ++i) {
        int idx = (start + i) % history_len;
        const float* y_i = y_history + (row * history_len + idx) * n_dofs;
        const float* s_i = s_history + (row * history_len + idx) * n_dofs;
        float rho_i = rho_history[row * history_len + idx];
        float alpha_i = alpha[idx];

        float y_dot_q = 0.0f;
        for (int j = 0; j < n_dofs; ++j) y_dot_q += y_i[j] * q[j];
        float beta = rho_i * y_dot_q;
        float diff = alpha_i - beta;

        for (int j = 0; j < n_dofs; ++j) q[j] += diff * s_i[j];
    }

    // Store negative direction (descent)
    float* d = search_direction + row * n_dofs;
    for (int j = 0; j < n_dofs; ++j) d[j] = -q[j];
}

// ============================================================================
// Update L-BFGS history: store s_k = last_step, y_k = g_curr - g_prev
// dim = n_batch
// ============================================================================

__global__ void ik_lbfgs_update_history_kernel(
    int n_batch, int n_dofs, int history_len,
    const float* __restrict__ last_step,       // [n_batch, n_dofs]
    const float* __restrict__ gradient,        // [n_batch, n_dofs]
    const float* __restrict__ gradient_prev,   // [n_batch, n_dofs]
    float* __restrict__ s_history,             // [n_batch, history_len, n_dofs]
    float* __restrict__ y_history,             // [n_batch, history_len, n_dofs]
    float* __restrict__ rho_history,           // [n_batch, history_len]
    int* __restrict__ history_count,
    int* __restrict__ history_start
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    const float* s_k = last_step + row * n_dofs;
    const float* g_curr = gradient + row * n_dofs;
    const float* g_prev = gradient_prev + row * n_dofs;

    // y_k = g_curr - g_prev, and y_dot_s
    float y_k[kMaxDofs];
    float y_dot_s = 0.0f;
    for (int j = 0; j < n_dofs; ++j) {
        y_k[j] = g_curr[j] - g_prev[j];
        y_dot_s += y_k[j] * s_k[j];
    }

    if (y_dot_s <= 1e-8f) return;

    float rho_k = 1.0f / y_dot_s;
    int count = history_count[row];
    int start = history_start[row];
    int write_idx = (start + count) % history_len;

    if (count < history_len) {
        history_count[row] = count + 1;
    } else {
        history_start[row] = (start + 1) % history_len;
    }

    float* s_out = s_history + (row * history_len + write_idx) * n_dofs;
    float* y_out = y_history + (row * history_len + write_idx) * n_dofs;
    for (int j = 0; j < n_dofs; ++j) {
        s_out[j] = s_k[j];
        y_out[j] = y_k[j];
    }
    rho_history[row * history_len + write_idx] = rho_k;
}

// ============================================================================
// Generate line search candidates: candidate_dq = alpha * search_direction
// Also copies joint_q for each candidate.
// dim = [n_batch * n_line_steps]
// ============================================================================

__global__ void ik_generate_candidates_kernel(
    int n_batch, int n_line_steps, int n_coords, int n_dofs,
    const float* __restrict__ joint_q,           // [n_batch, n_coords]
    const float* __restrict__ search_direction,  // [n_batch, n_dofs]
    const float* __restrict__ line_search_alphas, // [n_line_steps]
    float* __restrict__ candidate_q,             // [n_batch * n_line_steps, n_coords]
    float* __restrict__ candidate_dq             // [n_batch * n_line_steps, n_dofs]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_line_steps;
    if (tid >= total) return;

    int row = tid / n_line_steps;
    int step = tid % n_line_steps;
    float alpha = line_search_alphas[step];
    int out_idx = row * n_line_steps + step;

    for (int c = 0; c < n_coords; ++c)
        candidate_q[out_idx * n_coords + c] = joint_q[row * n_coords + c];
    for (int d = 0; d < n_dofs; ++d)
        candidate_dq[out_idx * n_dofs + d] = alpha * search_direction[row * n_dofs + d];
}

// ============================================================================
// Compute directional slope: slope = sum(gradient * search_direction)
// dim = n_batch
// ============================================================================

__global__ void ik_compute_slope_kernel(
    int n_batch, int n_dofs,
    const float* __restrict__ gradient,         // [n_batch, n_dofs]
    const float* __restrict__ search_direction, // [n_batch, n_dofs]
    float* __restrict__ slope                   // [n_batch]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float s = 0.0f;
    const float* g = gradient + row * n_dofs;
    const float* d = search_direction + row * n_dofs;
    for (int j = 0; j < n_dofs; ++j) s += g[j] * d[j];
    slope[row] = s;
}

// ============================================================================
// Compute candidate slopes for Wolfe curvature condition
// dim = [n_batch * n_line_steps]
// ============================================================================

__global__ void ik_compute_candidate_slopes_kernel(
    int n_batch, int n_line_steps, int n_dofs,
    const float* __restrict__ candidate_gradients, // [n_batch, n_line_steps, n_dofs]
    const float* __restrict__ search_direction,    // [n_batch, n_dofs]
    float* __restrict__ candidate_slopes           // [n_batch, n_line_steps]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_line_steps;
    if (tid >= total) return;

    int row = tid / n_line_steps;
    int step = tid % n_line_steps;

    float s = 0.0f;
    const float* g = candidate_gradients + (row * n_line_steps + step) * n_dofs;
    const float* d = search_direction + row * n_dofs;
    for (int j = 0; j < n_dofs; ++j) s += g[j] * d[j];
    candidate_slopes[row * n_line_steps + step] = s;
}

// ============================================================================
// Select best line search step using Strong Wolfe conditions
// dim = n_batch
// ============================================================================

__global__ void ik_select_best_step_kernel(
    int n_batch, int n_line_steps, int n_dofs,
    const float* __restrict__ candidate_costs,     // [n_batch, n_line_steps]
    const float* __restrict__ candidate_step_dq,   // [n_batch, n_line_steps, n_dofs]
    const float* __restrict__ cost_initial,        // [n_batch]
    const float* __restrict__ slope_initial,       // [n_batch]
    const float* __restrict__ candidate_slopes,    // [n_batch, n_line_steps]
    const float* __restrict__ line_search_alphas,  // [n_line_steps]
    float wolfe_c1, float wolfe_c2,
    int* __restrict__ best_step_idx,               // [n_batch]
    float* __restrict__ last_step                   // [n_batch, n_dofs]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float cost_k = cost_initial[row];
    float slope_k = slope_initial[row];

    int best_idx = -1;

    // Search backwards for largest step satisfying Wolfe
    for (int i = n_line_steps - 1; i >= 0; --i) {
        float cost_new = candidate_costs[row * n_line_steps + i];
        float alpha = line_search_alphas[i];

        bool armijo = cost_new <= cost_k + wolfe_c1 * alpha * slope_k;
        float slope_new = candidate_slopes[row * n_line_steps + i];
        bool curvature = fabsf(slope_new) <= wolfe_c2 * fabsf(slope_k);

        if (armijo && curvature) {
            best_idx = i;
            break;
        }
    }

    // Fallback: minimum cost
    if (best_idx == -1) {
        float min_cost = FLT_MAX;
        for (int i = 0; i < n_line_steps; ++i) {
            float c = candidate_costs[row * n_line_steps + i];
            if (c < min_cost) { min_cost = c; best_idx = i; }
        }
    }

    int accept_idx = best_idx;
    if (best_idx >= 0 && candidate_costs[row * n_line_steps + best_idx] >= cost_k) {
        accept_idx = -1;
    }

    best_step_idx[row] = accept_idx;

    float* ls = last_step + row * n_dofs;
    if (accept_idx >= 0) {
        const float* step = candidate_step_dq + (row * n_line_steps + accept_idx) * n_dofs;
        for (int j = 0; j < n_dofs; ++j) ls[j] = step[j];
    } else {
        for (int j = 0; j < n_dofs; ++j) ls[j] = 0.0f;
    }
}

// ============================================================================
// Apply best step: joint_q = candidate_q[best_idx] if accepted
// dim = n_batch
// ============================================================================

__global__ void ik_apply_best_step_kernel(
    int n_batch, int n_line_steps, int n_coords,
    const float* __restrict__ candidate_q_integrated, // [n_batch * n_line_steps, n_coords]
    const int* __restrict__ best_step_idx,            // [n_batch]
    float* __restrict__ joint_q                       // [n_batch, n_coords]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    int idx = best_step_idx[row];
    if (idx < 0) return;

    int src = row * n_line_steps + idx;
    for (int c = 0; c < n_coords; ++c) {
        joint_q[row * n_coords + c] = candidate_q_integrated[src * n_coords + c];
    }
}

// ============================================================================
// Scale-negate kernel for initial L-BFGS step: out = -scale * gradient
// dim = [n_batch * n_dofs]
// ============================================================================

__global__ void ik_scale_negate_kernel(
    int n_batch, int n_dofs, float scale,
    const float* __restrict__ gradient,
    float* __restrict__ output
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_dofs;
    if (tid >= total) return;
    output[tid] = -scale * gradient[tid];
}

}  // namespace ik
}  // namespace chysx
