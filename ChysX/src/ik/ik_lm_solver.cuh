// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Levenberg-Marquardt IK solver CUDA kernels.
// Solves (J^T J + lambda I) delta = -J^T r via Cholesky decomposition,
// with adaptive lambda (accept/reject with rho criterion).

#pragma once

#include <cfloat>

namespace chysx {
namespace ik {

// Maximum supported DOF count for stack-allocated matrices.
// Franka = 9 DOFs, typical robots <= 30 DOFs.
static constexpr int kMaxDofs = 64;
static constexpr int kMaxResiduals = 128;

// ============================================================================
// LM solve: build JtJ + lambda*I, Cholesky factor, solve for delta
// One thread per batch row.
// ============================================================================

__global__ void ik_lm_solve_kernel(
    int n_batch, int n_dofs, int n_residuals,
    const float* __restrict__ jacobian,      // [n_batch, n_residuals, n_dofs]
    const float* __restrict__ residuals,     // [n_batch, n_residuals]
    const float* __restrict__ lambda_values, // [n_batch]
    float* __restrict__ dq_dof,              // [n_batch, n_dofs]
    float* __restrict__ pred_reduction       // [n_batch]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float JtJ[kMaxDofs * kMaxDofs];
    float Jtr[kMaxDofs];
    float L[kMaxDofs * kMaxDofs];
    float delta[kMaxDofs];

    const float* J = jacobian + row * n_residuals * n_dofs;
    const float* r = residuals + row * n_residuals;
    float lam = lambda_values[row];

    // JtJ = J^T * J
    for (int i = 0; i < n_dofs; ++i) {
        for (int j = i; j < n_dofs; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n_residuals; ++k) {
                sum += J[k * n_dofs + i] * J[k * n_dofs + j];
            }
            JtJ[i * n_dofs + j] = sum;
            JtJ[j * n_dofs + i] = sum;
        }
    }

    // Jtr = J^T * r
    for (int i = 0; i < n_dofs; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < n_residuals; ++k) {
            sum += J[k * n_dofs + i] * r[k];
        }
        Jtr[i] = sum;
    }

    // A = JtJ + lambda * I
    for (int i = 0; i < n_dofs; ++i) {
        JtJ[i * n_dofs + i] += lam;
    }

    // Cholesky decomposition: L * L^T = A (lower triangular)
    for (int i = 0; i < n_dofs * n_dofs; ++i) L[i] = 0.0f;

    for (int j = 0; j < n_dofs; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < j; ++k) {
            sum += L[j * n_dofs + k] * L[j * n_dofs + k];
        }
        float diag = JtJ[j * n_dofs + j] - sum;
        L[j * n_dofs + j] = (diag > 0.0f) ? sqrtf(diag) : 1e-8f;

        for (int i = j + 1; i < n_dofs; ++i) {
            float s = 0.0f;
            for (int k = 0; k < j; ++k) {
                s += L[i * n_dofs + k] * L[j * n_dofs + k];
            }
            L[i * n_dofs + j] = (JtJ[i * n_dofs + j] - s) / L[j * n_dofs + j];
        }
    }

    // Forward substitution: L * y = -Jtr
    float y[kMaxDofs];
    for (int i = 0; i < n_dofs; ++i) {
        float s = 0.0f;
        for (int j = 0; j < i; ++j) {
            s += L[i * n_dofs + j] * y[j];
        }
        y[i] = (-Jtr[i] - s) / L[i * n_dofs + i];
    }

    // Back substitution: L^T * delta = y
    for (int i = n_dofs - 1; i >= 0; --i) {
        float s = 0.0f;
        for (int j = i + 1; j < n_dofs; ++j) {
            s += L[j * n_dofs + i] * delta[j];
        }
        delta[i] = (y[i] - s) / L[i * n_dofs + i];
    }

    // Store delta
    float* dq = dq_dof + row * n_dofs;
    for (int i = 0; i < n_dofs; ++i) {
        dq[i] = delta[i];
    }

    // Predicted reduction: 0.5 * delta^T * (lambda * delta - Jtr)
    float pred = 0.0f;
    for (int i = 0; i < n_dofs; ++i) {
        pred += delta[i] * (lam * delta[i] - Jtr[i]);
    }
    pred_reduction[row] = 0.5f * pred;
}

// ============================================================================
// Accept/reject decision
// rho = (cost_curr - cost_prop) / (pred_reduction + 1e-8)
// accept if rho >= rho_min
// dim = n_batch
// ============================================================================

__global__ void ik_accept_reject_kernel(
    int n_batch,
    const float* __restrict__ cost_curr,
    const float* __restrict__ cost_prop,
    const float* __restrict__ pred_red,
    float rho_min,
    int* __restrict__ accept_flags           // [n_batch]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    float rho = (cost_curr[row] - cost_prop[row]) / (pred_red[row] + 1e-8f);
    accept_flags[row] = (rho >= rho_min) ? 1 : 0;
}

// ============================================================================
// Update LM state: on accept copy proposed → current and shrink lambda;
// on reject grow lambda.
// dim = n_batch
// ============================================================================

__global__ void ik_update_lm_state_kernel(
    int n_batch, int n_coords, int n_residuals,
    const float* __restrict__ joint_q_proposed,
    const float* __restrict__ residuals_proposed,
    const float* __restrict__ costs_proposed,
    const int* __restrict__ accept_flags,
    float lambda_factor, float lambda_min, float lambda_max,
    float* __restrict__ joint_q_current,
    float* __restrict__ residuals_current,
    float* __restrict__ costs,
    float* __restrict__ lambda_values
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    if (accept_flags[row] == 1) {
        for (int i = 0; i < n_coords; ++i)
            joint_q_current[row * n_coords + i] = joint_q_proposed[row * n_coords + i];
        for (int i = 0; i < n_residuals; ++i)
            residuals_current[row * n_residuals + i] = residuals_proposed[row * n_residuals + i];
        costs[row] = costs_proposed[row];
        lambda_values[row] = lambda_values[row] / lambda_factor;
    } else {
        float new_lambda = lambda_values[row] * lambda_factor;
        lambda_values[row] = fminf(fmaxf(new_lambda, lambda_min), lambda_max);
    }
}

}  // namespace ik
}  // namespace chysx
