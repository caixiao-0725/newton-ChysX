// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// IK objective CUDA kernels: Position, Rotation, JointLimit
// residual and analytic Jacobian computation.

#pragma once

#include "../rigid/featherstone/spatial_math.cuh"
#include "../math/quat.cuh"

namespace chysx {
namespace ik {

using rigid::Vec3f;
using rigid::Quatf;
using rigid::SpatialVector;
using rigid::Transform7;
using math::quat_rotate;
using math::quat_inverse;
using math::quat_multiply;
using math::cross;
using math::dot;

// ============================================================================
// Position objective — residual (3 rows)
// weight * (target_pos - transform_point(body_q[link], link_offset))
// dim = n_batch
// ============================================================================

__global__ void ik_pos_residuals_kernel(
    int n_batch, int n_residuals,
    const float* __restrict__ body_q,        // [n_batch, body_count, 7]
    const Vec3f* __restrict__ target_pos,    // [n_problems]
    const int* __restrict__ problem_idx_map, // [n_batch]
    int link_index, Vec3f link_offset,
    int start_idx, float weight,
    int body_count,
    float* __restrict__ residuals            // [n_batch, n_residuals]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    int base = problem_idx_map[row];
    const float* bq = body_q + (row * body_count + link_index) * 7;
    Vec3f pos(bq[0], bq[1], bq[2]);
    Quatf rot(bq[3], bq[4], bq[5], bq[6]);
    Vec3f ee_pos = pos + quat_rotate(rot, link_offset);

    Vec3f target = target_pos[base];
    Vec3f error = target - ee_pos;

    float* r = residuals + row * n_residuals;
    r[start_idx + 0] = weight * error.x;
    r[start_idx + 1] = weight * error.y;
    r[start_idx + 2] = weight * error.z;
}

// ============================================================================
// Position objective — analytic Jacobian
// J[row, start+c, dof] = -weight * (S_linear + cross(S_angular, ee_pos))[c]
// dim = [n_batch * n_dofs]
// ============================================================================

__global__ void ik_pos_jac_analytic_kernel(
    int n_batch, int n_dofs, int n_residuals,
    int link_index, Vec3f link_offset,
    const unsigned char* __restrict__ affects_dof, // [n_dofs]
    const float* __restrict__ body_q,              // [n_batch, body_count, 7]
    const float* __restrict__ joint_S_s,           // [n_batch, n_dofs, 6]
    int start_idx, float weight,
    int body_count,
    float* __restrict__ jacobian                   // [n_batch, n_residuals, n_dofs]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_dofs;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / n_dofs;
        int dof = idx % n_dofs;

        if (affects_dof[dof] == 0) continue;

        const float* bq = body_q + (row * body_count + link_index) * 7;
        Vec3f pos(bq[0], bq[1], bq[2]);
        Quatf rot(bq[3], bq[4], bq[5], bq[6]);
        Vec3f ee_pos_world = pos + quat_rotate(rot, link_offset);

        const float* S = joint_S_s + (row * n_dofs + dof) * 6;
        Vec3f v_orig(S[0], S[1], S[2]);
        Vec3f omega(S[3], S[4], S[5]);
        Vec3f v_ee = v_orig + cross(omega, ee_pos_world);

        float* J = jacobian + row * n_residuals * n_dofs;
        J[(start_idx + 0) * n_dofs + dof] = -weight * v_ee.x;
        J[(start_idx + 1) * n_dofs + dof] = -weight * v_ee.y;
        J[(start_idx + 2) * n_dofs + dof] = -weight * v_ee.z;
    }
}

// ============================================================================
// Rotation objective — residual (3 rows)
// quaternion error → axis-angle vector
// dim = n_batch
// ============================================================================

__global__ void ik_rot_residuals_kernel(
    int n_batch, int n_residuals,
    const float* __restrict__ body_q,        // [n_batch, body_count, 7]
    const float* __restrict__ target_rot,    // [n_problems, 4] — (x,y,z,w)
    const int* __restrict__ problem_idx_map, // [n_batch]
    int link_index, Quatf link_offset_rotation,
    bool canonicalize_quat_err,
    int start_idx, float weight,
    int body_count,
    float* __restrict__ residuals            // [n_batch, n_residuals]
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_batch) return;

    int base = problem_idx_map[row];
    const float* bq = body_q + (row * body_count + link_index) * 7;
    Quatf body_rot(bq[3], bq[4], bq[5], bq[6]);

    Quatf actual_rot = quat_multiply(body_rot, link_offset_rotation);

    const float* tgt = target_rot + base * 4;
    Quatf target_quat(tgt[0], tgt[1], tgt[2], tgt[3]);

    Quatf q_err = quat_multiply(actual_rot, quat_inverse(target_quat));

    if (canonicalize_quat_err) {
        float d = actual_rot.x * target_quat.x + actual_rot.y * target_quat.y +
                  actual_rot.z * target_quat.z + actual_rot.w * target_quat.w;
        if (d < 0.0f) {
            q_err.x = -q_err.x; q_err.y = -q_err.y;
            q_err.z = -q_err.z; q_err.w = -q_err.w;
        }
    }

    float v_norm = sqrtf(q_err.x * q_err.x + q_err.y * q_err.y + q_err.z * q_err.z);
    float angle = 2.0f * atan2f(v_norm, q_err.w);

    Vec3f axis_angle;
    float eps = 1e-8f;
    if (v_norm > eps) {
        float inv_v = 1.0f / v_norm;
        axis_angle = Vec3f(q_err.x * inv_v, q_err.y * inv_v, q_err.z * inv_v) * angle;
    } else {
        axis_angle = Vec3f(2.0f * q_err.x, 2.0f * q_err.y, 2.0f * q_err.z);
    }

    float* r = residuals + row * n_residuals;
    r[start_idx + 0] = weight * axis_angle.x;
    r[start_idx + 1] = weight * axis_angle.y;
    r[start_idx + 2] = weight * axis_angle.z;
}

// ============================================================================
// Rotation objective — analytic Jacobian
// J[row, start+c, dof] = weight * S_angular[c]
// dim = [n_batch * n_dofs]
// ============================================================================

__global__ void ik_rot_jac_analytic_kernel(
    int n_batch, int n_dofs, int n_residuals,
    const unsigned char* __restrict__ affects_dof, // [n_dofs]
    const float* __restrict__ joint_S_s,           // [n_batch, n_dofs, 6]
    int start_idx, float weight,
    float* __restrict__ jacobian                   // [n_batch, n_residuals, n_dofs]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_dofs;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / n_dofs;
        int dof = idx % n_dofs;

        if (affects_dof[dof] == 0) continue;

        const float* S = joint_S_s + (row * n_dofs + dof) * 6;
        Vec3f omega(S[3], S[4], S[5]);

        float* J = jacobian + row * n_residuals * n_dofs;
        J[(start_idx + 0) * n_dofs + dof] = weight * omega.x;
        J[(start_idx + 1) * n_dofs + dof] = weight * omega.y;
        J[(start_idx + 2) * n_dofs + dof] = weight * omega.z;
    }
}

// ============================================================================
// JointLimit objective — residual (n_dofs rows)
// weight * (max(0, q - upper) + max(0, lower - q))
// dim = [n_batch * n_dofs]
// ============================================================================

__global__ void ik_limit_residuals_kernel(
    int n_batch, int n_dofs, int n_residuals, int n_coords,
    const float* __restrict__ joint_q,         // [n_batch, n_coords]
    const float* __restrict__ joint_limit_lower, // [n_dofs]
    const float* __restrict__ joint_limit_upper, // [n_dofs]
    const int* __restrict__ dof_to_coord,      // [n_dofs]
    int start_idx, float weight,
    float* __restrict__ residuals              // [n_batch, n_residuals]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_dofs;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / n_dofs;
        int dof = idx % n_dofs;

        int coord_idx = dof_to_coord[dof];
        if (coord_idx < 0) continue;

        float lower = joint_limit_lower[dof];
        float upper = joint_limit_upper[dof];

        if (upper - lower > 9.9e5f) continue;

        float q = joint_q[row * n_coords + coord_idx];
        float viol = fmaxf(0.0f, q - upper) + fmaxf(0.0f, lower - q);
        residuals[row * n_residuals + start_idx + dof] = weight * viol;
    }
}

// ============================================================================
// JointLimit objective — analytic Jacobian (diagonal)
// dim = [n_batch * n_dofs]
// ============================================================================

__global__ void ik_limit_jac_analytic_kernel(
    int n_batch, int n_dofs, int n_residuals, int n_coords,
    const float* __restrict__ joint_q,         // [n_batch, n_coords]
    const float* __restrict__ joint_limit_lower,
    const float* __restrict__ joint_limit_upper,
    const int* __restrict__ dof_to_coord,
    int start_idx, float weight,
    float* __restrict__ jacobian               // [n_batch, n_residuals, n_dofs]
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batch * n_dofs;
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int row = idx / n_dofs;
        int dof = idx % n_dofs;

        int coord_idx = dof_to_coord[dof];
        if (coord_idx < 0) continue;

        float lower = joint_limit_lower[dof];
        float upper = joint_limit_upper[dof];
        if (upper - lower > 9.9e5f) continue;

        float q = joint_q[row * n_coords + coord_idx];
        float grad = 0.0f;
        if (q >= upper) grad = weight;
        else if (q <= lower) grad = -weight;

        jacobian[row * n_residuals * n_dofs + (start_idx + dof) * n_dofs + dof] = grad;
    }
}

// ============================================================================
// Update target arrays (single problem)
// ============================================================================

__global__ void ik_update_target_position_kernel(
    int problem_idx, Vec3f new_pos,
    Vec3f* __restrict__ target_positions
)
{
    target_positions[problem_idx] = new_pos;
}

__global__ void ik_update_target_rotation_kernel(
    int problem_idx, float rx, float ry, float rz, float rw,
    float* __restrict__ target_rotations     // [n_problems, 4]
)
{
    float* t = target_rotations + problem_idx * 4;
    t[0] = rx; t[1] = ry; t[2] = rz; t[3] = rw;
}

}  // namespace ik
}  // namespace chysx
