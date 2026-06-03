// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for the Featherstone articulated-body solver.
// 1:1 port of Newton's Warp kernels from
// newton/_src/solvers/featherstone/kernels.py
//
// All kernels are launched with 1 thread per articulation (serial over
// joints within each articulation tree), matching Newton's approach.

#pragma once

#include "spatial_math.cuh"
#include "joint_types.cuh"

namespace chysx {
namespace rigid {

// ============================================================================
// compute_spatial_inertia: build body-frame spatial inertia from mass + I_3x3
// ============================================================================
__global__ void compute_spatial_inertia_kernel(
    const Mat3f* __restrict__ body_inertia,
    const float* __restrict__ body_mass,
    SpatialMatrix* __restrict__ body_I_m,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    body_I_m[tid] = build_spatial_inertia(body_inertia[tid], body_mass[tid]);
}

// ============================================================================
// compute_com_transforms: body_X_com[i] = transform(body_com[i], identity_quat)
// ============================================================================
__global__ void compute_com_transforms_kernel(
    const Vec3f* __restrict__ body_com,
    Transform7* __restrict__ body_X_com,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    body_X_com[tid] = Transform7(body_com[tid], quat_identity());
}

// ============================================================================
// zero_kinematic_body_forces: zero body_f for kinematic bodies
// ============================================================================
__global__ void zero_kinematic_body_forces_kernel(
    const int* __restrict__ body_flags,
    SpatialVector* __restrict__ body_f,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if ((body_flags[tid] & BodyFlags::KINEMATIC) == 0) return;
    body_f[tid] = SpatialVector();
}

// ============================================================================
// convert_body_force_com_to_origin: convert external body force from
// COM frame to origin frame and negate for Featherstone convention
// ============================================================================
__global__ void convert_body_force_com_to_origin_kernel(
    const Transform7* __restrict__ body_q,
    const Transform7* __restrict__ body_X_com,
    SpatialVector* __restrict__ body_f_ext,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    SpatialVector f_ext_com = body_f_ext[tid];
    if (spatial_length(f_ext_com) == 0.f) return;

    Transform7 body_q_com = tf_multiply(body_q[tid], body_X_com[tid]);
    Vec3f r_com = tf_get_translation(body_q_com);
    Vec3f force = f_ext_com.linear();
    Vec3f torque_com = f_ext_com.angular();

    body_f_ext[tid] = -SpatialVector(force, torque_com + cross(r_com, force));
}

// ============================================================================
// accumulate_free_distance_joint_f_to_body_force
// ============================================================================
__global__ void accumulate_free_distance_joint_f_kernel(
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_qd_start,
    const Transform7* __restrict__ body_q,
    const Transform7* __restrict__ body_X_com,
    const float* __restrict__ joint_f_public,
    SpatialVector* __restrict__ body_f_ext,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int jtype = joint_type[joint_id];
    if (jtype != FJointType::FREE && jtype != FJointType::DISTANCE) return;

    int qd_start = joint_qd_start[joint_id];
    int child = joint_child[joint_id];
    Transform7 X_sm = tf_multiply(body_q[child], body_X_com[child]);
    Vec3f r_com = tf_get_translation(X_sm);

    Vec3f force(joint_f_public[qd_start + 0],
                joint_f_public[qd_start + 1],
                joint_f_public[qd_start + 2]);
    Vec3f torque_com(joint_f_public[qd_start + 3],
                     joint_f_public[qd_start + 4],
                     joint_f_public[qd_start + 5]);

    SpatialVector f = -SpatialVector(force, torque_com + cross(r_com, force));

    // atomic add to body_f_ext[child]
    for (int i = 0; i < 6; ++i)
        atomicAdd(&body_f_ext[child][i], f[i]);
}

// ============================================================================
// convert_free_distance_joint_qd_public_to_internal
// ============================================================================
__global__ void convert_qd_public_to_internal_kernel(
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_qd_start,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ body_q,
    const Vec3f* __restrict__ body_com,
    const float* __restrict__ joint_qd_public,
    float* __restrict__ joint_qd_internal,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int qd_start = joint_qd_start[joint_id];
    int qd_end = joint_qd_start[joint_id + 1];
    int jtype = joint_type[joint_id];

    if (jtype != FJointType::FREE && jtype != FJointType::DISTANCE) {
        for (int i = qd_start; i < qd_end; ++i)
            joint_qd_internal[i] = joint_qd_public[i];
        return;
    }

    int parent = joint_parent[joint_id];
    int child = joint_child[joint_id];

    Transform7 X_wpj = joint_X_p[joint_id];
    if (parent >= 0)
        X_wpj = tf_multiply(body_q[parent], X_wpj);

    Quatf q_p = tf_get_rotation(X_wpj);
    Vec3f x_anchor_world = tf_get_translation(X_wpj);
    Vec3f x_child_com_world = tf_point(body_q[child], body_com[child]);
    Vec3f r_child_com_parent = quat_rotate_inv(q_p, x_child_com_world - x_anchor_world);

    Vec3f v_com_parent(joint_qd_public[qd_start + 0],
                       joint_qd_public[qd_start + 1],
                       joint_qd_public[qd_start + 2]);
    Vec3f omega_parent(joint_qd_public[qd_start + 3],
                       joint_qd_public[qd_start + 4],
                       joint_qd_public[qd_start + 5]);
    Vec3f v_internal = v_com_parent - cross(omega_parent, r_child_com_parent);

    joint_qd_internal[qd_start + 0] = v_internal.x;
    joint_qd_internal[qd_start + 1] = v_internal.y;
    joint_qd_internal[qd_start + 2] = v_internal.z;
    joint_qd_internal[qd_start + 3] = omega_parent.x;
    joint_qd_internal[qd_start + 4] = omega_parent.y;
    joint_qd_internal[qd_start + 5] = omega_parent.z;
}

// ============================================================================
// convert_free_distance_joint_qd_internal_to_public
// ============================================================================
__global__ void convert_qd_internal_to_public_kernel(
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_qd_start,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ body_q,
    const Vec3f* __restrict__ body_com,
    const float* __restrict__ joint_qd_internal,
    float* __restrict__ joint_qd_public,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int qd_start = joint_qd_start[joint_id];
    int qd_end = joint_qd_start[joint_id + 1];
    int jtype = joint_type[joint_id];

    if (jtype != FJointType::FREE && jtype != FJointType::DISTANCE) {
        for (int i = qd_start; i < qd_end; ++i)
            joint_qd_public[i] = joint_qd_internal[i];
        return;
    }

    int parent = joint_parent[joint_id];
    int child = joint_child[joint_id];

    Transform7 X_wpj = joint_X_p[joint_id];
    if (parent >= 0)
        X_wpj = tf_multiply(body_q[parent], X_wpj);

    Quatf q_p = tf_get_rotation(X_wpj);
    Vec3f x_anchor_world = tf_get_translation(X_wpj);
    Vec3f x_child_com_world = tf_point(body_q[child], body_com[child]);
    Vec3f r_child_com_parent = quat_rotate_inv(q_p, x_child_com_world - x_anchor_world);

    Vec3f v_internal(joint_qd_internal[qd_start + 0],
                     joint_qd_internal[qd_start + 1],
                     joint_qd_internal[qd_start + 2]);
    Vec3f omega(joint_qd_internal[qd_start + 3],
                joint_qd_internal[qd_start + 4],
                joint_qd_internal[qd_start + 5]);
    Vec3f v_com = v_internal + cross(omega, r_child_com_parent);

    joint_qd_public[qd_start + 0] = v_com.x;
    joint_qd_public[qd_start + 1] = v_com.y;
    joint_qd_public[qd_start + 2] = v_com.z;
    joint_qd_public[qd_start + 3] = omega.x;
    joint_qd_public[qd_start + 4] = omega.y;
    joint_qd_public[qd_start + 5] = omega.z;
}

// ============================================================================
// convert_free_distance_joint_f_public_to_internal
// ============================================================================
__global__ void convert_f_public_to_internal_kernel(
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_qd_start,
    const float* __restrict__ joint_f_public,
    float* __restrict__ joint_f_internal,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int qd_start = joint_qd_start[joint_id];
    int qd_end = joint_qd_start[joint_id + 1];
    int jtype = joint_type[joint_id];

    if (jtype != FJointType::FREE && jtype != FJointType::DISTANCE) {
        for (int i = qd_start; i < qd_end; ++i)
            joint_f_internal[i] = joint_f_public[i];
        return;
    }

    for (int i = qd_start; i < qd_end; ++i)
        joint_f_internal[i] = 0.f;
}

// ============================================================================
// eval_rigid_fk: forward kinematics (1 thread per articulation)
// ============================================================================
__global__ void eval_rigid_fk_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const float* __restrict__ joint_q,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ joint_X_c,
    const Transform7* __restrict__ body_X_com,
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,  // [joint_count * 2]
    Transform7* __restrict__ body_q,
    Transform7* __restrict__ body_q_com,
    int n_articulations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_articulations) return;

    int start = articulation_start[index];
    int end = articulation_start[index + 1];

    for (int i = start; i < end; ++i) {
        int parent = joint_parent[i];
        int child = joint_child[i];

        Transform7 X_pj = joint_X_p[i];
        Transform7 X_cj = joint_X_c[i];

        Transform7 X_wpj = X_pj;
        if (parent >= 0)
            X_wpj = tf_multiply(body_q[parent], X_wpj);

        int type = joint_type[i];
        int qd_start = joint_qd_start[i];
        int lin_axis_count = joint_dof_dim[i * 2 + 0];
        int ang_axis_count = joint_dof_dim[i * 2 + 1];
        int coord_start = joint_q_start[i];

        Transform7 X_j = jcalc_transform(
            type, joint_axis, qd_start, lin_axis_count, ang_axis_count,
            joint_q, coord_start);

        Transform7 X_wcj = tf_multiply(X_wpj, X_j);
        Transform7 X_wc = tf_multiply(X_wcj, tf_inverse(X_cj));

        Transform7 X_cm = body_X_com[child];
        Transform7 X_sm = tf_multiply(X_wc, X_cm);

        body_q[child] = X_wc;
        body_q_com[child] = X_sm;
    }
}

// ============================================================================
// eval_rigid_id: forward RNEA (1 thread per articulation)
// Computes body velocities, accelerations, Coriolis forces, spatial inertias
// ============================================================================
__global__ void eval_rigid_id_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_qd_start,
    const float* __restrict__ joint_qd,
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,
    const SpatialMatrix* __restrict__ body_I_m,
    const Transform7* __restrict__ body_q,
    const Transform7* __restrict__ body_q_com,
    const Transform7* __restrict__ joint_X_p,
    const int* __restrict__ body_world,
    const Vec3f* __restrict__ gravity,
    SpatialVector* __restrict__ joint_S_s,
    SpatialMatrix* __restrict__ body_I_s,
    SpatialVector* __restrict__ body_v_s,
    SpatialVector* __restrict__ body_f_s,
    SpatialVector* __restrict__ body_a_s,
    int n_articulations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_articulations) return;

    int start = articulation_start[index];
    int end = articulation_start[index + 1];

    for (int i = start; i < end; ++i) {
        int type = joint_type[i];
        int child = joint_child[i];
        int parent = joint_parent[i];
        int qd_start = joint_qd_start[i];

        Transform7 X_pj = joint_X_p[i];
        Transform7 X_wpj = X_pj;
        if (parent >= 0)
            X_wpj = tf_multiply(body_q[parent], X_wpj);

        int lin_axis_count = joint_dof_dim[i * 2 + 0];
        int ang_axis_count = joint_dof_dim[i * 2 + 1];

        SpatialVector v_j_s = jcalc_motion(
            type, joint_axis, lin_axis_count, ang_axis_count,
            X_wpj, joint_qd, qd_start, joint_S_s);

        SpatialVector v_parent_s;
        SpatialVector a_parent_s;
        if (parent >= 0) {
            v_parent_s = body_v_s[parent];
            a_parent_s = body_a_s[parent];
        }

        SpatialVector v_s = v_parent_s + v_j_s;
        SpatialVector a_s = a_parent_s + spatial_cross(v_s, v_j_s);

        Transform7 X_sm = body_q_com[child];
        SpatialMatrix I_m = body_I_m[child];

        float m = I_m(0, 0);
        int world_idx = body_world[child];
        Vec3f world_g = gravity[world_idx >= 0 ? world_idx : 0];
        Vec3f f_g = world_g * m;
        Vec3f r_com = tf_get_translation(X_sm);
        SpatialVector f_g_s(f_g, cross(r_com, f_g));

        SpatialMatrix I_s = transform_spatial_inertia(X_sm, I_m);
        SpatialVector f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s);

        body_v_s[child] = v_s;
        body_a_s[child] = a_s;
        body_f_s[child] = f_b_s - f_g_s;
        body_I_s[child] = I_s;
    }
}

// ============================================================================
// eval_rigid_tau: backward RNEA (1 thread per articulation)
// Computes generalized forces (tau) from body spatial forces
// ============================================================================
__global__ void eval_rigid_tau_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const int* __restrict__ joint_dof_dim,
    const float* __restrict__ target_pos,
    const float* __restrict__ target_vel,
    const float* __restrict__ joint_q,
    const float* __restrict__ joint_qd,
    const float* __restrict__ joint_f,
    const float* __restrict__ target_ke,
    const float* __restrict__ target_kd,
    const float* __restrict__ limit_lower,
    const float* __restrict__ limit_upper,
    const float* __restrict__ limit_ke,
    const float* __restrict__ limit_kd,
    const SpatialVector* __restrict__ joint_S_s,
    const SpatialVector* __restrict__ body_fb_s,
    const SpatialVector* __restrict__ body_f_ext,
    SpatialVector* __restrict__ body_ft_s,
    float* __restrict__ tau,
    int n_articulations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_articulations) return;

    int start = articulation_start[index];
    int end = articulation_start[index + 1];
    int count = end - start;

    for (int offset = 0; offset < count; ++offset) {
        int i = end - offset - 1;  // backward traversal

        int type = joint_type[i];
        int parent = joint_parent[i];
        int child = joint_child[i];
        int dof_start = joint_qd_start[i];
        int coord_start = joint_q_start[i];
        int lin_axis_count = joint_dof_dim[i * 2 + 0];
        int ang_axis_count = joint_dof_dim[i * 2 + 1];

        SpatialVector f_b_s = body_fb_s[child];
        SpatialVector f_t_s = body_ft_s[child];
        SpatialVector f_ext = body_f_ext[child];
        SpatialVector f_s = f_b_s + f_t_s + f_ext;

        jcalc_tau(type, target_ke, target_kd, limit_ke, limit_kd,
                  joint_S_s, joint_q, joint_qd, joint_f,
                  target_pos, target_vel, limit_lower, limit_upper,
                  coord_start, dof_start, lin_axis_count, ang_axis_count,
                  f_s, tau);

        if (parent >= 0) {
            for (int k = 0; k < 6; ++k)
                atomicAdd(&body_ft_s[parent][k], f_s[k]);
        }
    }
}

// ============================================================================
// eval_rigid_jacobian: build J (1 thread per articulation)
// ============================================================================

CHYSX_DI int dense_index(int stride, int i, int j) {
    return i * stride + j;
}

__global__ void eval_rigid_jacobian_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ articulation_J_start,
    const int* __restrict__ joint_ancestor,
    const int* __restrict__ joint_qd_start,
    const SpatialVector* __restrict__ joint_S_s,
    float* __restrict__ J,
    int n_articulations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_articulations) return;

    int joint_start = articulation_start[index];
    int joint_end = articulation_start[index + 1];
    int joint_count = joint_end - joint_start;
    int J_offset = articulation_J_start[index];
    int art_dof_start = joint_qd_start[joint_start];
    int art_dof_end = joint_qd_start[joint_end];
    int art_dof_count = art_dof_end - art_dof_start;

    for (int i = 0; i < joint_count; ++i) {
        int row_start = i * 6;
        int j = joint_start + i;

        while (j != -1) {
            int jdof_start = joint_qd_start[j];
            int jdof_end = joint_qd_start[j + 1];
            int jdof_count = jdof_end - jdof_start;

            for (int dof = 0; dof < jdof_count; ++dof) {
                int col = (jdof_start - art_dof_start) + dof;
                SpatialVector S = joint_S_s[jdof_start + dof];
                for (int k = 0; k < 6; ++k)
                    J[J_offset + dense_index(art_dof_count, row_start + k, col)] = S[k];
            }

            j = joint_ancestor[j];
        }
    }
}

// ============================================================================
// eval_rigid_mass: build block-diagonal M from body I_s (1 thread per art)
// ============================================================================
__global__ void eval_rigid_mass_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ articulation_M_start,
    const SpatialMatrix* __restrict__ body_I_s,
    float* __restrict__ M,
    int n_articulations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_articulations) return;

    int joint_start = articulation_start[index];
    int joint_end = articulation_start[index + 1];
    int joint_count = joint_end - joint_start;
    int M_offset = articulation_M_start[index];
    int stride = joint_count * 6;

    for (int l = 0; l < joint_count; ++l) {
        SpatialMatrix I = body_I_s[joint_start + l];
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                M[M_offset + dense_index(stride, l * 6 + i, l * 6 + j)] = I(i, j);
    }
}

// ============================================================================
// dense_gemm_batched: P = M*J or H = J^T*P  (1 thread per articulation)
// ============================================================================
__global__ void eval_dense_gemm_batched_kernel(
    const int* __restrict__ m_arr,
    const int* __restrict__ n_arr,
    const int* __restrict__ p_arr,
    bool transpose_A,
    bool transpose_B,
    const int* __restrict__ A_start,
    const int* __restrict__ B_start,
    const int* __restrict__ C_start,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n_batches)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= n_batches) return;

    int m = m_arr[batch];
    int n = n_arr[batch];
    int p = p_arr[batch];
    int a_off = A_start[batch];
    int b_off = B_start[batch];
    int c_off = C_start[batch];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int k = 0; k < p; ++k) {
                int a_i = transpose_A ? (k * m + i) : (i * p + k);
                int b_j = transpose_B ? (j * p + k) : (k * n + j);
                sum += A[a_off + a_i] * B[b_off + b_j];
            }
            C[c_off + i * n + j] = sum;
        }
    }
}

// ============================================================================
// dense_cholesky_batched: L L^T = H + diag(R)  (1 thread per articulation)
// ============================================================================
__global__ void eval_dense_cholesky_batched_kernel(
    const int* __restrict__ A_starts,
    const int* __restrict__ A_dim,
    const int* __restrict__ R_starts,
    const float* __restrict__ A,
    const float* __restrict__ R,
    float* __restrict__ L,
    int n_batches)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= n_batches) return;

    int n = A_dim[batch];
    int A_start = A_starts[batch];
    int R_start = R_starts[batch];

    for (int j = 0; j < n; ++j) {
        float s = A[A_start + dense_index(n, j, j)] + R[R_start + j];
        for (int k = 0; k < j; ++k) {
            float r = L[A_start + dense_index(n, j, k)];
            s -= r * r;
        }
        s = sqrtf(s);
        float invS = 1.f / s;
        L[A_start + dense_index(n, j, j)] = s;

        for (int i = j + 1; i < n; ++i) {
            float si = A[A_start + dense_index(n, i, j)];
            for (int k = 0; k < j; ++k)
                si -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)];
            L[A_start + dense_index(n, i, j)] = si * invS;
        }
    }
}

// ============================================================================
// dense_solve_batched: solve (L L^T) x = b  (1 thread per articulation)
// ============================================================================
__global__ void eval_dense_solve_batched_kernel(
    const int* __restrict__ L_start_arr,
    const int* __restrict__ L_dim,
    const int* __restrict__ b_start_arr,
    const float* __restrict__ L,
    const float* __restrict__ b,
    float* __restrict__ x,
    int n_batches)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= n_batches) return;

    int n = L_dim[batch];
    int L_start = L_start_arr[batch];
    int b_start = b_start_arr[batch];

    // Forward substitution: L y = b
    for (int i = 0; i < n; ++i) {
        float s = b[b_start + i];
        for (int j = 0; j < i; ++j)
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j];
        x[b_start + i] = s / L[L_start + dense_index(n, i, i)];
    }

    // Backward substitution: L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        float s = x[b_start + i];
        for (int j = i + 1; j < n; ++j)
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j];
        x[b_start + i] = s / L[L_start + dense_index(n, i, i)];
    }
}

// ============================================================================
// zero_kinematic_joint_qdd
// ============================================================================
__global__ void zero_kinematic_joint_qdd_kernel(
    const int* __restrict__ joint_child,
    const int* __restrict__ body_flags,
    const int* __restrict__ joint_qd_start,
    float* __restrict__ joint_qdd,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int child = joint_child[joint_id];
    if ((body_flags[child] & BodyFlags::KINEMATIC) == 0) return;

    int dof_start = joint_qd_start[joint_id];
    int dof_end = joint_qd_start[joint_id + 1];
    for (int i = dof_start; i < dof_end; ++i)
        joint_qdd[i] = 0.f;
}

// ============================================================================
// integrate_generalized_joints (1 thread per joint)
// ============================================================================
__global__ void integrate_generalized_joints_kernel(
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const int* __restrict__ joint_dof_dim,
    const Transform7* __restrict__ joint_X_c,
    const Vec3f* __restrict__ body_com,
    const float* __restrict__ joint_q,
    const float* __restrict__ joint_qd,
    const float* __restrict__ joint_qdd,
    float dt,
    float* __restrict__ joint_q_new,
    float* __restrict__ joint_qd_new,
    int n_joints)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_joints) return;

    int type = joint_type[index];
    int parent = joint_parent[index];
    int child = joint_child[index];
    int coord_start = joint_q_start[index];
    int dof_start = joint_qd_start[index];
    int lin_axis_count = joint_dof_dim[index * 2 + 0];
    int ang_axis_count = joint_dof_dim[index * 2 + 1];

    jcalc_integrate(parent, joint_X_c[index], body_com[child],
                    type, joint_q, joint_qd, joint_qdd,
                    coord_start, dof_start, lin_axis_count, ang_axis_count,
                    dt, joint_q_new, joint_qd_new);
}

// ============================================================================
// copy_kinematic_joint_state
// ============================================================================
__global__ void copy_kinematic_joint_state_kernel(
    const int* __restrict__ joint_child,
    const int* __restrict__ body_flags,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const float* __restrict__ joint_q_in,
    const float* __restrict__ joint_qd_in,
    float* __restrict__ joint_q_out,
    float* __restrict__ joint_qd_out,
    int n_joints)
{
    int joint_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (joint_id >= n_joints) return;

    int child = joint_child[joint_id];
    if ((body_flags[child] & BodyFlags::KINEMATIC) == 0) return;

    int q_start = joint_q_start[joint_id];
    int q_end = joint_q_start[joint_id + 1];
    for (int i = q_start; i < q_end; ++i)
        joint_q_out[i] = joint_q_in[i];

    int qd_start = joint_qd_start[joint_id];
    int qd_end = joint_qd_start[joint_id + 1];
    for (int i = qd_start; i < qd_end; ++i)
        joint_qd_out[i] = joint_qd_in[i];
}

// ============================================================================
// eval_fk_with_velocity_conversion (1 thread per articulation)
// Computes body_q and body_qd (COM twist) from joint_q and joint_qd
// ============================================================================
__global__ void eval_fk_with_velocity_conversion_kernel(
    const int* __restrict__ articulation_start,
    const float* __restrict__ joint_q,
    const float* __restrict__ joint_qd,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ joint_X_c,
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,
    const Vec3f* __restrict__ body_com,
    Transform7* __restrict__ body_q,
    SpatialVector* __restrict__ body_qd,
    int n_articulations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_articulations) return;

    int j_start = articulation_start[tid];
    int j_end = articulation_start[tid + 1];

    for (int i = j_start; i < j_end; ++i) {
        int parent = joint_parent[i];
        int child = joint_child[i];
        int type = joint_type[i];

        Transform7 X_pj = joint_X_p[i];
        Transform7 X_cj = joint_X_c[i];

        int q_start = joint_q_start[i];
        int qd_start = joint_qd_start[i];
        int lin_ax = joint_dof_dim[i * 2 + 0];
        int ang_ax = joint_dof_dim[i * 2 + 1];

        Transform7 X_j = tf_identity();
        SpatialVector v_j;

        if (type == FJointType::PRISMATIC) {
            Vec3f axis = joint_axis[qd_start];
            float q = joint_q[q_start];
            float qd = joint_qd[qd_start];
            X_j = Transform7(axis * q, quat_identity());
            v_j = SpatialVector(axis * qd, Vec3f());
        }

        if (type == FJointType::REVOLUTE) {
            Vec3f axis = joint_axis[qd_start];
            float q = joint_q[q_start];
            float qd = joint_qd[qd_start];
            X_j = Transform7(Vec3f(), quat_from_axis_angle(axis, q));
            v_j = SpatialVector(Vec3f(), axis * qd);
        }

        if (type == FJointType::BALL) {
            Quatf r(joint_q[q_start + 0], joint_q[q_start + 1],
                    joint_q[q_start + 2], joint_q[q_start + 3]);
            Vec3f w(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]);
            X_j = Transform7(Vec3f(), r);
            v_j = SpatialVector(Vec3f(), w);
        }

        if (type == FJointType::FREE || type == FJointType::DISTANCE) {
            Vec3f p(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]);
            Quatf r(joint_q[q_start + 3], joint_q[q_start + 4],
                    joint_q[q_start + 5], joint_q[q_start + 6]);
            Vec3f v(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]);
            Vec3f w(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]);
            X_j = Transform7(p, r);
            v_j = SpatialVector(v, w);
        }

        if (type == FJointType::D6) {
            Vec3f pos(0.f);
            Quatf rot = quat_identity();
            Vec3f vel_v(0.f);
            Vec3f vel_w(0.f);

            if (lin_ax > 0) { Vec3f ax = joint_axis[qd_start + 0]; pos += ax * joint_q[q_start + 0]; vel_v += ax * joint_qd[qd_start + 0]; }
            if (lin_ax > 1) { Vec3f ax = joint_axis[qd_start + 1]; pos += ax * joint_q[q_start + 1]; vel_v += ax * joint_qd[qd_start + 1]; }
            if (lin_ax > 2) { Vec3f ax = joint_axis[qd_start + 2]; pos += ax * joint_q[q_start + 2]; vel_v += ax * joint_qd[qd_start + 2]; }

            int iq = q_start + lin_ax;
            int iqd = qd_start + lin_ax;
            if (ang_ax == 1) {
                Vec3f ax = joint_axis[iqd];
                rot = quat_from_axis_angle(ax, joint_q[iq]);
                vel_w = ax * joint_qd[iqd];
            }
            if (ang_ax == 2) {
                compute_2d_rotational_dofs(
                    joint_axis[iqd + 0], joint_axis[iqd + 1],
                    joint_q[iq + 0], joint_q[iq + 1],
                    joint_qd[iqd + 0], joint_qd[iqd + 1],
                    rot, vel_w);
            }
            if (ang_ax == 3) {
                compute_3d_rotational_dofs(
                    joint_axis[iqd + 0], joint_axis[iqd + 1], joint_axis[iqd + 2],
                    joint_q[iq + 0], joint_q[iq + 1], joint_q[iq + 2],
                    joint_qd[iqd + 0], joint_qd[iqd + 1], joint_qd[iqd + 2],
                    rot, vel_w);
            }
            X_j = Transform7(pos, rot);
            v_j = SpatialVector(vel_v, vel_w);
        }

        Transform7 X_wpj = X_pj;
        if (parent >= 0)
            X_wpj = tf_multiply(body_q[parent], X_wpj);

        Transform7 X_wcj = tf_multiply(X_wpj, X_j);
        Transform7 X_wc = tf_multiply(X_wcj, tf_inverse(X_cj));

        Vec3f x_child_origin = tf_get_translation(X_wc);
        Vec3f v_parent_origin(0.f);
        Vec3f w_parent(0.f);
        if (parent >= 0) {
            SpatialVector v_wp = body_qd[parent];
            Transform7 X_wp = body_q[parent];
            w_parent = v_wp.angular();
            v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin);
        }

        Vec3f linear_joint_world = tf_vector(X_wpj, v_j.linear());
        Vec3f angular_joint_world = tf_vector(X_wpj, v_j.angular());

        Vec3f linear_joint_origin;
        if (type == FJointType::FREE || type == FJointType::DISTANCE) {
            SpatialVector v_j_world = transform_twist(X_wpj, v_j);
            linear_joint_origin = velocity_at_point(v_j_world, x_child_origin);
            angular_joint_world = v_j_world.angular();
        } else {
            Vec3f child_origin_offset = x_child_origin - tf_get_translation(X_wcj);
            linear_joint_origin = linear_joint_world + cross(angular_joint_world, child_origin_offset);
        }

        SpatialVector v_wc_origin(v_parent_origin + linear_joint_origin, w_parent + angular_joint_world);

        body_q[child] = X_wc;
        body_qd[child] = origin_twist_to_com_twist(v_wc_origin, X_wc, body_com[child]);
    }
}

// ============================================================================
// correct_free_distance_body_pose_from_world_twist
// ============================================================================
__global__ void correct_free_distance_body_pose_kernel(
    const int* __restrict__ joint_indices,
    const int* __restrict__ joint_child,
    const Vec3f* __restrict__ body_com,
    const Transform7* __restrict__ body_q_in,
    const SpatialVector* __restrict__ body_qd_out,
    Transform7* __restrict__ body_q_out,
    float dt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int joint_id = joint_indices[idx];
    int child = joint_child[joint_id];

    Transform7 X_wb = body_q_in[child];
    Vec3f b_com = body_com[child];
    SpatialVector qd_com_world = body_qd_out[child];

    Quatf q = tf_get_rotation(X_wb);
    Vec3f x_com = tf_point(X_wb, b_com);
    Vec3f v_com = qd_com_world.linear();
    Vec3f w = qd_com_world.angular();

    Quatf omega_q(w.x, w.y, w.z, 0.f);
    Quatf drdt = quat_multiply(omega_q, q) * 0.5f;
    Quatf q_new = quat_normalize(q + drdt * dt);
    Vec3f x_com_new = x_com + v_com * dt;
    Vec3f x_origin_new = x_com_new - quat_rotate(q_new, b_com);

    body_q_out[child] = Transform7(x_origin_new, q_new);
}

// ============================================================================
// reconstruct_free_distance_joint_q_from_body_pose
// ============================================================================
__global__ void reconstruct_joint_q_from_body_pose_kernel(
    const int* __restrict__ joint_indices,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const int* __restrict__ joint_q_start,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ joint_X_c,
    const Transform7* __restrict__ body_q,
    float* __restrict__ joint_q,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int joint_id = joint_indices[idx];
    int parent = joint_parent[joint_id];
    int child = joint_child[joint_id];

    Transform7 X_wpj = joint_X_p[joint_id];
    if (parent >= 0)
        X_wpj = tf_multiply(body_q[parent], X_wpj);

    Transform7 X_wcj = tf_multiply(body_q[child], joint_X_c[joint_id]);

    Vec3f x_err_c = quat_rotate_inv(
        tf_get_rotation(X_wpj),
        tf_get_translation(X_wcj) - tf_get_translation(X_wpj));

    Quatf q_pc = quat_multiply(quat_inverse(tf_get_rotation(X_wpj)), tf_get_rotation(X_wcj));

    int qs = joint_q_start[joint_id];
    joint_q[qs + 0] = x_err_c.x;
    joint_q[qs + 1] = x_err_c.y;
    joint_q[qs + 2] = x_err_c.z;
    joint_q[qs + 3] = q_pc.x;
    joint_q[qs + 4] = q_pc.y;
    joint_q[qs + 5] = q_pc.z;
    joint_q[qs + 6] = q_pc.w;
}

// ============================================================================
// eval_fk_with_velocity_conversion_from_joint (partial FK)
// ============================================================================
__global__ void eval_fk_velocity_from_joint_kernel(
    const int* __restrict__ articulation_start,
    const int* __restrict__ articulation_indices,
    const int* __restrict__ articulation_joint_start,
    const float* __restrict__ joint_q,
    const float* __restrict__ joint_qd,
    const int* __restrict__ joint_q_start,
    const int* __restrict__ joint_qd_start,
    const int* __restrict__ joint_type,
    const int* __restrict__ joint_parent,
    const int* __restrict__ joint_child,
    const Transform7* __restrict__ joint_X_p,
    const Transform7* __restrict__ joint_X_c,
    const Vec3f* __restrict__ joint_axis,
    const int* __restrict__ joint_dof_dim,
    const Vec3f* __restrict__ body_com,
    Transform7* __restrict__ body_q,
    SpatialVector* __restrict__ body_qd,
    int n)
{
    // Reuse the full FK kernel logic but starting from a specific joint
    // This is called for descendant free/distance joints after body pose correction
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int art_id = articulation_indices[tid];
    int j_start = articulation_joint_start[tid];
    int j_end = articulation_start[art_id + 1];

    for (int i = j_start; i < j_end; ++i) {
        int parent = joint_parent[i];
        int child = joint_child[i];
        int type = joint_type[i];

        Transform7 X_pj = joint_X_p[i];
        Transform7 X_cj = joint_X_c[i];

        int q_st = joint_q_start[i];
        int qd_st = joint_qd_start[i];
        int lin_ax = joint_dof_dim[i * 2 + 0];
        int ang_ax = joint_dof_dim[i * 2 + 1];

        Transform7 X_j = tf_identity();
        SpatialVector v_j;

        if (type == FJointType::PRISMATIC) {
            Vec3f axis = joint_axis[qd_st];
            X_j = Transform7(axis * joint_q[q_st], quat_identity());
            v_j = SpatialVector(axis * joint_qd[qd_st], Vec3f());
        }
        if (type == FJointType::REVOLUTE) {
            Vec3f axis = joint_axis[qd_st];
            X_j = Transform7(Vec3f(), quat_from_axis_angle(axis, joint_q[q_st]));
            v_j = SpatialVector(Vec3f(), axis * joint_qd[qd_st]);
        }
        if (type == FJointType::BALL) {
            Quatf r(joint_q[q_st], joint_q[q_st+1], joint_q[q_st+2], joint_q[q_st+3]);
            Vec3f w(joint_qd[qd_st], joint_qd[qd_st+1], joint_qd[qd_st+2]);
            X_j = Transform7(Vec3f(), r);
            v_j = SpatialVector(Vec3f(), w);
        }
        if (type == FJointType::FREE || type == FJointType::DISTANCE) {
            Vec3f p(joint_q[q_st], joint_q[q_st+1], joint_q[q_st+2]);
            Quatf r(joint_q[q_st+3], joint_q[q_st+4], joint_q[q_st+5], joint_q[q_st+6]);
            Vec3f v(joint_qd[qd_st], joint_qd[qd_st+1], joint_qd[qd_st+2]);
            Vec3f w(joint_qd[qd_st+3], joint_qd[qd_st+4], joint_qd[qd_st+5]);
            X_j = Transform7(p, r);
            v_j = SpatialVector(v, w);
        }
        if (type == FJointType::D6) {
            Vec3f pos(0.f); Quatf rot = quat_identity(); Vec3f vel_v(0.f); Vec3f vel_w(0.f);
            if (lin_ax > 0) { Vec3f ax = joint_axis[qd_st+0]; pos += ax*joint_q[q_st+0]; vel_v += ax*joint_qd[qd_st+0]; }
            if (lin_ax > 1) { Vec3f ax = joint_axis[qd_st+1]; pos += ax*joint_q[q_st+1]; vel_v += ax*joint_qd[qd_st+1]; }
            if (lin_ax > 2) { Vec3f ax = joint_axis[qd_st+2]; pos += ax*joint_q[q_st+2]; vel_v += ax*joint_qd[qd_st+2]; }
            int iq = q_st + lin_ax; int iqd = qd_st + lin_ax;
            if (ang_ax == 1) { rot = quat_from_axis_angle(joint_axis[iqd], joint_q[iq]); vel_w = joint_axis[iqd]*joint_qd[iqd]; }
            if (ang_ax == 2) { compute_2d_rotational_dofs(joint_axis[iqd], joint_axis[iqd+1], joint_q[iq], joint_q[iq+1], joint_qd[iqd], joint_qd[iqd+1], rot, vel_w); }
            if (ang_ax == 3) { compute_3d_rotational_dofs(joint_axis[iqd], joint_axis[iqd+1], joint_axis[iqd+2], joint_q[iq], joint_q[iq+1], joint_q[iq+2], joint_qd[iqd], joint_qd[iqd+1], joint_qd[iqd+2], rot, vel_w); }
            X_j = Transform7(pos, rot);
            v_j = SpatialVector(vel_v, vel_w);
        }

        Transform7 X_wpj = X_pj;
        if (parent >= 0) X_wpj = tf_multiply(body_q[parent], X_wpj);
        Transform7 X_wcj = tf_multiply(X_wpj, X_j);
        Transform7 X_wc = tf_multiply(X_wcj, tf_inverse(X_cj));

        Vec3f x_child_origin = tf_get_translation(X_wc);
        Vec3f v_parent_origin(0.f); Vec3f w_parent(0.f);
        if (parent >= 0) {
            SpatialVector v_wp = body_qd[parent]; Transform7 X_wp = body_q[parent];
            w_parent = v_wp.angular();
            v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin);
        }

        Vec3f linear_joint_world = tf_vector(X_wpj, v_j.linear());
        Vec3f angular_joint_world = tf_vector(X_wpj, v_j.angular());
        Vec3f linear_joint_origin;
        if (type == FJointType::FREE || type == FJointType::DISTANCE) {
            SpatialVector v_j_world = transform_twist(X_wpj, v_j);
            linear_joint_origin = velocity_at_point(v_j_world, x_child_origin);
            angular_joint_world = v_j_world.angular();
        } else {
            linear_joint_origin = linear_joint_world + cross(angular_joint_world, x_child_origin - tf_get_translation(X_wcj));
        }

        body_q[child] = X_wc;
        body_qd[child] = origin_twist_to_com_twist(SpatialVector(v_parent_origin + linear_joint_origin, w_parent + angular_joint_world), X_wc, body_com[child]);
    }
}

}  // namespace rigid
}  // namespace chysx
