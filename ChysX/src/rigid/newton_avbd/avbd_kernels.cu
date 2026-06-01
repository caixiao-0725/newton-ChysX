// SPDX-License-Identifier: Apache-2.0
//
// AVBD (Augmented Vertex Block Descent) rigid-body solver kernels.
// Ported from Newton's rigid_vbd_kernels.py.

#include "avbd_kernels.cuh"
#include "rigid_contact.h"
#include "rigid_joint.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace chysx {
namespace rigid {
namespace avbd {

namespace {

constexpr int kBlock = 256;
constexpr int kContactThreadsPerBody = 4;

inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

// ============================================================================
// Device helpers
// ============================================================================

// 6×6 block LDLT solve (fully unrolled, in-register).
__device__ Vec6Result ldlt6_solve(
    const Mat3f& h_ll, const Mat3f& h_aa, const Mat3f& h_al,
    const Vec3f& rhs_lin, const Vec3f& rhs_ang)
{
    // Unpack the symmetric 6×6 matrix:
    //   [h_ll   h_al^T]    rows 1-3: linear
    //   [h_al   h_aa  ]    rows 4-6: angular
    float A11 = h_ll(0,0), A12 = h_ll(0,1), A13 = h_ll(0,2);
    float A22 = h_ll(1,1), A23 = h_ll(1,2), A33 = h_ll(2,2);
    float A41 = h_al(0,0), A42 = h_al(0,1), A43 = h_al(0,2);
    float A51 = h_al(1,0), A52 = h_al(1,1), A53 = h_al(1,2);
    float A61 = h_al(2,0), A62 = h_al(2,1), A63 = h_al(2,2);
    float A44 = h_aa(0,0), A45 = h_aa(0,1), A46 = h_aa(0,2);
    float A55 = h_aa(1,1), A56 = h_aa(1,2), A66 = h_aa(2,2);

    float b1 = rhs_lin.x, b2 = rhs_lin.y, b3 = rhs_lin.z;
    float b4 = rhs_ang.x, b5 = rhs_ang.y, b6 = rhs_ang.z;

    // LDL^T factorization (row-by-row, no pivoting)
    float D1 = A11;
    float L21 = A12 / D1;
    float L31 = A13 / D1;
    float L41 = A41 / D1;
    float L51 = A51 / D1;
    float L61 = A61 / D1;

    float D2 = A22 - L21 * L21 * D1;
    float L32 = (A23 - L31 * L21 * D1) / D2;
    float L42 = (A42 - L41 * L21 * D1) / D2;
    float L52 = (A52 - L51 * L21 * D1) / D2;
    float L62 = (A62 - L61 * L21 * D1) / D2;

    float D3 = A33 - L31 * L31 * D1 - L32 * L32 * D2;
    float L43 = (A43 - L41 * L31 * D1 - L42 * L32 * D2) / D3;
    float L53 = (A53 - L51 * L31 * D1 - L52 * L32 * D2) / D3;
    float L63 = (A63 - L61 * L31 * D1 - L62 * L32 * D2) / D3;

    float D4 = A44 - L41 * L41 * D1 - L42 * L42 * D2 - L43 * L43 * D3;
    float L54 = (A45 - L51 * L41 * D1 - L52 * L42 * D2 - L53 * L43 * D3) / D4;
    float L64 = (A46 - L61 * L41 * D1 - L62 * L42 * D2 - L63 * L43 * D3) / D4;

    float D5 = A55 - L51 * L51 * D1 - L52 * L52 * D2
             - L53 * L53 * D3 - L54 * L54 * D4;
    float L65 = (A56 - L61 * L51 * D1 - L62 * L52 * D2
               - L63 * L53 * D3 - L64 * L54 * D4) / D5;

    float D6 = A66 - L61 * L61 * D1 - L62 * L62 * D2
             - L63 * L63 * D3 - L64 * L64 * D4 - L65 * L65 * D5;

    // Forward substitution: L y = b
    float y1 = b1;
    float y2 = b2 - L21 * y1;
    float y3 = b3 - L31 * y1 - L32 * y2;
    float y4 = b4 - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = b5 - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = b6 - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    // Diagonal solve: z = D^-1 y
    float z1 = y1 / D1, z2 = y2 / D2, z3 = y3 / D3;
    float z4 = y4 / D4, z5 = y5 / D5, z6 = y6 / D6;

    // Back substitution: L^T x = z
    float x6 = z6;
    float x5 = z5 - L65 * x6;
    float x4 = z4 - L54 * x5 - L64 * x6;
    float x3 = z3 - L43 * x4 - L53 * x5 - L63 * x6;
    float x2 = z2 - L32 * x3 - L42 * x4 - L52 * x5 - L62 * x6;
    float x1 = z1 - L21 * x2 - L31 * x3 - L41 * x4 - L51 * x5 - L61 * x6;

    return {{x1, x2, x3}, {x4, x5, x6}};
}

// Evaluate rigid contact force and Hessian for one body's side.
struct ContactForceResult {
    Vec3f force;
    Vec3f torque;
    Mat3f H_ll;
    Mat3f H_al;
    Mat3f H_aa;
};

__device__ ContactForceResult evaluate_contact(
    int body_a, int body_b,
    const Vec3f* body_pos, const Quatf* body_quat,
    const Vec3f* body_pos_prev, const Quatf* body_quat_prev,
    const Vec3f* body_com,
    Vec3f p0_local, Vec3f p1_local, Vec3f normal,
    float penetration_depth, float ke, float kd,
    const Vec3f& lam, float friction_mu, float friction_epsilon,
    int hard_contact, float dt, Vec3f friction_c0,
    bool is_body_a)
{
    ContactForceResult r{};

    if (ke <= 0.f) return r;

    float lambda_n = math::dot(lam, normal);

    // Early exit: no penetration and no stored lambda
    if (penetration_depth <= 1e-7f && lambda_n <= 0.f) return r;

    float fn = ke * penetration_depth + lambda_n;
    if (fn <= 0.f && lambda_n <= 0.f) return r;
    fn = fmaxf(fn, 0.f);

    // Contact points in world frame
    Vec3f pa_world, pb_world;
    if (body_a >= 0) {
        pa_world = math::transform_point(body_pos[body_a], body_quat[body_a], p0_local);
    } else {
        pa_world = p0_local;
    }
    if (body_b >= 0) {
        pb_world = math::transform_point(body_pos[body_b], body_quat[body_b], p1_local);
    } else {
        pb_world = p1_local;
    }

    Vec3f f_total = normal * fn;
    Mat3f K = math::outer(normal, normal) * ke;

    if (hard_contact) {
        // Tangential displacement
        Vec3f pa_prev, pb_prev;
        if (body_a >= 0) {
            pa_prev = math::transform_point(body_pos_prev[body_a], body_quat_prev[body_a], p0_local);
        } else {
            pa_prev = p0_local;
        }
        if (body_b >= 0) {
            pb_prev = math::transform_point(body_pos_prev[body_b], body_quat_prev[body_b], p1_local);
        } else {
            pb_prev = p1_local;
        }

        Vec3f v_rel = ((pb_world - pb_prev) - (pa_world - pa_prev)) * (1.f / dt);
        float vn = math::dot(v_rel, normal);
        Vec3f vt = v_rel - normal * vn;
        Vec3f ut = vt * (-dt);  // tangential displacement

        Vec3f lambda_t = lam - normal * lambda_n;
        Vec3f ft = (ut + friction_c0) * ke + lambda_t;

        float ft_len = math::length(ft);
        float cone_limit = friction_mu * fn;
        if (ft_len > cone_limit && ft_len > 1e-10f) {
            ft = ft * (cone_limit / ft_len);
        }

        f_total = f_total + ft;

        // Tangential Hessian: ke * (I - n n^T)
        Mat3f I3 = Mat3f::identity();
        Mat3f nnT = math::outer(normal, normal);
        K = K + (I3 - nnT) * ke;

        // Normal damping
        if (kd > 0.f && vn < 0.f && fn > 0.f) {
            f_total = f_total - normal * (kd * ke * vn);
            K = K + nnT * (kd * ke / dt);
        }
    } else {
        // Soft contact: IPC-style projected isotropic friction
        if (friction_mu > 0.f && fn > 0.f) {
            Vec3f pa_prev, pb_prev;
            if (body_a >= 0) {
                pa_prev = math::transform_point(body_pos_prev[body_a], body_quat_prev[body_a], p0_local);
            } else {
                pa_prev = p0_local;
            }
            if (body_b >= 0) {
                pb_prev = math::transform_point(body_pos_prev[body_b], body_quat_prev[body_b], p1_local);
            } else {
                pb_prev = p1_local;
            }

            Vec3f u = ((pb_world - pb_prev) - (pa_world - pa_prev));
            float un = math::dot(u, normal);
            Vec3f ut = u - normal * un;
            float ut_len = math::length(ut);

            float scale = 0.f;
            if (ut_len > friction_epsilon) {
                scale = 1.f / ut_len;
            } else if (friction_epsilon > 0.f) {
                scale = 1.f / friction_epsilon;
            }

            f_total = f_total - ut * (friction_mu * fn * scale);
            Mat3f I3 = Mat3f::identity();
            Mat3f nnT = math::outer(normal, normal);
            K = K + (I3 - nnT) * (friction_mu * fn * scale);
        }

        // Normal damping
        if (kd > 0.f && fn > 0.f) {
            Vec3f pa_prev, pb_prev;
            if (body_a >= 0) {
                pa_prev = math::transform_point(body_pos_prev[body_a], body_quat_prev[body_a], p0_local);
            } else {
                pa_prev = p0_local;
            }
            if (body_b >= 0) {
                pb_prev = math::transform_point(body_pos_prev[body_b], body_quat_prev[body_b], p1_local);
            } else {
                pb_prev = p1_local;
            }
            Vec3f v_rel = ((pb_world - pb_prev) - (pa_world - pa_prev)) * (1.f / dt);
            float vn = math::dot(v_rel, normal);
            if (vn < 0.f) {
                f_total = f_total - normal * (kd * ke * vn);
                K = K + math::outer(normal, normal) * (kd * ke / dt);
            }
        }
    }

    // Compute wrench and Hessian for the requested body side
    Vec3f cp_world = is_body_a ? pa_world : pb_world;
    int body_idx = is_body_a ? body_a : body_b;
    Vec3f body_com_world;
    if (body_idx >= 0) {
        body_com_world = math::transform_point(body_pos[body_idx], body_quat[body_idx],
                                               body_com[body_idx]);
    } else {
        body_com_world = Vec3f(0.f);
    }
    Vec3f lever = cp_world - body_com_world;
    Mat3f skew_r = math::skew(lever);

    float sign = is_body_a ? -1.f : 1.f;
    r.force = f_total * sign;
    r.torque = math::cross(lever, f_total * sign);
    r.H_ll = K;
    r.H_al = math::transpose(skew_r) * K * (-1.f);
    r.H_aa = math::transpose(skew_r) * K * skew_r;

    return r;
}

// Evaluate joint force/Hessian for a body (BALL or FIXED).
__device__ void evaluate_joint_linear(
    Vec3f anchor_parent_world, Vec3f anchor_child_world,
    Vec3f com_world, float ke, const Vec3f& lam, const Vec3f& C0, float alpha,
    Vec3f& out_force, Vec3f& out_torque,
    Mat3f& out_H_ll, Mat3f& out_H_al, Mat3f& out_H_aa,
    bool is_parent)
{
    Vec3f C = anchor_child_world - anchor_parent_world;
    Vec3f C_stab = C - C0 * alpha;

    Vec3f f = C_stab * ke + lam;
    Mat3f K_eff = Mat3f::identity() * ke;

    float sign = is_parent ? 1.f : -1.f;
    Vec3f force = f * sign;
    Vec3f cp = is_parent ? anchor_parent_world : anchor_child_world;
    Vec3f lever = cp - com_world;
    Mat3f skew_r = math::skew(lever);

    out_force = force;
    out_torque = math::cross(lever, force);
    out_H_ll = K_eff;
    out_H_al = math::transpose(skew_r) * K_eff * (-1.f);
    out_H_aa = math::transpose(skew_r) * K_eff * skew_r;
}

__device__ void evaluate_joint_angular(
    const Quatf& frame_parent_world, const Quatf& frame_child_world,
    float ke, const Vec3f& lam, const Vec3f& C0, float alpha,
    Vec3f& out_torque, Mat3f& out_H_aa,
    bool is_parent)
{
    // Relative rotation: R_rel = R_p^T * R_c
    Quatf dq = math::quat_multiply(math::quat_conjugate(frame_parent_world),
                                   frame_child_world);
    if (dq.w < 0.f) dq = -dq;

    // Axis-angle from quaternion
    Vec3f imag(dq.x, dq.y, dq.z);
    float sin_half = math::length(imag);
    Vec3f kappa(0.f);
    if (sin_half > 1e-10f) {
        float angle = 2.f * atan2f(sin_half, fabsf(dq.w));
        kappa = imag * (angle / sin_half);
    }

    Vec3f kappa_stab = kappa - C0 * alpha;
    Vec3f tau_local = kappa_stab * ke + lam;

    // Transform to world frame via parent rotation
    Mat3f R_p = math::quat_to_matrix(frame_parent_world);
    Vec3f tau_world = R_p * tau_local;

    float sign = is_parent ? -1.f : 1.f;
    out_torque = tau_world * sign;
    out_H_aa = R_p * (Mat3f::identity() * ke) * math::transpose(R_p);
}

// Atomic add for Vec3f
__device__ void atomicAdd3(Vec3f* addr, const Vec3f& val) {
    atomicAdd(&addr->x, val.x);
    atomicAdd(&addr->y, val.y);
    atomicAdd(&addr->z, val.z);
}

// Atomic add for Mat3f
__device__ void atomicAdd9(Mat3f* addr, const Mat3f& val) {
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&addr->data[i], val.data[i]);
    }
}

// ============================================================================
// Kernels
// ============================================================================

__global__ void snapshot_prev_kernel(
    const Vec3f* __restrict__ pos, const Quatf* __restrict__ quat,
    Vec3f* __restrict__ pos_prev, Quatf* __restrict__ quat_prev, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pos_prev[i] = pos[i];
        quat_prev[i] = quat[i];
    }
}

__global__ void forward_step_kernel(
    float dt, Vec3f gravity,
    const Vec3f* __restrict__ com, const Mat3f* __restrict__ inertia,
    const float* __restrict__ inv_mass, const Mat3f* __restrict__ inv_inertia,
    Vec3f* __restrict__ pos, Quatf* __restrict__ quat,
    Vec3f* __restrict__ vel, Vec3f* __restrict__ omega,
    Vec3f* __restrict__ inertia_pos, Quatf* __restrict__ inertia_quat,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float im = inv_mass[i];
    if (im <= 0.f) {
        inertia_pos[i] = pos[i];
        inertia_quat[i] = quat[i];
        return;
    }

    Vec3f p = pos[i];
    Quatf q = quat[i];
    Vec3f v = vel[i];
    Vec3f w = omega[i];
    Vec3f c = com[i];
    Mat3f I_local = inertia[i];
    Mat3f I_inv   = inv_inertia[i];

    // COM in world
    Vec3f x_com = p + math::quat_rotate(q, c);

    // Linear: semi-implicit Euler
    v = v + gravity * dt;
    Vec3f x_com_new = x_com + v * dt;

    // Angular: body-frame torque integration
    Mat3f R = math::quat_to_matrix(q);
    Vec3f w_body = math::transpose(R) * w;
    // Gyroscopic term: -omega × (I * omega)
    Vec3f gyro = math::cross(w_body, I_local * w_body);
    Vec3f w_body_new = w_body + I_inv * (gyro * (-1.f)) * dt;
    Vec3f w_new = R * w_body_new;

    // Quaternion integration
    Quatf q_new = math::quat_integrate(q, w_new, dt);

    // Shift back from COM to body origin
    Vec3f p_new = x_com_new - math::quat_rotate(q_new, c);

    pos[i] = p_new;
    quat[i] = q_new;
    vel[i] = v;
    omega[i] = w_new;
    inertia_pos[i] = p_new;
    inertia_quat[i] = q_new;
}

__global__ void build_body_contact_list_kernel(
    int contact_count, int body_count, int per_body_cap,
    const int* __restrict__ contact_shape0,
    const int* __restrict__ contact_shape1,
    const int* __restrict__ shape_body,
    int* __restrict__ body_contact_counts,
    int* __restrict__ body_contact_indices)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= contact_count) return;

    int b0 = shape_body[contact_shape0[c]];
    int b1 = shape_body[contact_shape1[c]];

    if (b0 >= 0) {
        int slot = atomicAdd(&body_contact_counts[b0], 1);
        if (slot < per_body_cap) {
            body_contact_indices[b0 * per_body_cap + slot] = c;
        }
    }
    if (b1 >= 0) {
        int slot = atomicAdd(&body_contact_counts[b1], 1);
        if (slot < per_body_cap) {
            body_contact_indices[b1 * per_body_cap + slot] = c;
        }
    }
}

__global__ void clear_int_array_kernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0;
}

__global__ void init_contact_avbd_kernel(
    int n, float* __restrict__ penalty_k,
    Vec3f* __restrict__ lambda, Vec3f* __restrict__ C0,
    int* __restrict__ stick_flag,
    const float* __restrict__ material_ke)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    penalty_k[i] = material_ke[i];
    lambda[i] = Vec3f(0.f);
    C0[i] = Vec3f(0.f);
    stick_flag[i] = 0;
}

__global__ void step_C0_lambda_contacts_kernel(
    int n, float alpha, float gamma,
    const Vec3f* __restrict__ body_pos, const Quatf* __restrict__ body_quat,
    const int*   __restrict__ contact_shape0, const int* __restrict__ contact_shape1,
    const int*   __restrict__ shape_body,
    const Vec3f* __restrict__ contact_point0, const Vec3f* __restrict__ contact_point1,
    const Vec3f* __restrict__ contact_normal,
    const float* __restrict__ contact_margin0, const float* __restrict__ contact_margin1,
    Vec3f* __restrict__ C0, float* __restrict__ penalty_k, Vec3f* __restrict__ lambda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int b0 = shape_body[contact_shape0[i]];
    int b1 = shape_body[contact_shape1[i]];

    // Compute current constraint value
    Vec3f pa = (b0 >= 0) ? math::transform_point(body_pos[b0], body_quat[b0], contact_point0[i])
                         : contact_point0[i];
    Vec3f pb = (b1 >= 0) ? math::transform_point(body_pos[b1], body_quat[b1], contact_point1[i])
                         : contact_point1[i];
    Vec3f d = pb - pa;
    Vec3f n_vec = contact_normal[i];
    float gap = contact_margin0[i] + contact_margin1[i];
    float C_n = gap - math::dot(n_vec, d);

    // Snapshot C0
    Vec3f C_full = n_vec * C_n;
    C0[i] = C_full;

    // Decay lambda and k
    lambda[i] = lambda[i] * gamma;
}

__global__ void step_C0_lambda_joints_kernel(
    int n, float alpha, float gamma,
    const Vec3f* __restrict__ body_pos, const Quatf* __restrict__ body_quat,
    const int*   __restrict__ joint_parent, const int* __restrict__ joint_child,
    const Vec3f* __restrict__ X_p_pos, const Quatf* __restrict__ X_p_quat,
    const Vec3f* __restrict__ X_c_pos, const Quatf* __restrict__ X_c_quat,
    const int*   __restrict__ joint_type,
    Vec3f* __restrict__ C0_lin, Vec3f* __restrict__ C0_ang,
    float* __restrict__ penalty_k_lin, float* __restrict__ penalty_k_ang,
    Vec3f* __restrict__ lambda_lin, Vec3f* __restrict__ lambda_ang)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int p = joint_parent[j];
    int c = joint_child[j];

    // World anchor positions
    Vec3f a_p = (p >= 0) ? math::transform_point(body_pos[p], body_quat[p], X_p_pos[j])
                         : X_p_pos[j];
    Vec3f a_c = (c >= 0) ? math::transform_point(body_pos[c], body_quat[c], X_c_pos[j])
                         : X_c_pos[j];

    C0_lin[j] = a_c - a_p;
    lambda_lin[j] = lambda_lin[j] * gamma;

    if (joint_type[j] == JOINT_FIXED) {
        // Angular C0
        Quatf fp = (p >= 0) ? math::quat_multiply(body_quat[p], X_p_quat[j]) : X_p_quat[j];
        Quatf fc = (c >= 0) ? math::quat_multiply(body_quat[c], X_c_quat[j]) : X_c_quat[j];
        Quatf dq = math::quat_multiply(math::quat_conjugate(fp), fc);
        if (dq.w < 0.f) dq = -dq;
        Vec3f imag(dq.x, dq.y, dq.z);
        float sh = math::length(imag);
        Vec3f kappa(0.f);
        if (sh > 1e-10f) {
            float angle = 2.f * atan2f(sh, fabsf(dq.w));
            kappa = imag * (angle / sh);
        }
        C0_ang[j] = kappa;
        lambda_ang[j] = lambda_ang[j] * gamma;
    }
}

__global__ void zero_scratch_kernel(
    Vec3f* __restrict__ forces, Vec3f* __restrict__ torques,
    Mat3f* __restrict__ h_ll, Mat3f* __restrict__ h_al, Mat3f* __restrict__ h_aa,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    forces[i] = Vec3f(0.f);
    torques[i] = Vec3f(0.f);
    h_ll[i] = Mat3f::zero();
    h_al[i] = Mat3f::zero();
    h_aa[i] = Mat3f::zero();
}

__global__ void accumulate_contacts_kernel(
    const int* __restrict__ body_ids, int group_size,
    float dt, float alpha, float friction_epsilon, int hard_contacts,
    const Vec3f* __restrict__ body_pos, const Quatf* __restrict__ body_quat,
    const Vec3f* __restrict__ body_pos_prev, const Quatf* __restrict__ body_quat_prev,
    const Vec3f* __restrict__ body_com, const float* __restrict__ body_inv_mass,
    int per_body_cap,
    const int*   __restrict__ body_contact_counts, const int* __restrict__ body_contact_indices,
    const int*   __restrict__ contact_shape0, const int* __restrict__ contact_shape1,
    const int*   __restrict__ shape_body,
    const Vec3f* __restrict__ contact_point0, const Vec3f* __restrict__ contact_point1,
    const Vec3f* __restrict__ contact_normal,
    const float* __restrict__ contact_margin0, const float* __restrict__ contact_margin1,
    const float* __restrict__ contact_penalty_k, const float* __restrict__ contact_material_kd,
    const float* __restrict__ contact_material_mu,
    const Vec3f* __restrict__ contact_lambda, const Vec3f* __restrict__ contact_C0,
    Vec3f* __restrict__ body_forces, Vec3f* __restrict__ body_torques,
    Mat3f* __restrict__ body_h_ll, Mat3f* __restrict__ body_h_al, Mat3f* __restrict__ body_h_aa)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int body_in_group = tid / kContactThreadsPerBody;
    int thread_lane   = tid % kContactThreadsPerBody;

    if (body_in_group >= group_size) return;

    int body_id = body_ids[body_in_group];
    float im = body_inv_mass[body_id];
    if (im <= 0.f) return;

    int n_contacts = body_contact_counts[body_id];
    if (n_contacts <= 0) return;

    Vec3f acc_f(0.f), acc_t(0.f);
    Mat3f acc_hll = Mat3f::zero();
    Mat3f acc_hal = Mat3f::zero();
    Mat3f acc_haa = Mat3f::zero();

    for (int ci = thread_lane; ci < n_contacts; ci += kContactThreadsPerBody) {
        if (ci >= per_body_cap) break;
        int c_idx = body_contact_indices[body_id * per_body_cap + ci];

        int s0 = contact_shape0[c_idx];
        int s1 = contact_shape1[c_idx];
        int b0 = shape_body[s0];
        int b1 = shape_body[s1];
        bool is_body_a = (b0 == body_id);

        Vec3f n = contact_normal[c_idx];
        Vec3f p0l = contact_point0[c_idx];
        Vec3f p1l = contact_point1[c_idx];
        float m0 = contact_margin0[c_idx];
        float m1 = contact_margin1[c_idx];
        float ke = contact_penalty_k[c_idx];
        float kd = contact_material_kd[c_idx];
        float mu = contact_material_mu[c_idx];
        Vec3f lam = contact_lambda[c_idx];
        Vec3f c0 = contact_C0[c_idx];

        // Compute world contact points
        Vec3f pa = (b0 >= 0) ? math::transform_point(body_pos[b0], body_quat[b0], p0l) : p0l;
        Vec3f pb = (b1 >= 0) ? math::transform_point(body_pos[b1], body_quat[b1], p1l) : p1l;
        Vec3f d = pb - pa;
        float C_n_raw = m0 + m1 - math::dot(n, d);

        // AVBD stabilization
        float C0_n = math::dot(n, c0);
        float C_eff = C_n_raw - alpha * C0_n;
        Vec3f C0_t = c0 - n * C0_n;
        Vec3f friction_c0 = C0_t * (1.f - alpha);

        float lambda_n = math::dot(lam, n);
        if (C_n_raw <= 1e-7f && lambda_n <= 0.f) continue;
        if (ke * C_eff + lambda_n <= 0.f && lambda_n <= 0.f) continue;


        ContactForceResult cr = evaluate_contact(
            b0, b1, body_pos, body_quat, body_pos_prev, body_quat_prev, body_com,
            p0l, p1l, n, C_eff, ke, kd, lam, mu, friction_epsilon,
            hard_contacts, dt, friction_c0, is_body_a);

        acc_f = acc_f + cr.force;
        acc_t = acc_t + cr.torque;
        acc_hll = acc_hll + cr.H_ll;
        acc_hal = acc_hal + cr.H_al;
        acc_haa = acc_haa + cr.H_aa;
    }

    // Atomic accumulate to body's scratch
    atomicAdd3(&body_forces[body_id], acc_f);
    atomicAdd3(&body_torques[body_id], acc_t);
    atomicAdd9(&body_h_ll[body_id], acc_hll);
    atomicAdd9(&body_h_al[body_id], acc_hal);
    atomicAdd9(&body_h_aa[body_id], acc_haa);
}

__global__ void solve_rigid_body_kernel(
    const int* __restrict__ body_ids, int group_size,
    float dt, float alpha,
    const Vec3f* __restrict__ pos, const Quatf* __restrict__ quat,
    const Vec3f* __restrict__ inertia_pos, const Quatf* __restrict__ inertia_quat,
    const Vec3f* __restrict__ com, const float* __restrict__ mass,
    const float* __restrict__ inv_mass,
    const Mat3f* __restrict__ inertia_tensor, const Mat3f* __restrict__ inv_inertia,
    const Vec3f* __restrict__ ext_f, const Vec3f* __restrict__ ext_t,
    const Mat3f* __restrict__ ext_hll, const Mat3f* __restrict__ ext_hal,
    const Mat3f* __restrict__ ext_haa,
    int max_jpb, const int* __restrict__ body_joint_count,
    const int*   __restrict__ body_joint_indices,
    int joint_count,
    const int*   __restrict__ joint_type, const int* __restrict__ joint_parent,
    const int*   __restrict__ joint_child,
    const Vec3f* __restrict__ X_p_pos, const Quatf* __restrict__ X_p_quat,
    const Vec3f* __restrict__ X_c_pos, const Quatf* __restrict__ X_c_quat,
    const float* __restrict__ jk_lin, const float* __restrict__ jk_ang,
    const Vec3f* __restrict__ jlam_lin, const Vec3f* __restrict__ jlam_ang,
    const Vec3f* __restrict__ jC0_lin, const Vec3f* __restrict__ jC0_ang,
    const int*   __restrict__ j_is_hard,
    Vec3f* __restrict__ pos_out, Quatf* __restrict__ quat_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= group_size) return;

    int bi = body_ids[tid];
    float im = inv_mass[bi];

    if (im <= 0.f) {
        pos_out[bi] = pos[bi];
        quat_out[bi] = quat[bi];
        return;
    }

    float m = mass[bi];
    float dt2 = dt * dt;
    float inertial_coeff = m / dt2;

    Quatf q = quat[bi];
    Vec3f p = pos[bi];
    Vec3f c = com[bi];
    Mat3f I_local = inertia_tensor[bi];

    // Current COM
    Vec3f com_cur = p + math::quat_rotate(q, c);

    // Target COM (from forward step)
    Vec3f inert_p = inertia_pos[bi];
    Quatf inert_q = inertia_quat[bi];
    Vec3f com_target = inert_p + math::quat_rotate(inert_q, c);

    // --- Inertial force & Hessian ---
    Vec3f f_lin = (com_target - com_cur) * inertial_coeff;

    // Angular inertial: axis-angle of (q_current^-1 * q_target)
    Quatf dq = math::quat_multiply(math::quat_conjugate(q), inert_q);
    if (dq.w < 0.f) dq = -dq;
    Vec3f imag(dq.x, dq.y, dq.z);
    float sin_half = math::length(imag);
    Vec3f theta_body(0.f);
    if (sin_half > 1e-10f) {
        float angle = 2.f * atan2f(sin_half, fabsf(dq.w));
        theta_body = imag * (angle / sin_half);
    }

    Vec3f tau_ang = I_local * theta_body * (1.f / dt2);
    Mat3f R = math::quat_to_matrix(q);
    Vec3f tau_world = R * tau_ang;

    Mat3f H_aa_inertial = R * I_local * math::transpose(R) * (1.f / dt2);

    // --- Assemble 6x6 system ---
    Mat3f H_ll = ext_hll[bi] + Mat3f::identity() * inertial_coeff;
    Mat3f H_al = ext_hal[bi];
    Mat3f H_aa = H_aa_inertial + ext_haa[bi];

    f_lin = f_lin + ext_f[bi];
    tau_world = tau_world + ext_t[bi];

    // --- Loop over adjacent joints ---
    int nj = (max_jpb > 0) ? body_joint_count[bi] : 0;
    for (int ji = 0; ji < nj && ji < max_jpb; ++ji) {
        int j = body_joint_indices[bi * max_jpb + ji];
        if (j < 0 || j >= joint_count) continue;

        int jp = joint_parent[j];
        int jc = joint_child[j];
        bool is_parent = (jp == bi);

        // World anchors
        Vec3f ap = (jp >= 0) ? math::transform_point(pos[jp], quat[jp], X_p_pos[j])
                             : X_p_pos[j];
        Vec3f ac = (jc >= 0) ? math::transform_point(pos[jc], quat[jc], X_c_pos[j])
                             : X_c_pos[j];

        float ke_lin = jk_lin[j];
        float a = j_is_hard[j] ? alpha : 0.f;

        Vec3f jf, jt;
        Mat3f jH_ll, jH_al, jH_aa;
        evaluate_joint_linear(ap, ac, com_cur, ke_lin,
                              jlam_lin[j], jC0_lin[j], a,
                              jf, jt, jH_ll, jH_al, jH_aa, is_parent);

        f_lin = f_lin + jf;
        tau_world = tau_world + jt;
        H_ll = H_ll + jH_ll;
        H_al = H_al + jH_al;
        H_aa = H_aa + jH_aa;

        // Angular constraint (FIXED joints only)
        if (joint_type[j] == JOINT_FIXED) {
            float ke_ang = jk_ang[j];
            Quatf fp = (jp >= 0) ? math::quat_multiply(quat[jp], X_p_quat[j]) : X_p_quat[j];
            Quatf fc = (jc >= 0) ? math::quat_multiply(quat[jc], X_c_quat[j]) : X_c_quat[j];

            Vec3f jt_ang;
            Mat3f jH_aa_ang;
            evaluate_joint_angular(fp, fc, ke_ang,
                                   jlam_ang[j], jC0_ang[j], a,
                                   jt_ang, jH_aa_ang, is_parent);

            tau_world = tau_world + jt_ang;
            H_aa = H_aa + jH_aa_ang;
        }
    }

    // --- Regularize angular Hessian ---
    float eps_a = 1e-9f * ((H_aa(0,0) + H_aa(1,1) + H_aa(2,2)) / 3.f + 1.f);
    H_aa(0,0) += eps_a;
    H_aa(1,1) += eps_a;
    H_aa(2,2) += eps_a;

    // --- Solve 6x6 block system ---
    Vec6Result dx = ldlt6_solve(H_ll, H_aa, H_al, f_lin, tau_world);

    // --- Pose update ---
    // Small-angle rotation approx
    Quatf q_new = math::quat_apply_small_rotation(q, dx.ang);
    Vec3f com_new = com_cur + dx.lin;
    Vec3f p_new = com_new - math::quat_rotate(q_new, c);

    pos_out[bi] = p_new;
    quat_out[bi] = q_new;
}

__global__ void update_duals_contacts_kernel(
    int n, float alpha, float beta,
    float stick_motion_eps, int hard_contacts,
    const Vec3f* __restrict__ body_pos, const Quatf* __restrict__ body_quat,
    const Vec3f* __restrict__ body_pos_prev, const Quatf* __restrict__ body_quat_prev,
    const float* __restrict__ body_inv_mass,
    const int*   __restrict__ contact_shape0, const int* __restrict__ contact_shape1,
    const int*   __restrict__ shape_body,
    const Vec3f* __restrict__ contact_point0, const Vec3f* __restrict__ contact_point1,
    const Vec3f* __restrict__ contact_normal,
    const float* __restrict__ contact_margin0, const float* __restrict__ contact_margin1,
    const float* __restrict__ contact_material_ke, const float* __restrict__ contact_material_mu,
    Vec3f* __restrict__ contact_C0,
    float* __restrict__ contact_penalty_k,
    Vec3f* __restrict__ contact_lambda,
    int*   __restrict__ contact_stick_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int b0 = shape_body[contact_shape0[idx]];
    int b1 = shape_body[contact_shape1[idx]];

    Vec3f n_vec = contact_normal[idx];
    Vec3f p0l = contact_point0[idx];
    Vec3f p1l = contact_point1[idx];
    float m0 = contact_margin0[idx];
    float m1 = contact_margin1[idx];

    Vec3f pa = (b0 >= 0) ? math::transform_point(body_pos[b0], body_quat[b0], p0l) : p0l;
    Vec3f pb = (b1 >= 0) ? math::transform_point(body_pos[b1], body_quat[b1], p1l) : p1l;
    Vec3f d = pb - pa;
    float C_n_raw = m0 + m1 - math::dot(n_vec, d);

    float k = contact_penalty_k[idx];
    Vec3f lam = contact_lambda[idx];
    Vec3f c0 = contact_C0[idx];

    if (hard_contacts) {
        float C0_n = math::dot(n_vec, c0);
        Vec3f C0_t = c0 - n_vec * C0_n;

        float C_stab_n = C_n_raw - alpha * C0_n;
        if (C_n_raw < 0.f) C_stab_n = C_n_raw;

        float lambda_n_old = math::dot(lam, n_vec);
        float lambda_n_new = fmaxf(lambda_n_old + k * C_stab_n, 0.f);

        // Tangential: displacement-based
        Vec3f pa_prev = (b0 >= 0) ? math::transform_point(body_pos_prev[b0], body_quat_prev[b0], p0l) : p0l;
        Vec3f pb_prev = (b1 >= 0) ? math::transform_point(body_pos_prev[b1], body_quat_prev[b1], p1l) : p1l;
        Vec3f du = (pa - pa_prev) - (pb - pb_prev);
        Vec3f du_t = du - n_vec * math::dot(n_vec, du);
        Vec3f residual_t = du_t + C0_t * (1.f - alpha);

        Vec3f lambda_t_old = lam - n_vec * lambda_n_old;
        Vec3f lambda_t_new = lambda_t_old + residual_t * k;

        // Coulomb cone clamp
        float mu = contact_material_mu[idx];
        float lt_len = math::length(lambda_t_new);
        float cone = mu * lambda_n_new;
        if (lt_len > cone && lt_len > 1e-10f) {
            lambda_t_new = lambda_t_new * (cone / lt_len);
        }

        lam = n_vec * lambda_n_new + lambda_t_new;

        // Stick flag
        int flag = STICK_NONE;
        if (lambda_n_new > 0.f && lt_len <= cone) {
            float res_len = math::length(residual_t);
            if (res_len < stick_motion_eps) {
                bool either_kinematic = (b0 < 0 || body_inv_mass[b0] <= 0.f) ||
                                       (b1 < 0 || body_inv_mass[b1] <= 0.f);
                flag = either_kinematic ? STICK_ANCHOR : STICK_DEADZONE;
            }
        }
        contact_stick_flag[idx] = flag;
    } else {
        contact_stick_flag[idx] = STICK_NONE;
    }

    contact_lambda[idx] = lam;

    // Penalty ramp
    if (C_n_raw > 0.f) {
        float ke_max = contact_material_ke[idx];
        k = fminf(k + beta * C_n_raw, ke_max);
    }
    contact_penalty_k[idx] = k;
}

__global__ void update_duals_joints_kernel(
    int n, float alpha, float gamma,
    const Vec3f* __restrict__ body_pos, const Quatf* __restrict__ body_quat,
    const int*   __restrict__ joint_type,
    const int*   __restrict__ joint_parent, const int* __restrict__ joint_child,
    const Vec3f* __restrict__ X_p_pos, const Quatf* __restrict__ X_p_quat,
    const Vec3f* __restrict__ X_c_pos, const Quatf* __restrict__ X_c_quat,
    const int*   __restrict__ j_is_hard,
    Vec3f* __restrict__ C0_lin, Vec3f* __restrict__ C0_ang,
    float* __restrict__ pk_lin, float* __restrict__ pk_ang,
    Vec3f* __restrict__ lam_lin, Vec3f* __restrict__ lam_ang)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    if (!j_is_hard[j]) return;

    int p = joint_parent[j];
    int c = joint_child[j];

    Vec3f ap = (p >= 0) ? math::transform_point(body_pos[p], body_quat[p], X_p_pos[j]) : X_p_pos[j];
    Vec3f ac = (c >= 0) ? math::transform_point(body_pos[c], body_quat[c], X_c_pos[j]) : X_c_pos[j];

    Vec3f C_lin = ac - ap;
    Vec3f C_stab = C_lin - C0_lin[j] * alpha;
    lam_lin[j] = lam_lin[j] + C_stab * pk_lin[j];

    if (joint_type[j] == JOINT_FIXED) {
        Quatf fp = (p >= 0) ? math::quat_multiply(body_quat[p], X_p_quat[j]) : X_p_quat[j];
        Quatf fc = (c >= 0) ? math::quat_multiply(body_quat[c], X_c_quat[j]) : X_c_quat[j];
        Quatf dq = math::quat_multiply(math::quat_conjugate(fp), fc);
        if (dq.w < 0.f) dq = -dq;
        Vec3f imag(dq.x, dq.y, dq.z);
        float sh = math::length(imag);
        Vec3f kappa(0.f);
        if (sh > 1e-10f) {
            float angle = 2.f * atan2f(sh, fabsf(dq.w));
            kappa = imag * (angle / sh);
        }
        Vec3f C_stab_ang = kappa - C0_ang[j] * alpha;
        lam_ang[j] = lam_ang[j] + C_stab_ang * pk_ang[j];
    }
}

__global__ void update_body_velocity_kernel(
    float dt, int n,
    const Vec3f* __restrict__ pos, const Quatf* __restrict__ quat,
    const Vec3f* __restrict__ com,
    Vec3f* __restrict__ pos_prev, Quatf* __restrict__ quat_prev,
    Vec3f* __restrict__ vel, Vec3f* __restrict__ omega,
    int apply_stick_deadzone,
    float stick_trans_eps, float stick_ang_eps,
    int per_body_cap,
    const int* __restrict__ body_contact_counts,
    const int* __restrict__ body_contact_indices,
    const int* __restrict__ contact_stick_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3f p = pos[i];
    Quatf q = quat[i];
    Vec3f p_prev = pos_prev[i];
    Quatf q_prev = quat_prev[i];
    Vec3f c = com[i];

    // Stick deadzone
    if (apply_stick_deadzone) {
        int nc = body_contact_counts[i];
        bool has_anchor = false, has_deadzone = false;
        for (int ci = 0; ci < nc && ci < per_body_cap; ++ci) {
            int c_idx = body_contact_indices[i * per_body_cap + ci];
            int flag = contact_stick_flag[c_idx];
            if (flag == STICK_ANCHOR)   has_anchor = true;
            if (flag == STICK_DEADZONE) has_deadzone = true;
        }
        if (has_deadzone && !has_anchor) {
            Vec3f dp = p - p_prev;
            float trans_disp = math::length(dp);
            Vec3f w_approx = math::quat_velocity(q, q_prev, 1.f);
            float ang_disp = math::length(w_approx);
            if (trans_disp < stick_trans_eps && ang_disp < stick_ang_eps) {
                p = p_prev;
                q = q_prev;
            }
        }
    }

    // BDF1 velocity at COM
    Vec3f com_cur = p + math::quat_rotate(q, c);
    Vec3f com_prev = p_prev + math::quat_rotate(q_prev, c);
    vel[i] = (com_cur - com_prev) * (1.f / dt);
    omega[i] = math::quat_velocity(q, q_prev, dt);

    pos_prev[i] = p;
    quat_prev[i] = q;
}

}  // namespace

// ============================================================================
// Launch wrappers
// ============================================================================

void launch_snapshot_prev(
    const Vec3f* pos, const Quatf* quat,
    Vec3f* pos_prev, Quatf* quat_prev,
    int n, std::uintptr_t stream)
{
    auto s = reinterpret_cast<cudaStream_t>(stream);
    snapshot_prev_kernel<<<grid(n), kBlock, 0, s>>>(pos, quat, pos_prev, quat_prev, n);
}

void launch_forward_step(
    float dt, Vec3f gravity,
    const Vec3f* com, const Mat3f* inertia,
    const float* inv_mass, const Mat3f* inv_inertia,
    Vec3f* pos, Quatf* quat, Vec3f* vel, Vec3f* omega,
    Vec3f* inertia_pos, Quatf* inertia_quat,
    int n, std::uintptr_t stream)
{
    auto s = reinterpret_cast<cudaStream_t>(stream);
    forward_step_kernel<<<grid(n), kBlock, 0, s>>>(
        dt, gravity, com, inertia, inv_mass, inv_inertia,
        pos, quat, vel, omega, inertia_pos, inertia_quat, n);
}

void launch_build_body_contact_list(
    int contact_count, int body_count, int per_body_cap,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    int* body_contact_counts, int* body_contact_indices,
    std::uintptr_t stream)
{
    auto s = reinterpret_cast<cudaStream_t>(stream);
    // Clear counts
    clear_int_array_kernel<<<grid(body_count), kBlock, 0, s>>>(body_contact_counts, body_count);
    if (contact_count > 0) {
        build_body_contact_list_kernel<<<grid(contact_count), kBlock, 0, s>>>(
            contact_count, body_count, per_body_cap,
            contact_shape0, contact_shape1, shape_body,
            body_contact_counts, body_contact_indices);
    }
}

void launch_init_contact_avbd(
    int contact_count,
    float* penalty_k, Vec3f* lambda, Vec3f* C0, int* stick_flag,
    const float* material_ke,
    std::uintptr_t stream)
{
    if (contact_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    init_contact_avbd_kernel<<<grid(contact_count), kBlock, 0, s>>>(
        contact_count, penalty_k, lambda, C0, stick_flag, material_ke);
}

void launch_step_C0_lambda_contacts(
    int contact_count, float alpha, float gamma,
    const Vec3f* body_pos, const Quatf* body_quat,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    const Vec3f* contact_point0, const Vec3f* contact_point1,
    const Vec3f* contact_normal, const float* contact_margin0,
    const float* contact_margin1,
    Vec3f* C0, float* penalty_k, Vec3f* lambda,
    std::uintptr_t stream)
{
    if (contact_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    step_C0_lambda_contacts_kernel<<<grid(contact_count), kBlock, 0, s>>>(
        contact_count, alpha, gamma,
        body_pos, body_quat, contact_shape0, contact_shape1, shape_body,
        contact_point0, contact_point1, contact_normal,
        contact_margin0, contact_margin1,
        C0, penalty_k, lambda);
}

void launch_step_C0_lambda_joints(
    int joint_count, float alpha, float gamma,
    const Vec3f* body_pos, const Quatf* body_quat,
    const int* joint_parent, const int* joint_child,
    const Vec3f* X_p_pos, const Quatf* X_p_quat,
    const Vec3f* X_c_pos, const Quatf* X_c_quat,
    const int* joint_type,
    Vec3f* C0_lin, Vec3f* C0_ang,
    float* penalty_k_lin, float* penalty_k_ang,
    Vec3f* lambda_lin, Vec3f* lambda_ang,
    std::uintptr_t stream)
{
    if (joint_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    step_C0_lambda_joints_kernel<<<grid(joint_count), kBlock, 0, s>>>(
        joint_count, alpha, gamma,
        body_pos, body_quat, joint_parent, joint_child,
        X_p_pos, X_p_quat, X_c_pos, X_c_quat, joint_type,
        C0_lin, C0_ang, penalty_k_lin, penalty_k_ang, lambda_lin, lambda_ang);
}

void launch_zero_scratch(
    Vec3f* forces, Vec3f* torques,
    Mat3f* hessian_ll, Mat3f* hessian_al, Mat3f* hessian_aa,
    int n, std::uintptr_t stream)
{
    auto s = reinterpret_cast<cudaStream_t>(stream);
    zero_scratch_kernel<<<grid(n), kBlock, 0, s>>>(
        forces, torques, hessian_ll, hessian_al, hessian_aa, n);
}

void launch_accumulate_contacts(
    const int* body_ids, int group_size,
    float dt, float alpha, float friction_epsilon,
    int hard_contacts,
    const Vec3f* body_pos, const Quatf* body_quat,
    const Vec3f* body_pos_prev, const Quatf* body_quat_prev,
    const Vec3f* body_com, const float* body_inv_mass,
    int per_body_cap,
    const int* body_contact_counts, const int* body_contact_indices,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    const Vec3f* contact_point0, const Vec3f* contact_point1,
    const Vec3f* contact_normal,
    const float* contact_margin0, const float* contact_margin1,
    const float* contact_penalty_k, const float* contact_material_kd,
    const float* contact_material_mu,
    const Vec3f* contact_lambda, const Vec3f* contact_C0,
    Vec3f* body_forces, Vec3f* body_torques,
    Mat3f* body_hessian_ll, Mat3f* body_hessian_al, Mat3f* body_hessian_aa,
    std::uintptr_t stream)
{
    int threads = group_size * kContactThreadsPerBody;
    if (threads <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    accumulate_contacts_kernel<<<grid(threads), kBlock, 0, s>>>(
        body_ids, group_size,
        dt, alpha, friction_epsilon, hard_contacts,
        body_pos, body_quat, body_pos_prev, body_quat_prev,
        body_com, body_inv_mass,
        per_body_cap, body_contact_counts, body_contact_indices,
        contact_shape0, contact_shape1, shape_body,
        contact_point0, contact_point1, contact_normal,
        contact_margin0, contact_margin1,
        contact_penalty_k, contact_material_kd, contact_material_mu,
        contact_lambda, contact_C0,
        body_forces, body_torques,
        body_hessian_ll, body_hessian_al, body_hessian_aa);
}

void launch_solve_rigid_body(
    const int* body_ids, int group_size,
    float dt, float alpha,
    const Vec3f* pos, const Quatf* quat,
    const Vec3f* inertia_pos, const Quatf* inertia_quat,
    const Vec3f* com, const float* mass, const float* inv_mass,
    const Mat3f* inertia, const Mat3f* inv_inertia,
    const Vec3f* ext_forces, const Vec3f* ext_torques,
    const Mat3f* ext_hessian_ll, const Mat3f* ext_hessian_al,
    const Mat3f* ext_hessian_aa,
    int max_joints_per_body,
    const int* body_joint_count, const int* body_joint_indices,
    int joint_count,
    const int* joint_type, const int* joint_parent, const int* joint_child,
    const Vec3f* X_p_pos, const Quatf* X_p_quat,
    const Vec3f* X_c_pos, const Quatf* X_c_quat,
    const float* joint_penalty_k_lin, const float* joint_penalty_k_ang,
    const Vec3f* joint_lambda_lin, const Vec3f* joint_lambda_ang,
    const Vec3f* joint_C0_lin, const Vec3f* joint_C0_ang,
    const int* joint_is_hard,
    Vec3f* pos_out, Quatf* quat_out,
    std::uintptr_t stream)
{
    if (group_size <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    solve_rigid_body_kernel<<<grid(group_size), kBlock, 0, s>>>(
        body_ids, group_size, dt, alpha,
        pos, quat, inertia_pos, inertia_quat,
        com, mass, inv_mass, inertia, inv_inertia,
        ext_forces, ext_torques, ext_hessian_ll, ext_hessian_al, ext_hessian_aa,
        max_joints_per_body, body_joint_count, body_joint_indices,
        joint_count, joint_type, joint_parent, joint_child,
        X_p_pos, X_p_quat, X_c_pos, X_c_quat,
        joint_penalty_k_lin, joint_penalty_k_ang,
        joint_lambda_lin, joint_lambda_ang,
        joint_C0_lin, joint_C0_ang, joint_is_hard,
        pos_out, quat_out);
}

void launch_update_duals_contacts(
    int contact_count, float alpha, float beta,
    float stick_motion_eps, int hard_contacts,
    const Vec3f* body_pos, const Quatf* body_quat,
    const Vec3f* body_pos_prev, const Quatf* body_quat_prev,
    const float* body_inv_mass,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    const Vec3f* contact_point0, const Vec3f* contact_point1,
    const Vec3f* contact_normal,
    const float* contact_margin0, const float* contact_margin1,
    const float* contact_material_ke, const float* contact_material_mu,
    Vec3f* contact_C0,
    float* contact_penalty_k, Vec3f* contact_lambda,
    int* contact_stick_flag,
    std::uintptr_t stream)
{
    if (contact_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    update_duals_contacts_kernel<<<grid(contact_count), kBlock, 0, s>>>(
        contact_count, alpha, beta, stick_motion_eps, hard_contacts,
        body_pos, body_quat, body_pos_prev, body_quat_prev, body_inv_mass,
        contact_shape0, contact_shape1, shape_body,
        contact_point0, contact_point1, contact_normal,
        contact_margin0, contact_margin1,
        contact_material_ke, contact_material_mu,
        contact_C0, contact_penalty_k, contact_lambda, contact_stick_flag);
}

void launch_update_duals_joints(
    int joint_count, float alpha, float gamma,
    const Vec3f* body_pos, const Quatf* body_quat,
    const int* joint_type, const int* joint_parent, const int* joint_child,
    const Vec3f* X_p_pos, const Quatf* X_p_quat,
    const Vec3f* X_c_pos, const Quatf* X_c_quat,
    const int* joint_is_hard,
    Vec3f* C0_lin, Vec3f* C0_ang,
    float* penalty_k_lin, float* penalty_k_ang,
    Vec3f* lambda_lin, Vec3f* lambda_ang,
    std::uintptr_t stream)
{
    if (joint_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    update_duals_joints_kernel<<<grid(joint_count), kBlock, 0, s>>>(
        joint_count, alpha, gamma,
        body_pos, body_quat, joint_type, joint_parent, joint_child,
        X_p_pos, X_p_quat, X_c_pos, X_c_quat, joint_is_hard,
        C0_lin, C0_ang, penalty_k_lin, penalty_k_ang, lambda_lin, lambda_ang);
}

void launch_update_body_velocity(
    float dt, int body_count,
    const Vec3f* pos, const Quatf* quat,
    const Vec3f* com,
    Vec3f* pos_prev, Quatf* quat_prev,
    Vec3f* vel, Vec3f* omega,
    int apply_stick_deadzone,
    float stick_freeze_translation_eps,
    float stick_freeze_angular_eps,
    int per_body_cap,
    const int* body_contact_counts,
    const int* body_contact_indices,
    const int* contact_stick_flag,
    std::uintptr_t stream)
{
    if (body_count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);
    update_body_velocity_kernel<<<grid(body_count), kBlock, 0, s>>>(
        dt, body_count, pos, quat, com,
        pos_prev, quat_prev, vel, omega,
        apply_stick_deadzone, stick_freeze_translation_eps, stick_freeze_angular_eps,
        per_body_cap, body_contact_counts, body_contact_indices, contact_stick_flag);
}

}  // namespace avbd
}  // namespace rigid
}  // namespace chysx
