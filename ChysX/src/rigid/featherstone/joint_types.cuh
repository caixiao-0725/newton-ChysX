// SPDX-License-Identifier: Apache-2.0
//
// Joint type enumeration and per-joint coordinate computations (jcalc_*).
// Direct C++/CUDA port of Newton's Warp kernels from
// newton/_src/solvers/featherstone/kernels.py

#pragma once

#include "spatial_math.cuh"

namespace chysx {
namespace rigid {

// ============================================================================
// Joint types — values match Newton JointType enum
// ============================================================================
namespace FJointType {
    constexpr int PRISMATIC = 0;
    constexpr int REVOLUTE  = 1;
    constexpr int BALL      = 2;
    constexpr int FIXED     = 3;
    constexpr int FREE      = 4;
    constexpr int DISTANCE  = 5;
    constexpr int D6        = 6;
    constexpr int CABLE     = 7;
}

// ============================================================================
// BodyFlags — values match Newton BodyFlags enum
// ============================================================================
namespace BodyFlags {
    constexpr int DYNAMIC   = 1 << 0;
    constexpr int KINEMATIC = 1 << 1;
}

// ============================================================================
// D6 joint helper: compute 2D rotational DOFs
// ============================================================================

CHYSX_HDI void compute_2d_rotational_dofs(
    const Vec3f& axis_0_in, const Vec3f& axis_1_in,
    float q0, float q1, float qd0, float qd1,
    Quatf& rot_out, Vec3f& vel_out)
{
    Quatf q_off = quat_from_matrix(mat3_from_cols(axis_0_in, axis_1_in, cross(axis_0_in, axis_1_in)));

    Vec3f local_0 = quat_rotate(q_off, Vec3f(1.f, 0.f, 0.f));
    Vec3f local_1 = quat_rotate(q_off, Vec3f(0.f, 1.f, 0.f));

    Vec3f a0 = local_0;
    Quatf q_0 = quat_from_axis_angle(a0, q0);

    Vec3f a1 = quat_rotate(q_0, local_1);
    Quatf q_1 = quat_from_axis_angle(a1, q1);

    rot_out = quat_multiply(q_1, q_0);
    vel_out = a0 * qd0 + a1 * qd1;
}

// ============================================================================
// D6 joint helper: compute 3D rotational DOFs
// ============================================================================

CHYSX_HDI void compute_3d_rotational_dofs(
    const Vec3f& axis_0_in, const Vec3f& axis_1_in, const Vec3f& axis_2_in,
    float q0, float q1, float q2, float qd0, float qd1, float qd2,
    Quatf& rot_out, Vec3f& vel_out)
{
    Quatf q_off = quat_from_matrix(mat3_from_cols(axis_0_in, axis_1_in, axis_2_in));

    Vec3f local_0 = quat_rotate(q_off, Vec3f(1.f, 0.f, 0.f));
    Vec3f local_1 = quat_rotate(q_off, Vec3f(0.f, 1.f, 0.f));
    Vec3f local_2 = quat_rotate(q_off, Vec3f(0.f, 0.f, 1.f));

    Vec3f a0 = local_0;
    Quatf q_0 = quat_from_axis_angle(a0, q0);

    Vec3f a1 = quat_rotate(q_0, local_1);
    Quatf q_1 = quat_from_axis_angle(a1, q1);

    Vec3f a2 = quat_rotate(quat_multiply(q_1, q_0), local_2);
    Quatf q_2 = quat_from_axis_angle(a2, q2);

    rot_out = quat_multiply(q_2, quat_multiply(q_1, q_0));
    vel_out = a0 * qd0 + a1 * qd1 + a2 * qd2;
}

// ============================================================================
// jcalc_transform: compute the relative transform across a joint
// ============================================================================

CHYSX_HDI Transform7 jcalc_transform(
    int type,
    const Vec3f* joint_axis,
    int axis_start,
    int lin_axis_count,
    int ang_axis_count,
    const float* joint_q,
    int q_start)
{
    if (type == FJointType::PRISMATIC) {
        float q = joint_q[q_start];
        Vec3f axis = joint_axis[axis_start];
        return Transform7(axis * q, quat_identity());
    }

    if (type == FJointType::REVOLUTE) {
        float q = joint_q[q_start];
        Vec3f axis = joint_axis[axis_start];
        return Transform7(Vec3f(), quat_from_axis_angle(axis, q));
    }

    if (type == FJointType::BALL) {
        float qx = joint_q[q_start + 0];
        float qy = joint_q[q_start + 1];
        float qz = joint_q[q_start + 2];
        float qw = joint_q[q_start + 3];
        return Transform7(Vec3f(), Quatf(qx, qy, qz, qw));
    }

    if (type == FJointType::FIXED) {
        return tf_identity();
    }

    if (type == FJointType::FREE || type == FJointType::DISTANCE) {
        float px = joint_q[q_start + 0];
        float py = joint_q[q_start + 1];
        float pz = joint_q[q_start + 2];
        float qx = joint_q[q_start + 3];
        float qy = joint_q[q_start + 4];
        float qz = joint_q[q_start + 5];
        float qw = joint_q[q_start + 6];
        return Transform7(Vec3f(px, py, pz), Quatf(qx, qy, qz, qw));
    }

    if (type == FJointType::D6) {
        Vec3f pos(0.f);
        Quatf rot = quat_identity();

        if (lin_axis_count > 0)
            pos += joint_axis[axis_start + 0] * joint_q[q_start + 0];
        if (lin_axis_count > 1)
            pos += joint_axis[axis_start + 1] * joint_q[q_start + 1];
        if (lin_axis_count > 2)
            pos += joint_axis[axis_start + 2] * joint_q[q_start + 2];

        int ia = axis_start + lin_axis_count;
        int iq = q_start + lin_axis_count;
        if (ang_axis_count == 1) {
            rot = quat_from_axis_angle(joint_axis[ia], joint_q[iq]);
        }
        if (ang_axis_count == 2) {
            Vec3f vel_unused;
            compute_2d_rotational_dofs(
                joint_axis[ia + 0], joint_axis[ia + 1],
                joint_q[iq + 0], joint_q[iq + 1], 0.f, 0.f,
                rot, vel_unused);
        }
        if (ang_axis_count == 3) {
            Vec3f vel_unused;
            compute_3d_rotational_dofs(
                joint_axis[ia + 0], joint_axis[ia + 1], joint_axis[ia + 2],
                joint_q[iq + 0], joint_q[iq + 1], joint_q[iq + 2],
                0.f, 0.f, 0.f,
                rot, vel_unused);
        }

        return Transform7(pos, rot);
    }

    return tf_identity();
}

// ============================================================================
// jcalc_motion: compute motion subspace vectors and joint velocity
// ============================================================================

CHYSX_HDI SpatialVector jcalc_motion(
    int type,
    const Vec3f* joint_axis,
    int lin_axis_count,
    int ang_axis_count,
    const Transform7& X_sc,
    const float* joint_qd,
    int qd_start,
    SpatialVector* joint_S_s)
{
    if (type == FJointType::PRISMATIC) {
        Vec3f axis = joint_axis[qd_start];
        SpatialVector S_s = transform_twist(X_sc, SpatialVector(axis, Vec3f()));
        SpatialVector v_j_s = S_s * joint_qd[qd_start];
        joint_S_s[qd_start] = S_s;
        return v_j_s;
    }

    if (type == FJointType::REVOLUTE) {
        Vec3f axis = joint_axis[qd_start];
        SpatialVector S_s = transform_twist(X_sc, SpatialVector(Vec3f(), axis));
        SpatialVector v_j_s = S_s * joint_qd[qd_start];
        joint_S_s[qd_start] = S_s;
        return v_j_s;
    }

    if (type == FJointType::D6) {
        SpatialVector v_j_s;
        if (lin_axis_count > 0) {
            Vec3f axis = joint_axis[qd_start + 0];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(axis, Vec3f()));
            v_j_s += S_s * joint_qd[qd_start + 0];
            joint_S_s[qd_start + 0] = S_s;
        }
        if (lin_axis_count > 1) {
            Vec3f axis = joint_axis[qd_start + 1];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(axis, Vec3f()));
            v_j_s += S_s * joint_qd[qd_start + 1];
            joint_S_s[qd_start + 1] = S_s;
        }
        if (lin_axis_count > 2) {
            Vec3f axis = joint_axis[qd_start + 2];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(axis, Vec3f()));
            v_j_s += S_s * joint_qd[qd_start + 2];
            joint_S_s[qd_start + 2] = S_s;
        }
        if (ang_axis_count > 0) {
            Vec3f axis = joint_axis[qd_start + lin_axis_count + 0];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(Vec3f(), axis));
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0];
            joint_S_s[qd_start + lin_axis_count + 0] = S_s;
        }
        if (ang_axis_count > 1) {
            Vec3f axis = joint_axis[qd_start + lin_axis_count + 1];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(Vec3f(), axis));
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1];
            joint_S_s[qd_start + lin_axis_count + 1] = S_s;
        }
        if (ang_axis_count > 2) {
            Vec3f axis = joint_axis[qd_start + lin_axis_count + 2];
            SpatialVector S_s = transform_twist(X_sc, SpatialVector(Vec3f(), axis));
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2];
            joint_S_s[qd_start + lin_axis_count + 2] = S_s;
        }
        return v_j_s;
    }

    if (type == FJointType::BALL) {
        SpatialVector S_0 = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 1.f, 0.f, 0.f));
        SpatialVector S_1 = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 0.f, 1.f, 0.f));
        SpatialVector S_2 = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 0.f, 0.f, 1.f));
        joint_S_s[qd_start + 0] = S_0;
        joint_S_s[qd_start + 1] = S_1;
        joint_S_s[qd_start + 2] = S_2;
        return S_0 * joint_qd[qd_start + 0]
             + S_1 * joint_qd[qd_start + 1]
             + S_2 * joint_qd[qd_start + 2];
    }

    if (type == FJointType::FIXED) {
        return SpatialVector();
    }

    if (type == FJointType::FREE || type == FJointType::DISTANCE) {
        SpatialVector v_j_s = transform_twist(X_sc, SpatialVector(
            joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2],
            joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]));

        joint_S_s[qd_start + 0] = transform_twist(X_sc, SpatialVector(1.f, 0.f, 0.f, 0.f, 0.f, 0.f));
        joint_S_s[qd_start + 1] = transform_twist(X_sc, SpatialVector(0.f, 1.f, 0.f, 0.f, 0.f, 0.f));
        joint_S_s[qd_start + 2] = transform_twist(X_sc, SpatialVector(0.f, 0.f, 1.f, 0.f, 0.f, 0.f));
        joint_S_s[qd_start + 3] = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 1.f, 0.f, 0.f));
        joint_S_s[qd_start + 4] = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 0.f, 1.f, 0.f));
        joint_S_s[qd_start + 5] = transform_twist(X_sc, SpatialVector(0.f, 0.f, 0.f, 0.f, 0.f, 1.f));
        return v_j_s;
    }

    return SpatialVector();
}

// ============================================================================
// joint_force: PD drive + joint limits for a single DOF
// ============================================================================

CHYSX_HDI float joint_force(
    float q, float qd,
    float target_pos, float target_vel,
    float target_ke, float target_kd,
    float limit_lower, float limit_upper,
    float limit_ke, float limit_kd)
{
    float limit_f = 0.f;
    float damping_f = 0.f;
    float target_f = target_ke * (target_pos - q) + target_kd * (target_vel - qd);

    if (q < limit_lower) {
        limit_f = limit_ke * (limit_lower - q);
        damping_f = -limit_kd * qd;
        target_f = 0.f;
    } else if (q > limit_upper) {
        limit_f = limit_ke * (limit_upper - q);
        damping_f = -limit_kd * qd;
        target_f = 0.f;
    }

    return limit_f + damping_f + target_f;
}

// ============================================================================
// jcalc_tau: compute generalized forces from spatial body force
// ============================================================================

CHYSX_HDI void jcalc_tau(
    int type,
    const float* target_ke,
    const float* target_kd,
    const float* limit_ke_arr,
    const float* limit_kd_arr,
    const SpatialVector* joint_S_s,
    const float* joint_q,
    const float* joint_qd,
    const float* joint_f,
    const float* target_pos,
    const float* target_vel,
    const float* limit_lower,
    const float* limit_upper,
    int coord_start, int dof_start,
    int lin_axis_count, int ang_axis_count,
    const SpatialVector& body_f_s,
    float* tau)
{
    if (type == FJointType::BALL) {
        for (int i = 0; i < 3; ++i) {
            SpatialVector S_s = joint_S_s[dof_start + i];
            tau[dof_start + i] = -spatial_dot(S_s, body_f_s) + joint_f[dof_start + i];
        }
        return;
    }

    if (type == FJointType::FREE || type == FJointType::DISTANCE) {
        for (int i = 0; i < 6; ++i) {
            SpatialVector S_s = joint_S_s[dof_start + i];
            tau[dof_start + i] = -spatial_dot(S_s, body_f_s) + joint_f[dof_start + i];
        }
        return;
    }

    if (type == FJointType::PRISMATIC || type == FJointType::REVOLUTE || type == FJointType::D6) {
        int axis_count = lin_axis_count + ang_axis_count;
        for (int i = 0; i < axis_count; ++i) {
            int j = dof_start + i;
            SpatialVector S_s = joint_S_s[j];

            float q = joint_q[coord_start + i];
            float qd = joint_qd[j];

            float drive_f = joint_force(
                q, qd,
                target_pos[j], target_vel[j],
                target_ke[j], target_kd[j],
                limit_lower[j], limit_upper[j],
                limit_ke_arr[j], limit_kd_arr[j]);

            tau[j] = -spatial_dot(S_s, body_f_s) + drive_f + joint_f[j];
        }
        return;
    }

    // FIXED: no tau
}

// ============================================================================
// jcalc_integrate: symplectic Euler integration of generalized coordinates
// ============================================================================

CHYSX_HDI void jcalc_integrate(
    int parent,
    const Transform7& joint_X_c,
    const Vec3f& body_com_child,
    int type,
    const float* joint_q,
    const float* joint_qd,
    const float* joint_qdd,
    int coord_start, int dof_start,
    int lin_axis_count, int ang_axis_count,
    float dt,
    float* joint_q_new,
    float* joint_qd_new)
{
    if (type == FJointType::FIXED)
        return;

    if (type == FJointType::PRISMATIC || type == FJointType::REVOLUTE) {
        float qdd = joint_qdd[dof_start];
        float qd = joint_qd[dof_start];
        float q = joint_q[coord_start];

        float qd_new = qd + qdd * dt;
        float q_new = q + qd_new * dt;

        joint_qd_new[dof_start] = qd_new;
        joint_q_new[coord_start] = q_new;
        return;
    }

    if (type == FJointType::BALL) {
        Vec3f m_j(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2]);
        Vec3f w_j(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2]);
        Quatf r_j(joint_q[coord_start + 0], joint_q[coord_start + 1],
                  joint_q[coord_start + 2], joint_q[coord_start + 3]);

        Vec3f w_j_new = w_j + m_j * dt;
        Quatf omega_q(w_j_new.x, w_j_new.y, w_j_new.z, 0.f);
        Quatf drdt_j = quat_multiply(omega_q, r_j) * 0.5f;
        Quatf r_j_new = quat_normalize(r_j + drdt_j * dt);

        joint_q_new[coord_start + 0] = r_j_new.x;
        joint_q_new[coord_start + 1] = r_j_new.y;
        joint_q_new[coord_start + 2] = r_j_new.z;
        joint_q_new[coord_start + 3] = r_j_new.w;

        joint_qd_new[dof_start + 0] = w_j_new.x;
        joint_qd_new[dof_start + 1] = w_j_new.y;
        joint_qd_new[dof_start + 2] = w_j_new.z;
        return;
    }

    if (type == FJointType::FREE || type == FJointType::DISTANCE) {
        if (parent < 0) {
            Vec3f a_parent(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2]);
            Vec3f alpha(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5]);

            Vec3f v_parent(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2]);
            Vec3f omega(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5]);

            Vec3f p(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2]);
            Quatf r(joint_q[coord_start + 3], joint_q[coord_start + 4],
                    joint_q[coord_start + 5], joint_q[coord_start + 6]);

            Vec3f r_com_joint = tf_point(tf_inverse(joint_X_c), body_com_child);
            Vec3f x_com = p + quat_rotate(r, r_com_joint);
            Vec3f v_com = v_parent + cross(omega, x_com);
            Vec3f a_com = a_parent + cross(alpha, x_com) + cross(omega, v_com);

            Vec3f omega_new = omega + alpha * dt;
            Vec3f v_com_new = v_com + a_com * dt;

            Quatf omega_q(omega_new.x, omega_new.y, omega_new.z, 0.f);
            Quatf drdt = quat_multiply(omega_q, r) * 0.5f;
            Quatf r_new = quat_normalize(r + drdt * dt);
            Vec3f x_com_new = x_com + v_com_new * dt;
            Vec3f p_new = x_com_new - quat_rotate(r_new, r_com_joint);
            Vec3f v_parent_new = v_com_new - cross(omega_new, x_com_new);

            joint_q_new[coord_start + 0] = p_new.x;
            joint_q_new[coord_start + 1] = p_new.y;
            joint_q_new[coord_start + 2] = p_new.z;
            joint_q_new[coord_start + 3] = r_new.x;
            joint_q_new[coord_start + 4] = r_new.y;
            joint_q_new[coord_start + 5] = r_new.z;
            joint_q_new[coord_start + 6] = r_new.w;

            joint_qd_new[dof_start + 0] = v_parent_new.x;
            joint_qd_new[dof_start + 1] = v_parent_new.y;
            joint_qd_new[dof_start + 2] = v_parent_new.z;
            joint_qd_new[dof_start + 3] = omega_new.x;
            joint_qd_new[dof_start + 4] = omega_new.y;
            joint_qd_new[dof_start + 5] = omega_new.z;
            return;
        }

        // Descendant FREE/DISTANCE: stay in parent-origin coordinates
        Vec3f a_s(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2]);
        Vec3f m_s(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5]);

        Vec3f v_s(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2]);
        Vec3f w_s(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5]);

        w_s = w_s + m_s * dt;
        v_s = v_s + a_s * dt;

        Vec3f p_s(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2]);
        Vec3f dpdt_s = v_s + cross(w_s, p_s);

        Quatf r_s(joint_q[coord_start + 3], joint_q[coord_start + 4],
                  joint_q[coord_start + 5], joint_q[coord_start + 6]);
        Quatf omega_q(w_s.x, w_s.y, w_s.z, 0.f);
        Quatf drdt_s = quat_multiply(omega_q, r_s) * 0.5f;

        Vec3f p_s_new = p_s + dpdt_s * dt;
        Quatf r_s_new = quat_normalize(r_s + drdt_s * dt);

        joint_q_new[coord_start + 0] = p_s_new.x;
        joint_q_new[coord_start + 1] = p_s_new.y;
        joint_q_new[coord_start + 2] = p_s_new.z;
        joint_q_new[coord_start + 3] = r_s_new.x;
        joint_q_new[coord_start + 4] = r_s_new.y;
        joint_q_new[coord_start + 5] = r_s_new.z;
        joint_q_new[coord_start + 6] = r_s_new.w;

        joint_qd_new[dof_start + 0] = v_s.x;
        joint_qd_new[dof_start + 1] = v_s.y;
        joint_qd_new[dof_start + 2] = v_s.z;
        joint_qd_new[dof_start + 3] = w_s.x;
        joint_qd_new[dof_start + 4] = w_s.y;
        joint_qd_new[dof_start + 5] = w_s.z;
        return;
    }

    if (type == FJointType::D6) {
        int axis_count = lin_axis_count + ang_axis_count;
        for (int i = 0; i < axis_count; ++i) {
            float qdd = joint_qdd[dof_start + i];
            float qd = joint_qd[dof_start + i];
            float q = joint_q[coord_start + i];

            float qd_new = qd + qdd * dt;
            float q_new = q + qd_new * dt;

            joint_qd_new[dof_start + i] = qd_new;
            joint_q_new[coord_start + i] = q_new;
        }
        return;
    }
}

}  // namespace rigid
}  // namespace chysx
