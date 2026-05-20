// SPDX-License-Identifier: Apache-2.0
//
// AVBD rigid-body solver kernel declarations.
// Ported from Newton's rigid_vbd_kernels.py to CUDA C++.

#pragma once

#include "../math/matrix.cuh"
#include "../math/quat.cuh"
#include "../math/vec.cuh"

#include <cstdint>

namespace chysx {
namespace rigid {
namespace avbd {

using math::Vec3f;
using math::Quatf;
using math::Mat3f;

// ---------- Device functions (also used in tests) ---------------------------

// In-register 6×6 SPD block LDLT solve.
//   H = [h_ll  h_al^T]    rhs = [rhs_lin]
//       [h_al  h_aa  ]          [rhs_ang]
// Returns (x_lin, x_ang).
struct Vec6Result { Vec3f lin; Vec3f ang; };

// --- Kernel launch wrappers (called from RigidSimulator::step) --------------

void launch_snapshot_prev(
    const Vec3f* pos, const Quatf* quat,
    Vec3f* pos_prev, Quatf* quat_prev,
    int n, std::uintptr_t stream);

void launch_forward_step(
    float dt, Vec3f gravity,
    const Vec3f* com, const Mat3f* inertia,
    const float* inv_mass, const Mat3f* inv_inertia,
    Vec3f* pos, Quatf* quat, Vec3f* vel, Vec3f* omega,
    Vec3f* inertia_pos, Quatf* inertia_quat,
    int n, std::uintptr_t stream);

void launch_build_body_contact_list(
    int contact_count, int body_count, int per_body_cap,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    int* body_contact_counts, int* body_contact_indices,
    std::uintptr_t stream);

void launch_init_contact_avbd(
    int contact_count,
    float* penalty_k, Vec3f* lambda, Vec3f* C0, int* stick_flag,
    const float* material_ke,
    std::uintptr_t stream);

void launch_step_C0_lambda_contacts(
    int contact_count, float alpha, float gamma,
    const Vec3f* body_pos, const Quatf* body_quat,
    const int* contact_shape0, const int* contact_shape1,
    const int* shape_body,
    const Vec3f* contact_point0, const Vec3f* contact_point1,
    const Vec3f* contact_normal, const float* contact_margin0,
    const float* contact_margin1,
    Vec3f* C0, float* penalty_k, Vec3f* lambda,
    std::uintptr_t stream);

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
    std::uintptr_t stream);

void launch_zero_scratch(
    Vec3f* forces, Vec3f* torques,
    Mat3f* hessian_ll, Mat3f* hessian_al, Mat3f* hessian_aa,
    int n, std::uintptr_t stream);

void launch_accumulate_contacts(
    const int* body_ids, int group_size,
    float dt, float alpha, float friction_epsilon,
    int hard_contacts,
    // Body data
    const Vec3f* body_pos, const Quatf* body_quat,
    const Vec3f* body_pos_prev, const Quatf* body_quat_prev,
    const Vec3f* body_com, const float* body_inv_mass,
    // Contact data
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
    // Output (atomically accumulated)
    Vec3f* body_forces, Vec3f* body_torques,
    Mat3f* body_hessian_ll, Mat3f* body_hessian_al, Mat3f* body_hessian_aa,
    std::uintptr_t stream);

void launch_solve_rigid_body(
    const int* body_ids, int group_size,
    float dt, float alpha,
    // Body data
    const Vec3f* pos, const Quatf* quat,
    const Vec3f* inertia_pos, const Quatf* inertia_quat,
    const Vec3f* com, const float* mass, const float* inv_mass,
    const Mat3f* inertia, const Mat3f* inv_inertia,
    // External (contact) contributions
    const Vec3f* ext_forces, const Vec3f* ext_torques,
    const Mat3f* ext_hessian_ll, const Mat3f* ext_hessian_al,
    const Mat3f* ext_hessian_aa,
    // Joint data
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
    // Output
    Vec3f* pos_out, Quatf* quat_out,
    std::uintptr_t stream);

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
    std::uintptr_t stream);

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
    std::uintptr_t stream);

void launch_update_body_velocity(
    float dt, int body_count,
    const Vec3f* pos, const Quatf* quat,
    const Vec3f* com,
    Vec3f* pos_prev, Quatf* quat_prev,
    Vec3f* vel, Vec3f* omega,
    // Stick deadzone
    int apply_stick_deadzone,
    float stick_freeze_translation_eps,
    float stick_freeze_angular_eps,
    int per_body_cap,
    const int* body_contact_counts,
    const int* body_contact_indices,
    const int* contact_stick_flag,
    std::uintptr_t stream);

}  // namespace avbd
}  // namespace rigid
}  // namespace chysx
