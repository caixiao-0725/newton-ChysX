// SPDX-License-Identifier: Apache-2.0
//
// FeatherstoneSolver implementation — CUDA kernel launches and step().

#include "featherstone_solver.h"
#include "featherstone_kernels.cuh"

#include <cuda_runtime.h>
#include <cstring>

namespace chysx {
namespace rigid {

namespace {
constexpr int kBlockSize = 128;

int div_up(int n, int d) { return (n + d - 1) / d; }

void cuda_zero(void* ptr, size_t bytes, cudaStream_t stream) {
    cudaMemsetAsync(ptr, 0, bytes, stream);
}

void cuda_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

}  // namespace

// ============================================================================
// set_model
// ============================================================================
void FeatherstoneSolver::set_model(const ArticulationModel& model) {
    model_ = &model;

    // Check for kinematic bodies/joints by copying flags back to host
    if (model.body_count > 0) {
        // body_flags is a CudaArray set from Python with host+device data
        has_kinematic_bodies_ = false;
        has_kinematic_joints_ = false;
        // We read from the CPU side which Python populates before calling set_model
        for (int i = 0; i < model.body_count; ++i) {
            if (model.body_flags.cpu_data()[i] & BodyFlags::KINEMATIC) {
                has_kinematic_bodies_ = true;
                has_kinematic_joints_ = true;
                break;
            }
        }
    }

    compute_articulation_indices();
    allocate_buffers();

    // Compute spatial inertia at COM
    if (model.body_count > 0) {
        cudaStream_t stream = 0;
        compute_spatial_inertia_kernel<<<div_up(model.body_count, kBlockSize), kBlockSize, 0, stream>>>(
            model.body_inertia.gpu_data(), model.body_mass.gpu_data(),
            body_I_m_.gpu_data(), model.body_count);

        compute_com_transforms_kernel<<<div_up(model.body_count, kBlockSize), kBlockSize, 0, stream>>>(
            model.body_com.gpu_data(), body_X_com_.gpu_data(), model.body_count);
    }
}

// ============================================================================
// compute_articulation_indices
// ============================================================================
void FeatherstoneSolver::compute_articulation_indices() {
    const auto& model = *model_;
    if (model.joint_count == 0) return;

    // Read from CPU side of the CudaArrays (populated by Python before set_model)
    const int* art_start = model.articulation_start.cpu_data();
    const int* jq_start = model.joint_q_start.cpu_data();
    const int* jqd_start = model.joint_qd_start.cpu_data();

    J_size_ = 0; M_size_ = 0; H_size_ = 0;

    std::vector<int> h_J_start, h_M_start, h_H_start;
    std::vector<int> h_M_rows, h_H_rows, h_J_rows, h_J_cols;
    std::vector<int> h_dof_start, h_coord_start;

    for (int i = 0; i < model.articulation_count; ++i) {
        int first_joint = art_start[i];
        int last_joint = art_start[i + 1];
        int first_dof = jqd_start[first_joint];
        int last_dof = jqd_start[last_joint];
        int joint_count = last_joint - first_joint;
        int dof_count = last_dof - first_dof;

        h_J_start.push_back(J_size_);
        h_M_start.push_back(M_size_);
        h_H_start.push_back(H_size_);
        h_dof_start.push_back(first_dof);
        h_coord_start.push_back(jq_start[first_joint]);

        h_M_rows.push_back(joint_count * 6);
        h_H_rows.push_back(dof_count);
        h_J_rows.push_back(joint_count * 6);
        h_J_cols.push_back(dof_count);

        J_size_ += 6 * joint_count * dof_count;
        M_size_ += 6 * joint_count * 6 * joint_count;
        H_size_ += dof_count * dof_count;
    }

    // Upload host vectors to CudaArrays (allocate host+device, copy to cpu, then h2d)
    auto upload = [](CudaArray<int>& arr, const std::vector<int>& v) {
        int n = static_cast<int>(v.size());
        arr.resize(n);
        std::memcpy(arr.cpu_data(), v.data(), n * sizeof(int));
        arr.copy_to_device();
    };
    upload(articulation_J_start_, h_J_start);
    upload(articulation_M_start_, h_M_start);
    upload(articulation_H_start_, h_H_start);
    upload(articulation_M_rows_, h_M_rows);
    upload(articulation_H_rows_, h_H_rows);
    upload(articulation_J_rows_, h_J_rows);
    upload(articulation_J_cols_, h_J_cols);
    upload(articulation_dof_start_, h_dof_start);
    upload(articulation_coord_start_, h_coord_start);
}

// ============================================================================
// allocate_buffers
// ============================================================================
void FeatherstoneSolver::allocate_buffers() {
    const auto& model = *model_;

    if (model.body_count > 0) {
        body_I_m_.resize(model.body_count);
        body_X_com_.resize(model.body_count);
        body_q_.resize(model.body_count);
        body_q_com_.resize(model.body_count);
        body_qd_.resize(model.body_count);
        body_v_s_.resize(model.body_count);
        body_a_s_.resize(model.body_count);
        body_f_s_.resize(model.body_count);
        body_ft_s_.resize(model.body_count);
        body_I_s_.resize(model.body_count);
        body_q_prev_.resize(model.body_count);
    }

    if (model.joint_dof_count > 0) {
        joint_S_s_.resize(model.joint_dof_count);
        joint_tau_.resize(model.joint_dof_count);
        joint_qdd_.resize(model.joint_dof_count);
        joint_qd_internal_in_.resize(model.joint_dof_count);
        joint_qd_internal_out_.resize(model.joint_dof_count);
        joint_f_internal_.resize(model.joint_dof_count);
    }

    if (J_size_ > 0) {
        J_.resize(J_size_);
        M_.resize(M_size_);
        P_.resize(J_size_);
        H_.resize(H_size_);
        L_.resize(H_size_);
    }
}

// ============================================================================
// step
// ============================================================================
void FeatherstoneSolver::step(
    const ArticulationState& state_in,
    ArticulationState& state_out,
    const ControlInputs& control,
    float* body_f_ext_ptr,     // [body_count * 6]
    float dt,
    std::uintptr_t cuda_stream_int)
{
    if (!model_ || model_->joint_count == 0) return;

    const auto& model = *model_;
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream_int);
    SpatialVector* body_f_ext = reinterpret_cast<SpatialVector*>(body_f_ext_ptr);

    int n_art = model.articulation_count;
    int n_bodies = model.body_count;
    int n_joints = model.joint_count;
    int n_dofs = model.joint_dof_count;

    // ---- 1. Forward kinematics ----
    eval_rigid_fk_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_q_start.gpu_data(),
        model.joint_qd_start.gpu_data(),
        state_in.joint_q.gpu_data(),
        model.joint_X_p.gpu_data(),
        model.joint_X_c.gpu_data(),
        body_X_com_.gpu_data(),
        model.joint_axis.gpu_data(),
        model.joint_dof_dim.gpu_data(),
        body_q_.gpu_data(),
        body_q_com_.gpu_data(),
        n_art);

    // Save body_q_prev for descendant correction if needed
    if (model.n_descendant_free_distance > 0) {
        cuda_copy(body_q_prev_.gpu_data(), body_q_.gpu_data(),
                  n_bodies * sizeof(Transform7), stream);
    }

    // ---- 2. Convert body forces: COM → origin frame ----
    convert_body_force_com_to_origin_kernel<<<div_up(n_bodies, kBlockSize), kBlockSize, 0, stream>>>(
        body_q_.gpu_data(), body_X_com_.gpu_data(), body_f_ext, n_bodies);

    // ---- 3. Accumulate FREE/DISTANCE joint_f into body forces ----
    accumulate_free_distance_joint_f_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
        model.joint_type.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_qd_start.gpu_data(),
        body_q_.gpu_data(),
        body_X_com_.gpu_data(),
        control.joint_f,
        body_f_ext,
        n_joints);

    // ---- 4. Zero kinematic body forces ----
    if (has_kinematic_bodies_) {
        zero_kinematic_body_forces_kernel<<<div_up(n_bodies, kBlockSize), kBlockSize, 0, stream>>>(
            model.body_flags.gpu_data(), body_f_ext, n_bodies);
    }

    // ---- 5. Convert qd public → internal ----
    convert_qd_public_to_internal_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_qd_start.gpu_data(),
        model.joint_X_p.gpu_data(),
        body_q_.gpu_data(),
        model.body_com.gpu_data(),
        state_in.joint_qd.gpu_data(),
        joint_qd_internal_in_.gpu_data(),
        n_joints);

    // ---- 6. Convert joint_f public → internal ----
    convert_f_public_to_internal_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
        model.joint_type.gpu_data(),
        model.joint_qd_start.gpu_data(),
        control.joint_f,
        joint_f_internal_.gpu_data(),
        n_joints);

    // ---- 7. Forward RNEA: velocities, Coriolis, spatial inertia ----
    cuda_zero(body_f_s_.gpu_data(), n_bodies * sizeof(SpatialVector), stream);

    eval_rigid_id_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_qd_start.gpu_data(),
        joint_qd_internal_in_.gpu_data(),
        model.joint_axis.gpu_data(),
        model.joint_dof_dim.gpu_data(),
        body_I_m_.gpu_data(),
        body_q_.gpu_data(),
        body_q_com_.gpu_data(),
        model.joint_X_p.gpu_data(),
        model.body_world.gpu_data(),
        model.gravity.gpu_data(),
        joint_S_s_.gpu_data(),
        body_I_s_.gpu_data(),
        body_v_s_.gpu_data(),
        body_f_s_.gpu_data(),
        body_a_s_.gpu_data(),
        n_art);

    // ---- 8. Backward RNEA: generalized forces ----
    cuda_zero(body_ft_s_.gpu_data(), n_bodies * sizeof(SpatialVector), stream);

    eval_rigid_tau_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_q_start.gpu_data(),
        model.joint_qd_start.gpu_data(),
        model.joint_dof_dim.gpu_data(),
        control.joint_target_pos,
        control.joint_target_vel,
        state_in.joint_q.gpu_data(),
        joint_qd_internal_in_.gpu_data(),
        joint_f_internal_.gpu_data(),
        model.joint_target_ke.gpu_data(),
        model.joint_target_kd.gpu_data(),
        model.joint_limit_lower.gpu_data(),
        model.joint_limit_upper.gpu_data(),
        model.joint_limit_ke.gpu_data(),
        model.joint_limit_kd.gpu_data(),
        joint_S_s_.gpu_data(),
        body_f_s_.gpu_data(),
        body_f_ext,
        body_ft_s_.gpu_data(),
        joint_tau_.gpu_data(),
        n_art);

    // ---- 9. Build Jacobian ----
    eval_rigid_jacobian_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        articulation_J_start_.gpu_data(),
        model.joint_ancestor.gpu_data(),
        model.joint_qd_start.gpu_data(),
        joint_S_s_.gpu_data(),
        J_.gpu_data(),
        n_art);

    // ---- 10. Build block-diagonal mass matrix M ----
    eval_rigid_mass_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        articulation_M_start_.gpu_data(),
        body_I_s_.gpu_data(),
        M_.gpu_data(),
        n_art);

    // ---- 11. P = M * J ----
    eval_dense_gemm_batched_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        articulation_M_rows_.gpu_data(),
        articulation_J_cols_.gpu_data(),
        articulation_J_rows_.gpu_data(),
        false, false,
        articulation_M_start_.gpu_data(),
        articulation_J_start_.gpu_data(),
        articulation_J_start_.gpu_data(),  // P uses same layout as J
        M_.gpu_data(),
        J_.gpu_data(),
        P_.gpu_data(),
        n_art);

    // ---- 12. H = J^T * P ----
    eval_dense_gemm_batched_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        articulation_J_cols_.gpu_data(),
        articulation_J_cols_.gpu_data(),
        articulation_J_rows_.gpu_data(),
        true, false,
        articulation_J_start_.gpu_data(),
        articulation_J_start_.gpu_data(),
        articulation_H_start_.gpu_data(),
        J_.gpu_data(),
        P_.gpu_data(),
        H_.gpu_data(),
        n_art);

    // ---- 13. Cholesky: L L^T = H + diag(armature) ----
    cuda_zero(L_.gpu_data(), H_size_ * sizeof(float), stream);

    eval_dense_cholesky_batched_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        articulation_H_start_.gpu_data(),
        articulation_H_rows_.gpu_data(),
        articulation_dof_start_.gpu_data(),
        H_.gpu_data(),
        model.joint_armature.gpu_data(),
        L_.gpu_data(),
        n_art);

    // ---- 14. Solve: H qdd = tau ----
    cuda_zero(joint_qdd_.gpu_data(), n_dofs * sizeof(float), stream);

    eval_dense_solve_batched_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        articulation_H_start_.gpu_data(),
        articulation_H_rows_.gpu_data(),
        articulation_dof_start_.gpu_data(),
        L_.gpu_data(),
        joint_tau_.gpu_data(),
        joint_qdd_.gpu_data(),
        n_art);

    // ---- 15. Zero kinematic qdd ----
    if (has_kinematic_joints_) {
        zero_kinematic_joint_qdd_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
            model.joint_child.gpu_data(),
            model.body_flags.gpu_data(),
            model.joint_qd_start.gpu_data(),
            joint_qdd_.gpu_data(),
            n_joints);
    }

    // ---- 16. Integrate generalized joints ----
    integrate_generalized_joints_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_q_start.gpu_data(),
        model.joint_qd_start.gpu_data(),
        model.joint_dof_dim.gpu_data(),
        model.joint_X_c.gpu_data(),
        model.body_com.gpu_data(),
        state_in.joint_q.gpu_data(),
        joint_qd_internal_in_.gpu_data(),
        joint_qdd_.gpu_data(),
        dt,
        state_out.joint_q.gpu_data(),
        joint_qd_internal_out_.gpu_data(),
        n_joints);

    // ---- 17. Copy kinematic joint state ----
    if (has_kinematic_joints_) {
        copy_kinematic_joint_state_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
            model.joint_child.gpu_data(),
            model.body_flags.gpu_data(),
            model.joint_q_start.gpu_data(),
            model.joint_qd_start.gpu_data(),
            state_in.joint_q.gpu_data(),
            state_in.joint_qd.gpu_data(),
            state_out.joint_q.gpu_data(),
            joint_qd_internal_out_.gpu_data(),
            n_joints);
    }

    // ---- 18. FK with velocity conversion (body_q, body_qd) ----
    eval_fk_with_velocity_conversion_kernel<<<div_up(n_art, kBlockSize), kBlockSize, 0, stream>>>(
        model.articulation_start.gpu_data(),
        state_out.joint_q.gpu_data(),
        joint_qd_internal_out_.gpu_data(),
        model.joint_q_start.gpu_data(),
        model.joint_qd_start.gpu_data(),
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_X_p.gpu_data(),
        model.joint_X_c.gpu_data(),
        model.joint_axis.gpu_data(),
        model.joint_dof_dim.gpu_data(),
        model.body_com.gpu_data(),
        body_q_.gpu_data(),
        body_qd_.gpu_data(),
        n_art);

    // ---- 19. Descendant FREE/DISTANCE body pose correction ----
    if (model.n_descendant_free_distance > 0) {
        int n_desc = model.n_descendant_free_distance;

        correct_free_distance_body_pose_kernel<<<div_up(n_desc, kBlockSize), kBlockSize, 0, stream>>>(
            model.descendant_free_distance_joint_indices.gpu_data(),
            model.joint_child.gpu_data(),
            model.body_com.gpu_data(),
            body_q_prev_.gpu_data(),
            body_qd_.gpu_data(),
            body_q_.gpu_data(),
            dt,
            n_desc);

        reconstruct_joint_q_from_body_pose_kernel<<<div_up(n_desc, kBlockSize), kBlockSize, 0, stream>>>(
            model.descendant_free_distance_joint_indices.gpu_data(),
            model.joint_parent.gpu_data(),
            model.joint_child.gpu_data(),
            model.joint_q_start.gpu_data(),
            model.joint_X_p.gpu_data(),
            model.joint_X_c.gpu_data(),
            body_q_.gpu_data(),
            state_out.joint_q.gpu_data(),
            n_desc);

        eval_fk_velocity_from_joint_kernel<<<div_up(n_desc, kBlockSize), kBlockSize, 0, stream>>>(
            model.articulation_start.gpu_data(),
            model.descendant_free_distance_articulation_ids.gpu_data(),
            model.descendant_free_distance_joint_starts.gpu_data(),
            state_out.joint_q.gpu_data(),
            joint_qd_internal_out_.gpu_data(),
            model.joint_q_start.gpu_data(),
            model.joint_qd_start.gpu_data(),
            model.joint_type.gpu_data(),
            model.joint_parent.gpu_data(),
            model.joint_child.gpu_data(),
            model.joint_X_p.gpu_data(),
            model.joint_X_c.gpu_data(),
            model.joint_axis.gpu_data(),
            model.joint_dof_dim.gpu_data(),
            model.body_com.gpu_data(),
            body_q_.gpu_data(),
            body_qd_.gpu_data(),
            n_desc);
    }

    // ---- 20. Convert qd internal → public ----
    convert_qd_internal_to_public_kernel<<<div_up(n_joints, kBlockSize), kBlockSize, 0, stream>>>(
        model.joint_type.gpu_data(),
        model.joint_parent.gpu_data(),
        model.joint_child.gpu_data(),
        model.joint_qd_start.gpu_data(),
        model.joint_X_p.gpu_data(),
        body_q_.gpu_data(),
        model.body_com.gpu_data(),
        joint_qd_internal_out_.gpu_data(),
        state_out.joint_qd.gpu_data(),
        n_joints);
}

}  // namespace rigid
}  // namespace chysx
