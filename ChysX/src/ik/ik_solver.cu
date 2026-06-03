// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// ChysX IK Solver — implementation.
// Full port of Newton's IKSolver + IKOptimizerLM + IKOptimizerLBFGS.

#include "ik_solver.h"
#include "ik_kernels.cuh"
#include "ik_objectives.cuh"
#include "ik_lm_solver.cuh"
#include "ik_lbfgs_solver.cuh"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace chysx {
namespace ik {

static int div_up(int n, int d) { return (n + d - 1) / d; }

// ============================================================================
// Model & config setup
// ============================================================================

void IKSolver::set_model(const ArticulationModel& model) {
    model_ = &model;
}

void IKSolver::set_config(const IKConfig& config) {
    config_ = config;
}

void IKSolver::add_objective(const IKObjectiveDesc& desc) {
    objectives_.push_back(desc);
}

// ============================================================================
// Finalize: compute residual layout, allocate all buffers
// ============================================================================

void IKSolver::finalize() {
    n_problems_ = config_.n_problems;
    n_seeds_ = config_.n_seeds;
    n_expanded_ = n_problems_ * n_seeds_;
    n_coords_ = model_->joint_coord_count;
    n_dofs_ = model_->joint_dof_count;
    body_count_ = model_->body_count;
    joint_count_ = model_->joint_count;

    // Compute residual layout
    n_residuals_ = 0;
    for (auto& obj : objectives_) {
        obj.residual_offset = n_residuals_;
        if (obj.type == IKObjectiveType::POSITION) {
            obj.residual_dim = 3;
        } else if (obj.type == IKObjectiveType::ROTATION) {
            obj.residual_dim = 3;
        } else if (obj.type == IKObjectiveType::JOINT_LIMIT) {
            obj.residual_dim = n_dofs_;
        }
        n_residuals_ += obj.residual_dim;
    }

    // Main buffers
    joint_q_expanded_.resize(n_expanded_ * n_coords_);
    joint_q_proposed_.resize(n_expanded_ * n_coords_);
    joint_qd_zero_.resize(n_expanded_ * n_dofs_);
    joint_qd_scratch_.resize(n_expanded_ * n_dofs_);
    problem_idx_.resize(n_expanded_);
    best_indices_.resize(n_problems_);

    // Zero joint_qd
    cudaMemset(joint_qd_zero_.gpu_data(), 0, n_expanded_ * n_dofs_ * sizeof(float));

    // Fill problem_idx
    ik_fill_problem_idx_kernel<<<div_up(n_expanded_, kBlockSize), kBlockSize>>>(
        n_expanded_, n_seeds_, problem_idx_.gpu_data());

    // FK workspace
    X_local_.resize(n_expanded_ * joint_count_ * 7);
    body_q_.resize(n_expanded_ * body_count_ * 7);
    joint_S_s_.resize(n_expanded_ * n_dofs_ * 6);

    // Residual / Jacobian
    residuals_.resize(n_expanded_ * n_residuals_);
    residuals_proposed_.resize(n_expanded_ * n_residuals_);
    jacobian_.resize(n_expanded_ * n_residuals_ * n_dofs_);

    // Costs
    costs_.resize(n_expanded_);
    costs_proposed_.resize(n_expanded_);

    // LM workspace
    lambda_values_.resize(n_expanded_);
    dq_dof_.resize(n_expanded_ * n_dofs_);
    pred_reduction_.resize(n_expanded_);
    accept_flags_.resize(n_expanded_);

    // L-BFGS workspace
    if (config_.optimizer == IKOptimizerType::LBFGS) {
        int H = config_.history_len;
        gradient_.resize(n_expanded_ * n_dofs_);
        gradient_prev_.resize(n_expanded_ * n_dofs_);
        search_direction_.resize(n_expanded_ * n_dofs_);
        last_step_dq_.resize(n_expanded_ * n_dofs_);
        s_history_.resize(n_expanded_ * H * n_dofs_);
        y_history_.resize(n_expanded_ * H * n_dofs_);
        rho_history_.resize(n_expanded_ * H);
        alpha_scratch_.resize(n_expanded_ * H);
        history_count_.resize(n_expanded_);
        history_start_.resize(n_expanded_);
        initial_slope_.resize(n_expanded_);
        best_step_idx_.resize(n_expanded_);

        // Line search
        n_line_steps_ = (int)config_.line_search_alphas.size();
        if (n_line_steps_ > 0) {
            line_search_alphas_.resize(n_line_steps_);
            std::vector<float> alphas_host = config_.line_search_alphas;
            line_search_alphas_.resize(n_line_steps_);
            cudaMemcpy(line_search_alphas_.gpu_data(), alphas_host.data(),
                       n_line_steps_ * sizeof(float), cudaMemcpyHostToDevice);

            int n_cand = n_expanded_ * n_line_steps_;
            candidate_q_.resize(n_cand * n_coords_);
            candidate_dq_.resize(n_cand * n_dofs_);
            candidate_q_integrated_.resize(n_cand * n_coords_);
            candidate_qd_scratch_.resize(n_cand * n_dofs_);
            candidate_residuals_.resize(n_cand * n_residuals_);
            candidate_costs_.resize(n_cand);
            candidate_gradients_.resize(n_expanded_ * n_line_steps_ * n_dofs_);
            candidate_slopes_.resize(n_expanded_ * n_line_steps_);
            candidate_jacobian_.resize(n_cand * n_residuals_ * n_dofs_);
        }
    }

    // Per-objective data
    obj_data_.resize(objectives_.size());
    for (size_t i = 0; i < objectives_.size(); ++i) {
        auto& od = obj_data_[i];
        od.desc = objectives_[i];

        if (od.desc.type == IKObjectiveType::POSITION) {
            od.target_positions.resize(n_problems_);
            // Build affects_dof from model topology
            std::vector<int> joint_child_h(joint_count_);
            std::vector<int> joint_parent_h(joint_count_);
            std::vector<int> joint_qd_start_h(joint_count_ + 1);
            std::vector<int> joint_q_start_h(joint_count_ + 1);
            cudaMemcpy(joint_child_h.data(), model_->joint_child.gpu_data(),
                       joint_count_ * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_parent_h.data(), model_->joint_parent.gpu_data(),
                       joint_count_ * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_qd_start_h.data(), model_->joint_qd_start.gpu_data(),
                       (joint_count_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_q_start_h.data(), model_->joint_q_start.gpu_data(),
                       (joint_count_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);

            // dof_to_joint
            std::vector<int> dof_to_joint(n_dofs_, -1);
            for (int j = 0; j < joint_count_; ++j) {
                for (int d = joint_qd_start_h[j]; d < joint_qd_start_h[j + 1]; ++d) {
                    dof_to_joint[d] = j;
                }
            }

            // body_to_joint
            std::vector<int> body_to_joint(body_count_, -1);
            for (int j = 0; j < joint_count_; ++j) {
                int child = joint_child_h[j];
                if (child >= 0 && child < body_count_) body_to_joint[child] = j;
            }

            // Walk ancestor chain
            std::vector<bool> ancestors(joint_count_, false);
            int body = od.desc.link_index;
            while (body >= 0) {
                int j = (body < body_count_) ? body_to_joint[body] : -1;
                if (j >= 0) {
                    ancestors[j] = true;
                    body = joint_parent_h[j];
                } else {
                    body = -1;
                }
            }

            std::vector<unsigned char> affects(n_dofs_, 0);
            for (int d = 0; d < n_dofs_; ++d) {
                if (dof_to_joint[d] >= 0 && ancestors[dof_to_joint[d]])
                    affects[d] = 1;
            }
            od.affects_dof.resize(n_dofs_);
            cudaMemcpy(od.affects_dof.gpu_data(), affects.data(), n_dofs_, cudaMemcpyHostToDevice);

        } else if (od.desc.type == IKObjectiveType::ROTATION) {
            od.target_rotations.resize(n_problems_ * 4);
            // Same affects_dof logic as position
            std::vector<int> joint_child_h(joint_count_);
            std::vector<int> joint_parent_h(joint_count_);
            std::vector<int> joint_qd_start_h(joint_count_ + 1);
            cudaMemcpy(joint_child_h.data(), model_->joint_child.gpu_data(),
                       joint_count_ * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_parent_h.data(), model_->joint_parent.gpu_data(),
                       joint_count_ * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_qd_start_h.data(), model_->joint_qd_start.gpu_data(),
                       (joint_count_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);

            std::vector<int> dof_to_joint(n_dofs_, -1);
            for (int j = 0; j < joint_count_; ++j)
                for (int d = joint_qd_start_h[j]; d < joint_qd_start_h[j + 1]; ++d)
                    dof_to_joint[d] = j;

            std::vector<int> body_to_joint(body_count_, -1);
            for (int j = 0; j < joint_count_; ++j) {
                int child = joint_child_h[j];
                if (child >= 0 && child < body_count_) body_to_joint[child] = j;
            }

            std::vector<bool> ancestors(joint_count_, false);
            int body = od.desc.link_index;
            while (body >= 0) {
                int j = (body < body_count_) ? body_to_joint[body] : -1;
                if (j >= 0) { ancestors[j] = true; body = joint_parent_h[j]; }
                else body = -1;
            }

            std::vector<unsigned char> affects(n_dofs_, 0);
            for (int d = 0; d < n_dofs_; ++d)
                if (dof_to_joint[d] >= 0 && ancestors[dof_to_joint[d]])
                    affects[d] = 1;
            od.affects_dof.resize(n_dofs_);
            cudaMemcpy(od.affects_dof.gpu_data(), affects.data(), n_dofs_, cudaMemcpyHostToDevice);

        } else if (od.desc.type == IKObjectiveType::JOINT_LIMIT) {
            // Build dof_to_coord
            std::vector<int> joint_q_start_h(joint_count_ + 1);
            std::vector<int> joint_qd_start_h(joint_count_ + 1);
            std::vector<int> joint_dof_dim_h(joint_count_ * 2);
            cudaMemcpy(joint_q_start_h.data(), model_->joint_q_start.gpu_data(),
                       (joint_count_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_qd_start_h.data(), model_->joint_qd_start.gpu_data(),
                       (joint_count_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(joint_dof_dim_h.data(), model_->joint_dof_dim.gpu_data(),
                       joint_count_ * 2 * sizeof(int), cudaMemcpyDeviceToHost);

            std::vector<int> d2c(n_dofs_, -1);
            for (int j = 0; j < joint_count_; ++j) {
                int dof0 = joint_qd_start_h[j];
                int coord0 = joint_q_start_h[j];
                int lin = joint_dof_dim_h[j * 2 + 0];
                int ang = joint_dof_dim_h[j * 2 + 1];
                for (int k = 0; k < lin + ang; ++k)
                    d2c[dof0 + k] = coord0 + k;
            }
            od.dof_to_coord.resize(n_dofs_);
            cudaMemcpy(od.dof_to_coord.gpu_data(), d2c.data(), n_dofs_ * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    // Sampling setup
    joint_lower_.resize(n_coords_);
    joint_upper_.resize(n_coords_);
    joint_bounded_.resize(n_coords_);

    finalized_ = true;
}

// ============================================================================
// Target updates
// ============================================================================

void IKSolver::set_target_position(int obj_idx, int problem_idx, float x, float y, float z) {
    Vec3f pos(x, y, z);
    ik_update_target_position_kernel<<<1, 1>>>(
        problem_idx, pos, obj_data_[obj_idx].target_positions.gpu_data());
}

void IKSolver::set_target_rotation(int obj_idx, int problem_idx, float rx, float ry, float rz, float rw) {
    ik_update_target_rotation_kernel<<<1, 1>>>(
        problem_idx, rx, ry, rz, rw, obj_data_[obj_idx].target_rotations.gpu_data());
}

// ============================================================================
// Helper: compute FK (two-pass)
// ============================================================================

void IKSolver::compute_fk_(const float* joint_q, float* body_q_out, int n_batch_actual) {
    int total = n_batch_actual * joint_count_;
    int grid = div_up(total, kBlockSize);

    ik_fk_local_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, joint_count_,
        model_->joint_type.gpu_data(),
        joint_q,
        model_->joint_q_start.gpu_data(),
        model_->joint_qd_start.gpu_data(),
        model_->joint_axis.gpu_data(),
        model_->joint_dof_dim.gpu_data(),
        model_->joint_X_p.gpu_data(),
        model_->joint_X_c.gpu_data(),
        n_coords_,
        X_local_.gpu_data());

    ik_fk_accum_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, joint_count_,
        model_->joint_parent.gpu_data(),
        X_local_.gpu_data(),
        body_q_out);
}

// ============================================================================
// Helper: compute motion subspace
// ============================================================================

void IKSolver::compute_motion_subspace_(const float* body_q, int n_batch_actual) {
    int total = n_batch_actual * joint_count_;
    int grid = div_up(total, kBlockSize);

    cudaMemset(joint_qd_zero_.gpu_data(), 0, n_batch_actual * n_dofs_ * sizeof(float));

    ik_motion_subspace_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, joint_count_, n_dofs_,
        model_->joint_type.gpu_data(),
        model_->joint_parent.gpu_data(),
        model_->joint_qd_start.gpu_data(),
        joint_qd_zero_.gpu_data(),
        model_->joint_axis.gpu_data(),
        model_->joint_dof_dim.gpu_data(),
        body_q,
        model_->joint_X_p.gpu_data(),
        joint_S_s_.gpu_data());
}

// ============================================================================
// Helper: compute residuals for all objectives
// ============================================================================

void IKSolver::compute_residuals_(const float* joint_q, const float* body_q,
                                   float* residuals_out, int n_batch_actual) {
    cudaMemset(residuals_out, 0, n_batch_actual * n_residuals_ * sizeof(float));

    for (size_t i = 0; i < obj_data_.size(); ++i) {
        auto& od = obj_data_[i];
        int start = od.desc.residual_offset;

        if (od.desc.type == IKObjectiveType::POSITION) {
            int grid = div_up(n_batch_actual, kBlockSize);
            ik_pos_residuals_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_residuals_,
                body_q, od.target_positions.gpu_data(),
                problem_idx_.gpu_data(),
                od.desc.link_index, od.desc.link_offset,
                start, od.desc.weight, body_count_,
                residuals_out);

        } else if (od.desc.type == IKObjectiveType::ROTATION) {
            int grid = div_up(n_batch_actual, kBlockSize);
            ik_rot_residuals_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_residuals_,
                body_q, od.target_rotations.gpu_data(),
                problem_idx_.gpu_data(),
                od.desc.link_index, od.desc.link_offset_rotation,
                od.desc.canonicalize_quat_err,
                start, od.desc.weight, body_count_,
                residuals_out);

        } else if (od.desc.type == IKObjectiveType::JOINT_LIMIT) {
            int total = n_batch_actual * n_dofs_;
            int grid = div_up(total, kBlockSize);
            ik_limit_residuals_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_dofs_, n_residuals_, n_coords_,
                joint_q,
                model_->joint_limit_lower.gpu_data(),
                model_->joint_limit_upper.gpu_data(),
                od.dof_to_coord.gpu_data(),
                start, od.desc.weight,
                residuals_out);
        }
    }
}

// ============================================================================
// Helper: compute analytic Jacobian
// ============================================================================

void IKSolver::compute_jacobian_(const float* joint_q, const float* body_q, int n_batch_actual) {
    cudaMemset(jacobian_.gpu_data(), 0, n_batch_actual * n_residuals_ * n_dofs_ * sizeof(float));

    compute_motion_subspace_(body_q, n_batch_actual);

    for (size_t i = 0; i < obj_data_.size(); ++i) {
        auto& od = obj_data_[i];
        int start = od.desc.residual_offset;

        if (od.desc.type == IKObjectiveType::POSITION) {
            int total = n_batch_actual * n_dofs_;
            int grid = div_up(total, kBlockSize);
            ik_pos_jac_analytic_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_dofs_, n_residuals_,
                od.desc.link_index, od.desc.link_offset,
                od.affects_dof.gpu_data(),
                body_q, joint_S_s_.gpu_data(),
                start, od.desc.weight, body_count_,
                jacobian_.gpu_data());

        } else if (od.desc.type == IKObjectiveType::ROTATION) {
            int total = n_batch_actual * n_dofs_;
            int grid = div_up(total, kBlockSize);
            ik_rot_jac_analytic_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_dofs_, n_residuals_,
                od.affects_dof.gpu_data(),
                joint_S_s_.gpu_data(),
                start, od.desc.weight,
                jacobian_.gpu_data());

        } else if (od.desc.type == IKObjectiveType::JOINT_LIMIT) {
            int total = n_batch_actual * n_dofs_;
            int grid = div_up(total, kBlockSize);
            ik_limit_jac_analytic_kernel<<<grid, kBlockSize>>>(
                n_batch_actual, n_dofs_, n_residuals_, n_coords_,
                joint_q,
                model_->joint_limit_lower.gpu_data(),
                model_->joint_limit_upper.gpu_data(),
                od.dof_to_coord.gpu_data(),
                start, od.desc.weight,
                jacobian_.gpu_data());
        }
    }
}

// ============================================================================
// Helper: compute costs
// ============================================================================

void IKSolver::compute_costs_(const float* residuals, float* costs_out, int n_batch_actual) {
    int grid = div_up(n_batch_actual, kBlockSize);
    ik_compute_costs_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, n_residuals_, residuals, costs_out);
}

// ============================================================================
// Helper: integrate dq via jcalc_integrate
// ============================================================================

void IKSolver::integrate_dq_(const float* joint_q_curr, const float* dq, float dt,
                              float* joint_q_out, int n_batch_actual) {
    int total = n_batch_actual * joint_count_;
    int grid = div_up(total, kBlockSize);

    cudaMemset(joint_qd_scratch_.gpu_data(), 0, n_batch_actual * n_dofs_ * sizeof(float));

    ik_integrate_dq_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, joint_count_, n_coords_, n_dofs_,
        model_->joint_type.gpu_data(),
        model_->joint_parent.gpu_data(),
        model_->joint_child.gpu_data(),
        model_->joint_q_start.gpu_data(),
        model_->joint_qd_start.gpu_data(),
        model_->joint_dof_dim.gpu_data(),
        model_->joint_X_c.gpu_data(),
        model_->body_com.gpu_data(),
        joint_q_curr, joint_qd_zero_.gpu_data(), dq, dt,
        joint_q_out, joint_qd_scratch_.gpu_data());
}

// ============================================================================
// Sampling
// ============================================================================

void IKSolver::sample_(const float* joint_q_in) {
    uint32_t seed = config_.rng_seed + rng_counter_++;
    int grid = div_up(n_expanded_, kBlockSize);

    switch (config_.sampler) {
    case IKSamplerType::NONE:
        ik_sample_none_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_coords_, n_seeds_, joint_q_in, joint_q_expanded_.gpu_data());
        break;
    case IKSamplerType::GAUSS:
        ik_sample_gauss_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_coords_, n_seeds_, joint_q_in,
            joint_lower_.gpu_data(), joint_upper_.gpu_data(), joint_bounded_.gpu_data(),
            config_.noise_std, seed, joint_q_expanded_.gpu_data());
        break;
    case IKSamplerType::UNIFORM:
        ik_sample_uniform_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_coords_,
            joint_lower_.gpu_data(), joint_upper_.gpu_data(), joint_bounded_.gpu_data(),
            seed, joint_q_expanded_.gpu_data());
        break;
    case IKSamplerType::ROBERTS:
        ik_sample_roberts_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_coords_, n_seeds_, joint_q_in,
            roberts_basis_.gpu_data(),
            joint_lower_.gpu_data(), joint_upper_.gpu_data(), joint_bounded_.gpu_data(),
            joint_q_expanded_.gpu_data());
        break;
    }
}

// ============================================================================
// LM iteration
// ============================================================================

void IKSolver::lm_step_(float* joint_q, float step_size, int iteration) {
    // FK + residuals (only on first iteration; subsequent reuse accepted state)
    if (iteration == 0) {
        compute_fk_(joint_q, body_q_.gpu_data(), n_expanded_);
        compute_residuals_(joint_q, body_q_.gpu_data(), residuals_.gpu_data(), n_expanded_);
    }

    // Costs
    compute_costs_(residuals_.gpu_data(), costs_.gpu_data(), n_expanded_);

    // Jacobian
    compute_fk_(joint_q, body_q_.gpu_data(), n_expanded_);
    compute_jacobian_(joint_q, body_q_.gpu_data(), n_expanded_);

    // LM solve
    cudaMemset(dq_dof_.gpu_data(), 0, n_expanded_ * n_dofs_ * sizeof(float));
    int grid = div_up(n_expanded_, kBlockSize);
    ik_lm_solve_kernel<<<grid, kBlockSize>>>(
        n_expanded_, n_dofs_, n_residuals_,
        jacobian_.gpu_data(), residuals_.gpu_data(), lambda_values_.gpu_data(),
        dq_dof_.gpu_data(), pred_reduction_.gpu_data());

    // Integrate
    integrate_dq_(joint_q, dq_dof_.gpu_data(), step_size, joint_q_proposed_.gpu_data(), n_expanded_);

    // Proposed FK + residuals + costs
    compute_fk_(joint_q_proposed_.gpu_data(), body_q_.gpu_data(), n_expanded_);
    compute_residuals_(joint_q_proposed_.gpu_data(), body_q_.gpu_data(),
                       residuals_proposed_.gpu_data(), n_expanded_);
    compute_costs_(residuals_proposed_.gpu_data(), costs_proposed_.gpu_data(), n_expanded_);

    // Accept/reject
    ik_accept_reject_kernel<<<grid, kBlockSize>>>(
        n_expanded_,
        costs_.gpu_data(), costs_proposed_.gpu_data(), pred_reduction_.gpu_data(),
        config_.rho_min, accept_flags_.gpu_data());

    // Update state
    ik_update_lm_state_kernel<<<grid, kBlockSize>>>(
        n_expanded_, n_coords_, n_residuals_,
        joint_q_proposed_.gpu_data(), residuals_proposed_.gpu_data(),
        costs_proposed_.gpu_data(), accept_flags_.gpu_data(),
        config_.lambda_factor, config_.lambda_min, config_.lambda_max,
        joint_q, residuals_.gpu_data(), costs_.gpu_data(), lambda_values_.gpu_data());
}

// ============================================================================
// L-BFGS gradient helper
// ============================================================================

void IKSolver::lbfgs_gradient_(const float* joint_q, float* gradient_out, int n_batch_actual) {
    compute_fk_(joint_q, body_q_.gpu_data(), n_batch_actual);
    compute_residuals_(joint_q, body_q_.gpu_data(), residuals_.gpu_data(), n_batch_actual);
    compute_jacobian_(joint_q, body_q_.gpu_data(), n_batch_actual);

    int grid = div_up(n_batch_actual, kBlockSize);
    ik_compute_gradient_kernel<<<grid, kBlockSize>>>(
        n_batch_actual, n_dofs_, n_residuals_,
        jacobian_.gpu_data(), residuals_.gpu_data(), gradient_out);
}

// ============================================================================
// L-BFGS line search
// ============================================================================

void IKSolver::lbfgs_line_search_(float* joint_q) {
    if (n_line_steps_ == 0) return;

    int n_cand = n_expanded_ * n_line_steps_;

    // Generate candidates
    int grid = div_up(n_cand, kBlockSize);
    ik_generate_candidates_kernel<<<grid, kBlockSize>>>(
        n_expanded_, n_line_steps_, n_coords_, n_dofs_,
        joint_q, search_direction_.gpu_data(), line_search_alphas_.gpu_data(),
        candidate_q_.gpu_data(), candidate_dq_.gpu_data());

    // Integrate candidates
    cudaMemset(candidate_qd_scratch_.gpu_data(), 0, n_cand * n_dofs_ * sizeof(float));
    int total = n_cand * joint_count_;
    int grid2 = div_up(total, kBlockSize);
    ik_integrate_dq_kernel<<<grid2, kBlockSize>>>(
        n_cand, joint_count_, n_coords_, n_dofs_,
        model_->joint_type.gpu_data(),
        model_->joint_parent.gpu_data(),
        model_->joint_child.gpu_data(),
        model_->joint_q_start.gpu_data(),
        model_->joint_qd_start.gpu_data(),
        model_->joint_dof_dim.gpu_data(),
        model_->joint_X_c.gpu_data(),
        model_->body_com.gpu_data(),
        candidate_q_.gpu_data(), candidate_qd_scratch_.gpu_data(),
        candidate_dq_.gpu_data(), 1.0f,
        candidate_q_integrated_.gpu_data(), candidate_qd_scratch_.gpu_data());

    // FK + residuals + Jacobian for all candidates
    // Use temporary buffers sized for n_cand
    // We need body_q for candidates: reuse body_q_ buffer if large enough, else allocate
    CudaArray<float> cand_body_q;
    cand_body_q.resize(n_cand * body_count_ * 7);

    int total_fk = n_cand * joint_count_;
    int grid_fk = div_up(total_fk, kBlockSize);
    CudaArray<float> cand_X_local;
    cand_X_local.resize(n_cand * joint_count_ * 7);

    ik_fk_local_kernel<<<grid_fk, kBlockSize>>>(
        n_cand, joint_count_,
        model_->joint_type.gpu_data(),
        candidate_q_integrated_.gpu_data(),
        model_->joint_q_start.gpu_data(),
        model_->joint_qd_start.gpu_data(),
        model_->joint_axis.gpu_data(),
        model_->joint_dof_dim.gpu_data(),
        model_->joint_X_p.gpu_data(),
        model_->joint_X_c.gpu_data(),
        n_coords_,
        cand_X_local.gpu_data());

    ik_fk_accum_kernel<<<grid_fk, kBlockSize>>>(
        n_cand, joint_count_,
        model_->joint_parent.gpu_data(),
        cand_X_local.gpu_data(),
        cand_body_q.gpu_data());

    // Candidate residuals using a temporary problem_idx for candidates
    CudaArray<int> cand_problem_idx;
    cand_problem_idx.resize(n_cand);
    // Each candidate maps to: (cand_row / n_line_steps) -> base expanded row -> problem
    // For simplicity, we construct the mapping on device
    // cand_problem_idx[c] = problem_idx_[c / n_line_steps]
    // We'll use a simple kernel
    auto fill_cand_idx = [&]() {
        // Inline lambda - launch a fill kernel
        struct {
            int n_cand, n_line_steps;
            const int* problem_idx;
            int* cand_idx;
        } args = {n_cand, n_line_steps_, problem_idx_.gpu_data(), cand_problem_idx.gpu_data()};
        // Use a simple memcpy pattern via kernel
        // For now, just set each to floor(c / n_line_steps) / n_seeds
        // Actually we need: problem_idx_map for candidates = problem_idx_[c / n_line_steps]
    };
    // Simple fill via ik_fill_problem_idx with adjusted stride
    // We'll use a direct kernel call instead
    {
        int g = div_up(n_cand, kBlockSize);
        // Manually set: for each c in [0, n_cand), cand_problem_idx[c] = problem_idx_[c / n_line_steps_]
        // We need a tiny kernel for this. Use lambda host code to copy problem_idx to host and replicate.
        std::vector<int> pi_h(n_expanded_);
        cudaMemcpy(pi_h.data(), problem_idx_.gpu_data(), n_expanded_ * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> cpi_h(n_cand);
        for (int c = 0; c < n_cand; ++c) cpi_h[c] = pi_h[c / n_line_steps_];
        cudaMemcpy(cand_problem_idx.gpu_data(), cpi_h.data(), n_cand * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Candidate residuals
    cudaMemset(candidate_residuals_.gpu_data(), 0, n_cand * n_residuals_ * sizeof(float));
    for (size_t i = 0; i < obj_data_.size(); ++i) {
        auto& od = obj_data_[i];
        int start = od.desc.residual_offset;

        if (od.desc.type == IKObjectiveType::POSITION) {
            int g = div_up(n_cand, kBlockSize);
            ik_pos_residuals_kernel<<<g, kBlockSize>>>(
                n_cand, n_residuals_,
                cand_body_q.gpu_data(), od.target_positions.gpu_data(),
                cand_problem_idx.gpu_data(),
                od.desc.link_index, od.desc.link_offset,
                start, od.desc.weight, body_count_,
                candidate_residuals_.gpu_data());
        } else if (od.desc.type == IKObjectiveType::ROTATION) {
            int g = div_up(n_cand, kBlockSize);
            ik_rot_residuals_kernel<<<g, kBlockSize>>>(
                n_cand, n_residuals_,
                cand_body_q.gpu_data(), od.target_rotations.gpu_data(),
                cand_problem_idx.gpu_data(),
                od.desc.link_index, od.desc.link_offset_rotation,
                od.desc.canonicalize_quat_err,
                start, od.desc.weight, body_count_,
                candidate_residuals_.gpu_data());
        } else if (od.desc.type == IKObjectiveType::JOINT_LIMIT) {
            int total_l = n_cand * n_dofs_;
            int g = div_up(total_l, kBlockSize);
            ik_limit_residuals_kernel<<<g, kBlockSize>>>(
                n_cand, n_dofs_, n_residuals_, n_coords_,
                candidate_q_integrated_.gpu_data(),
                model_->joint_limit_lower.gpu_data(),
                model_->joint_limit_upper.gpu_data(),
                od.dof_to_coord.gpu_data(),
                start, od.desc.weight,
                candidate_residuals_.gpu_data());
        }
    }

    // Candidate costs
    {
        int g = div_up(n_cand, kBlockSize);
        ik_compute_costs_kernel<<<g, kBlockSize>>>(
            n_cand, n_residuals_, candidate_residuals_.gpu_data(), candidate_costs_.gpu_data());
    }

    // Candidate Jacobian + gradients for Wolfe curvature
    // Compute motion subspace for candidates
    CudaArray<float> cand_S_s;
    cand_S_s.resize(n_cand * n_dofs_ * 6);
    CudaArray<float> cand_qd_zero;
    cand_qd_zero.resize(n_cand * n_dofs_);
    cudaMemset(cand_qd_zero.gpu_data(), 0, n_cand * n_dofs_ * sizeof(float));

    {
        int total_ms = n_cand * joint_count_;
        int g = div_up(total_ms, kBlockSize);
        ik_motion_subspace_kernel<<<g, kBlockSize>>>(
            n_cand, joint_count_, n_dofs_,
            model_->joint_type.gpu_data(),
            model_->joint_parent.gpu_data(),
            model_->joint_qd_start.gpu_data(),
            cand_qd_zero.gpu_data(),
            model_->joint_axis.gpu_data(),
            model_->joint_dof_dim.gpu_data(),
            cand_body_q.gpu_data(),
            model_->joint_X_p.gpu_data(),
            cand_S_s.gpu_data());
    }

    // Candidate Jacobian
    cudaMemset(candidate_jacobian_.gpu_data(), 0, n_cand * n_residuals_ * n_dofs_ * sizeof(float));
    for (size_t i = 0; i < obj_data_.size(); ++i) {
        auto& od = obj_data_[i];
        int start = od.desc.residual_offset;

        if (od.desc.type == IKObjectiveType::POSITION) {
            int total_j = n_cand * n_dofs_;
            int g = div_up(total_j, kBlockSize);
            ik_pos_jac_analytic_kernel<<<g, kBlockSize>>>(
                n_cand, n_dofs_, n_residuals_,
                od.desc.link_index, od.desc.link_offset,
                od.affects_dof.gpu_data(),
                cand_body_q.gpu_data(), cand_S_s.gpu_data(),
                start, od.desc.weight, body_count_,
                candidate_jacobian_.gpu_data());
        } else if (od.desc.type == IKObjectiveType::ROTATION) {
            int total_j = n_cand * n_dofs_;
            int g = div_up(total_j, kBlockSize);
            ik_rot_jac_analytic_kernel<<<g, kBlockSize>>>(
                n_cand, n_dofs_, n_residuals_,
                od.affects_dof.gpu_data(),
                cand_S_s.gpu_data(),
                start, od.desc.weight,
                candidate_jacobian_.gpu_data());
        } else if (od.desc.type == IKObjectiveType::JOINT_LIMIT) {
            int total_j = n_cand * n_dofs_;
            int g = div_up(total_j, kBlockSize);
            ik_limit_jac_analytic_kernel<<<g, kBlockSize>>>(
                n_cand, n_dofs_, n_residuals_, n_coords_,
                candidate_q_integrated_.gpu_data(),
                model_->joint_limit_lower.gpu_data(),
                model_->joint_limit_upper.gpu_data(),
                od.dof_to_coord.gpu_data(),
                start, od.desc.weight,
                candidate_jacobian_.gpu_data());
        }
    }

    // Candidate gradients = J^T r per candidate
    {
        int g = div_up(n_cand, kBlockSize);
        ik_compute_gradient_kernel<<<g, kBlockSize>>>(
            n_cand, n_dofs_, n_residuals_,
            candidate_jacobian_.gpu_data(), candidate_residuals_.gpu_data(),
            candidate_gradients_.gpu_data());
    }

    // Candidate slopes
    {
        int g = div_up(n_cand, kBlockSize);
        ik_compute_candidate_slopes_kernel<<<g, kBlockSize>>>(
            n_expanded_, n_line_steps_, n_dofs_,
            candidate_gradients_.gpu_data(), search_direction_.gpu_data(),
            candidate_slopes_.gpu_data());
    }

    // Select best step
    {
        int g = div_up(n_expanded_, kBlockSize);
        ik_select_best_step_kernel<<<g, kBlockSize>>>(
            n_expanded_, n_line_steps_, n_dofs_,
            candidate_costs_.gpu_data(),
            candidate_dq_.gpu_data(),
            costs_.gpu_data(), initial_slope_.gpu_data(),
            candidate_slopes_.gpu_data(), line_search_alphas_.gpu_data(),
            config_.wolfe_c1, config_.wolfe_c2,
            best_step_idx_.gpu_data(), last_step_dq_.gpu_data());
    }

    // Apply best step
    {
        int g = div_up(n_expanded_, kBlockSize);
        ik_apply_best_step_kernel<<<g, kBlockSize>>>(
            n_expanded_, n_line_steps_, n_coords_,
            candidate_q_integrated_.gpu_data(),
            best_step_idx_.gpu_data(), joint_q);
    }
}

// ============================================================================
// L-BFGS iteration
// ============================================================================

void IKSolver::lbfgs_step_(float* joint_q, int iteration) {
    compute_costs_(residuals_.gpu_data(), costs_.gpu_data(), n_expanded_);

    lbfgs_gradient_(joint_q, gradient_.gpu_data(), n_expanded_);

    if (iteration == 0) {
        cudaMemcpy(gradient_prev_.gpu_data(), gradient_.gpu_data(),
                   n_expanded_ * n_dofs_ * sizeof(float), cudaMemcpyDeviceToDevice);

        // Initial step: -1e-2 * gradient
        int total = n_expanded_ * n_dofs_;
        int grid = div_up(total, kBlockSize);
        ik_scale_negate_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_dofs_, 1e-2f,
            gradient_.gpu_data(), last_step_dq_.gpu_data());

        integrate_dq_(joint_q, last_step_dq_.gpu_data(), 1.0f,
                      joint_q_proposed_.gpu_data(), n_expanded_);
        cudaMemcpy(joint_q, joint_q_proposed_.gpu_data(),
                   n_expanded_ * n_coords_ * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }

    // Update history
    {
        int grid = div_up(n_expanded_, kBlockSize);
        ik_lbfgs_update_history_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_dofs_, config_.history_len,
            last_step_dq_.gpu_data(), gradient_.gpu_data(), gradient_prev_.gpu_data(),
            s_history_.gpu_data(), y_history_.gpu_data(), rho_history_.gpu_data(),
            history_count_.gpu_data(), history_start_.gpu_data());
    }

    // Search direction
    {
        int grid = div_up(n_expanded_, kBlockSize);
        ik_lbfgs_search_direction_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_dofs_, config_.history_len,
            gradient_.gpu_data(),
            s_history_.gpu_data(), y_history_.gpu_data(), rho_history_.gpu_data(),
            history_count_.gpu_data(), history_start_.gpu_data(),
            config_.h0_scale, alpha_scratch_.gpu_data(),
            search_direction_.gpu_data());
    }

    // Initial slope
    {
        int grid = div_up(n_expanded_, kBlockSize);
        ik_compute_slope_kernel<<<grid, kBlockSize>>>(
            n_expanded_, n_dofs_,
            gradient_.gpu_data(), search_direction_.gpu_data(),
            initial_slope_.gpu_data());
    }

    // Save gradient_prev
    cudaMemcpy(gradient_prev_.gpu_data(), gradient_.gpu_data(),
               n_expanded_ * n_dofs_ * sizeof(float), cudaMemcpyDeviceToDevice);

    // Line search
    lbfgs_line_search_(joint_q);
}

// ============================================================================
// Main step
// ============================================================================

void IKSolver::step(const float* joint_q_in, float* joint_q_out,
                     int iterations, float step_size,
                     std::uintptr_t /*cuda_stream*/) {
    // Sample
    sample_(joint_q_in);

    float* work_q = joint_q_expanded_.gpu_data();

    if (config_.optimizer == IKOptimizerType::LM) {
        // Initialize lambda
        std::vector<float> lam_h(n_expanded_, config_.lambda_initial);
        cudaMemcpy(lambda_values_.gpu_data(), lam_h.data(),
                   n_expanded_ * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < iterations; ++i) {
            lm_step_(work_q, step_size, i);
        }
    } else {
        // L-BFGS
        cudaMemset(history_count_.gpu_data(), 0, n_expanded_ * sizeof(int));
        cudaMemset(history_start_.gpu_data(), 0, n_expanded_ * sizeof(int));

        // Initial FK + residuals
        compute_fk_(work_q, body_q_.gpu_data(), n_expanded_);
        compute_residuals_(work_q, body_q_.gpu_data(), residuals_.gpu_data(), n_expanded_);

        for (int i = 0; i < iterations; ++i) {
            lbfgs_step_(work_q, i);
        }
    }

    // Compute final costs
    compute_fk_(work_q, body_q_.gpu_data(), n_expanded_);
    compute_residuals_(work_q, body_q_.gpu_data(), residuals_.gpu_data(), n_expanded_);
    compute_costs_(residuals_.gpu_data(), costs_.gpu_data(), n_expanded_);

    // Select best seeds and copy to output
    if (n_seeds_ > 1) {
        int grid = div_up(n_problems_, kBlockSize);
        ik_select_best_seed_kernel<<<grid, kBlockSize>>>(
            n_problems_, n_seeds_, costs_.gpu_data(), best_indices_.gpu_data());

        int total = n_problems_ * n_coords_;
        grid = div_up(total, kBlockSize);
        ik_gather_best_seed_kernel<<<grid, kBlockSize>>>(
            n_problems_, n_seeds_, n_coords_,
            work_q, best_indices_.gpu_data(),
            const_cast<float*>(joint_q_out));
    } else {
        cudaMemcpy(const_cast<float*>(joint_q_out), work_q,
                   n_problems_ * n_coords_ * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

}  // namespace ik
}  // namespace chysx
