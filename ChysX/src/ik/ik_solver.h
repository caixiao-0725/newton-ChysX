// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// ChysX IK Solver — main class declaration.
// Full port of Newton's IKSolver + IKOptimizerLM + IKOptimizerLBFGS.

#pragma once

#include <vector>
#include <cstdint>

#include "../memory/cuda_array.h"
#include "../rigid/featherstone/featherstone_solver.h"
#include "ik_types.h"

namespace chysx {
namespace ik {

using rigid::ArticulationModel;
using rigid::Transform7;
using rigid::SpatialVector;
using rigid::Vec3f;
using rigid::Quatf;
using rigid::Mat3f;

class IKSolver {
public:
    IKSolver() = default;

    void set_model(const ArticulationModel& model);
    void set_config(const IKConfig& config);
    void add_objective(const IKObjectiveDesc& desc);
    void finalize();

    // Main solve: reads joint_q_in [n_problems * n_coords],
    // writes joint_q_out [n_problems * n_coords].
    void step(const float* joint_q_in, float* joint_q_out,
              int iterations, float step_size,
              std::uintptr_t cuda_stream = 0);

    // Target updates
    void set_target_position(int obj_idx, int problem_idx, float x, float y, float z);
    void set_target_rotation(int obj_idx, int problem_idx, float rx, float ry, float rz, float rw);

    // Pointer accessors
    float* joint_q_ptr() { return joint_q_expanded_.gpu_data(); }
    float* costs_ptr() { return costs_.gpu_data(); }
    int n_expanded() const { return n_expanded_; }
    int n_dofs() const { return n_dofs_; }
    int n_coords() const { return n_coords_; }
    int n_residuals() const { return n_residuals_; }

private:
    const ArticulationModel* model_ = nullptr;
    IKConfig config_;
    std::vector<IKObjectiveDesc> objectives_;
    bool finalized_ = false;

    int n_problems_ = 0;
    int n_seeds_ = 1;
    int n_expanded_ = 0;
    int n_coords_ = 0;
    int n_dofs_ = 0;
    int n_residuals_ = 0;
    int body_count_ = 0;
    int joint_count_ = 0;

    // Expanded batch buffers
    CudaArray<float> joint_q_expanded_;      // [n_expanded, n_coords]
    CudaArray<float> joint_q_proposed_;      // [n_expanded, n_coords]
    CudaArray<float> joint_qd_zero_;         // [n_expanded, n_dofs]
    CudaArray<float> joint_qd_scratch_;      // [n_expanded, n_dofs]
    CudaArray<int>   problem_idx_;           // [n_expanded]
    CudaArray<int>   best_indices_;          // [n_problems]

    // FK workspace
    CudaArray<float> X_local_;               // [n_expanded, joint_count, 7]
    CudaArray<float> body_q_;                // [n_expanded, body_count, 7]
    CudaArray<float> joint_S_s_;             // [n_expanded, n_dofs, 6]

    // Residual / Jacobian workspace
    CudaArray<float> residuals_;             // [n_expanded, n_residuals]
    CudaArray<float> residuals_proposed_;    // [n_expanded, n_residuals]
    CudaArray<float> jacobian_;              // [n_expanded, n_residuals, n_dofs]

    // Cost workspace
    CudaArray<float> costs_;                 // [n_expanded]
    CudaArray<float> costs_proposed_;        // [n_expanded]

    // LM workspace
    CudaArray<float> lambda_values_;         // [n_expanded]
    CudaArray<float> dq_dof_;                // [n_expanded, n_dofs]
    CudaArray<float> pred_reduction_;        // [n_expanded]
    CudaArray<int>   accept_flags_;          // [n_expanded]

    // L-BFGS workspace
    CudaArray<float> gradient_;              // [n_expanded, n_dofs]
    CudaArray<float> gradient_prev_;         // [n_expanded, n_dofs]
    CudaArray<float> search_direction_;      // [n_expanded, n_dofs]
    CudaArray<float> last_step_dq_;          // [n_expanded, n_dofs]
    CudaArray<float> s_history_;             // [n_expanded, history_len, n_dofs]
    CudaArray<float> y_history_;             // [n_expanded, history_len, n_dofs]
    CudaArray<float> rho_history_;           // [n_expanded, history_len]
    CudaArray<float> alpha_scratch_;         // [n_expanded, history_len]
    CudaArray<int>   history_count_;         // [n_expanded]
    CudaArray<int>   history_start_;         // [n_expanded]
    CudaArray<float> initial_slope_;         // [n_expanded]
    CudaArray<int>   best_step_idx_;         // [n_expanded]

    // L-BFGS line search workspace
    int n_line_steps_ = 0;
    CudaArray<float> line_search_alphas_;    // [n_line_steps]
    CudaArray<float> candidate_q_;           // [n_expanded * n_line_steps, n_coords]
    CudaArray<float> candidate_dq_;          // [n_expanded * n_line_steps, n_dofs]
    CudaArray<float> candidate_q_integrated_;// [n_expanded * n_line_steps, n_coords]
    CudaArray<float> candidate_qd_scratch_;  // [n_expanded * n_line_steps, n_dofs]
    CudaArray<float> candidate_residuals_;   // [n_expanded * n_line_steps, n_residuals]
    CudaArray<float> candidate_costs_;       // [n_expanded * n_line_steps]
    CudaArray<float> candidate_gradients_;   // [n_expanded, n_line_steps, n_dofs]
    CudaArray<float> candidate_slopes_;      // [n_expanded, n_line_steps]
    CudaArray<float> candidate_jacobian_;    // [n_expanded * n_line_steps, n_residuals, n_dofs]

    // Per-objective data
    struct ObjectiveData {
        IKObjectiveDesc desc;
        CudaArray<unsigned char> affects_dof;  // [n_dofs], for position/rotation
        CudaArray<int> dof_to_coord;           // [n_dofs], for joint limits
        CudaArray<Vec3f> target_positions;     // [n_problems], for position
        CudaArray<float> target_rotations;     // [n_problems * 4], for rotation
    };
    std::vector<ObjectiveData> obj_data_;

    // Sampling
    CudaArray<float> joint_lower_;           // [n_coords]
    CudaArray<float> joint_upper_;           // [n_coords]
    CudaArray<int>   joint_bounded_;         // [n_coords]
    CudaArray<float> roberts_basis_;         // [n_coords] or empty
    uint32_t rng_counter_ = 0;

    // Helpers
    void sample_(const float* joint_q_in);
    void lm_step_(float* joint_q, float step_size, int iteration);
    void lbfgs_step_(float* joint_q, int iteration);

    void compute_fk_(const float* joint_q, float* body_q_out, int n_batch_actual);
    void compute_motion_subspace_(const float* body_q, int n_batch_actual);
    void compute_residuals_(const float* joint_q, const float* body_q, float* residuals_out, int n_batch_actual);
    void compute_jacobian_(const float* joint_q, const float* body_q, int n_batch_actual);
    void compute_costs_(const float* residuals, float* costs_out, int n_batch_actual);
    void integrate_dq_(const float* joint_q_curr, const float* dq, float dt, float* joint_q_out, int n_batch_actual);

    // L-BFGS line search helpers
    void lbfgs_gradient_(const float* joint_q, float* gradient_out, int n_batch_actual);
    void lbfgs_line_search_(float* joint_q);
};

}  // namespace ik
}  // namespace chysx
