// SPDX-License-Identifier: Apache-2.0
//
// chysx::rigid::FeatherstoneSolver
//
// 1:1 port of Newton's SolverFeatherstone (Warp → C++/CUDA).
// Solves forward dynamics for articulated rigid bodies using the
// Composite Rigid Body Algorithm (CRBA):
//
//   H = J^T M J           (mass matrix in generalized coordinates)
//   H qdd = tau            (solve via Cholesky)
//   q += qd*dt + qdd*dt^2  (symplectic Euler)

#pragma once

#include <cstdint>
#include <vector>

#include "../../memory/cuda_array.h"
#include "spatial_math.cuh"
#include "joint_types.cuh"

namespace chysx {
namespace rigid {

// ============================================================================
// Model data: constant across timesteps, set once from Python
// ============================================================================
struct ArticulationModel {
    int body_count = 0;
    int joint_count = 0;
    int articulation_count = 0;
    int joint_coord_count = 0;    // total size of joint_q
    int joint_dof_count = 0;      // total size of joint_qd

    // Per-joint arrays (device)
    CudaArray<int>        joint_type;
    CudaArray<int>        joint_parent;
    CudaArray<int>        joint_child;
    CudaArray<int>        joint_q_start;     // [joint_count + 1]
    CudaArray<int>        joint_qd_start;    // [joint_count + 1]
    CudaArray<int>        joint_ancestor;    // [joint_count]
    CudaArray<Vec3f>      joint_axis;        // [joint_dof_count]
    CudaArray<int>        joint_dof_dim;     // [joint_count * 2] flattened (lin, ang)
    CudaArray<Transform7> joint_X_p;         // [joint_count]
    CudaArray<Transform7> joint_X_c;         // [joint_count]

    // Per-body arrays (device)
    CudaArray<Vec3f>  body_com;         // [body_count]
    CudaArray<Mat3f>  body_inertia;     // [body_count]
    CudaArray<float>  body_mass;        // [body_count]
    CudaArray<int>    body_flags;       // [body_count]
    CudaArray<int>    body_world;       // [body_count]

    // Per-DOF arrays (device)
    CudaArray<float>  joint_target_ke;
    CudaArray<float>  joint_target_kd;
    CudaArray<float>  joint_limit_lower;
    CudaArray<float>  joint_limit_upper;
    CudaArray<float>  joint_limit_ke;
    CudaArray<float>  joint_limit_kd;
    CudaArray<float>  joint_armature;     // [joint_dof_count]

    // Articulation arrays
    CudaArray<int>    articulation_start; // [articulation_count + 1]

    // Gravity per world
    CudaArray<Vec3f>  gravity;            // [num_worlds]

    // FREE/DISTANCE descendant info (optional)
    CudaArray<int>    descendant_free_distance_joint_indices;   // joints with FREE/DISTANCE that have parent >= 0
    CudaArray<int>    descendant_free_distance_articulation_ids;
    CudaArray<int>    descendant_free_distance_joint_starts;
    int               n_descendant_free_distance = 0;
};

// ============================================================================
// State: varies per timestep
// ============================================================================
struct ArticulationState {
    CudaArray<float>  joint_q;     // [joint_coord_count]
    CudaArray<float>  joint_qd;    // [joint_dof_count]
};

// ============================================================================
// FeatherstoneSolver
// ============================================================================
class FeatherstoneSolver {
public:
    FeatherstoneSolver() = default;

    // Initialize from host arrays. Copies data to GPU.
    void set_model(const ArticulationModel& model);

    // Control inputs for current step (device pointers)
    struct ControlInputs {
        float* joint_target_pos = nullptr;  // [dof_count]
        float* joint_target_vel = nullptr;  // [dof_count]
        float* joint_f = nullptr;           // [dof_count]
    };

    // Run one Featherstone step.
    // Reads state_in, writes state_out (may alias state_in).
    // body_f_ext: [body_count] external forces (6 floats each), in COM frame.
    //   The solver will convert to origin frame internally.
    void step(
        const ArticulationState& state_in,
        ArticulationState& state_out,
        const ControlInputs& control,
        float* body_f_ext,       // [body_count * 6], will be modified in-place
        float dt,
        std::uintptr_t cuda_stream = 0);

    // After step(), body_q and body_qd are written here:
    float* body_q_ptr() { return reinterpret_cast<float*>(body_q_.gpu_data()); }
    float* body_qd_ptr() { return reinterpret_cast<float*>(body_qd_.gpu_data()); }
    int body_count() const { return model_ ? model_->body_count : 0; }
    int joint_count() const { return model_ ? model_->joint_count : 0; }

    // Direct access to internal arrays for coupled solver use
    SpatialVector* body_v_s_ptr() { return body_v_s_.gpu_data(); }

private:
    const ArticulationModel* model_ = nullptr;

    // Precomputed model-level data
    CudaArray<SpatialMatrix> body_I_m_;      // [body_count] spatial inertia at COM
    CudaArray<Transform7>    body_X_com_;    // [body_count]

    // Articulation index arrays
    CudaArray<int> articulation_J_start_;
    CudaArray<int> articulation_M_start_;
    CudaArray<int> articulation_H_start_;
    CudaArray<int> articulation_M_rows_;
    CudaArray<int> articulation_H_rows_;
    CudaArray<int> articulation_J_rows_;
    CudaArray<int> articulation_J_cols_;
    CudaArray<int> articulation_dof_start_;
    CudaArray<int> articulation_coord_start_;

    // System matrices
    CudaArray<float> J_;
    CudaArray<float> M_;
    CudaArray<float> P_;
    CudaArray<float> H_;
    CudaArray<float> L_;

    // Per-step temporaries
    CudaArray<Transform7>     body_q_;          // [body_count]
    CudaArray<Transform7>     body_q_com_;      // [body_count]
    CudaArray<SpatialVector>  body_qd_;         // [body_count] (v_com, omega)
    CudaArray<SpatialVector>  body_v_s_;        // [body_count]
    CudaArray<SpatialVector>  body_a_s_;        // [body_count]
    CudaArray<SpatialVector>  body_f_s_;        // [body_count]
    CudaArray<SpatialVector>  body_ft_s_;       // [body_count]
    CudaArray<SpatialMatrix>  body_I_s_;        // [body_count]
    CudaArray<SpatialVector>  joint_S_s_;       // [dof_count]

    CudaArray<float> joint_tau_;                // [dof_count]
    CudaArray<float> joint_qdd_;                // [dof_count]
    CudaArray<float> joint_qd_internal_in_;     // [dof_count]
    CudaArray<float> joint_qd_internal_out_;    // [dof_count]
    CudaArray<float> joint_f_internal_;         // [dof_count]

    CudaArray<Transform7> body_q_prev_;         // [body_count] for descendant correction

    int J_size_ = 0;
    int M_size_ = 0;
    int H_size_ = 0;

    bool has_kinematic_bodies_ = false;
    bool has_kinematic_joints_ = false;

    void allocate_buffers();
    void compute_articulation_indices();
};

}  // namespace rigid
}  // namespace chysx
