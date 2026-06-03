// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Type definitions for the ChysX IK solver.
// Mirrors Newton's IK enums and configuration structures.

#pragma once

#include <cstdint>
#include <vector>
#include "../math/vec.cuh"
#include "../math/quat.cuh"
#include "../rigid/featherstone/spatial_math.cuh"

namespace chysx {
namespace ik {

using math::Vec3f;
using math::Vec4f;
using math::Quatf;
using rigid::Transform7;

enum class IKOptimizerType { LM, LBFGS };
enum class IKSamplerType { NONE, GAUSS, UNIFORM, ROBERTS };

enum class IKObjectiveType : int {
    POSITION    = 0,
    ROTATION    = 1,
    JOINT_LIMIT = 2,
};

struct IKObjectiveDesc {
    IKObjectiveType type = IKObjectiveType::POSITION;
    int link_index = -1;
    Vec3f link_offset;
    Quatf link_offset_rotation = math::quat_identity();
    float weight = 1.0f;
    bool canonicalize_quat_err = true;
    int residual_dim = 0;
    int residual_offset = 0;
};

struct IKConfig {
    IKOptimizerType optimizer = IKOptimizerType::LM;
    IKSamplerType sampler = IKSamplerType::NONE;
    int n_problems = 1;
    int n_seeds = 1;
    int iterations = 24;
    float step_size = 1.0f;

    // LM parameters
    float lambda_initial = 0.1f;
    float lambda_factor = 2.0f;
    float lambda_min = 1e-5f;
    float lambda_max = 1e10f;
    float rho_min = 1e-3f;

    // L-BFGS parameters
    int history_len = 10;
    float h0_scale = 1.0f;
    float wolfe_c1 = 1e-4f;
    float wolfe_c2 = 0.9f;
    std::vector<float> line_search_alphas = {0.1f, 0.2f, 0.5f, 0.8f, 1.0f, 1.5f, 2.0f, 3.0f};

    // Sampling parameters
    float noise_std = 0.1f;
    uint32_t rng_seed = 12345;

    // Convergence tolerance for early termination (0 = disabled)
    float convergence_tol = 0.0f;
};

}  // namespace ik
}  // namespace chysx
