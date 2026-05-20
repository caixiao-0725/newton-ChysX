// SPDX-License-Identifier: Apache-2.0
//
// SoA joint data for the AVBD solver (BALL + FIXED types).

#pragma once

#include "../math/matrix.cuh"
#include "../math/quat.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace rigid {

enum JointType : int {
    JOINT_BALL  = 0,
    JOINT_FIXED = 1,
};

struct RigidJointData {
    CudaArray<int>  type;               // JointType
    CudaArray<int>  parent;             // body index (-1 = world)
    CudaArray<int>  child;              // body index

    // Anchor transforms in body frame
    CudaArray<math::Vec3f>  X_p_pos;    // parent-frame anchor position
    CudaArray<math::Quatf>  X_p_quat;   // parent-frame anchor orientation
    CudaArray<math::Vec3f>  X_c_pos;    // child-frame anchor position
    CudaArray<math::Quatf>  X_c_quat;   // child-frame anchor orientation

    // AVBD per-constraint state.
    // BALL = 1 linear slot, FIXED = 1 linear + 1 angular slot.
    // We store max 2 slots per joint (pad with zeros for BALL).
    CudaArray<float>        penalty_k_lin;
    CudaArray<float>        penalty_k_ang;
    CudaArray<math::Vec3f>  lambda_lin;
    CudaArray<math::Vec3f>  lambda_ang;
    CudaArray<math::Vec3f>  C0_lin;
    CudaArray<math::Vec3f>  C0_ang;
    CudaArray<int>          is_hard;    // 1 = hard constraint

    // Per-body adjacency (CSR-style): for each body, which joints touch it
    CudaArray<int>          body_joint_count;    // [body_count]
    CudaArray<int>          body_joint_indices;  // [body_count * max_joints_per_body]
    int max_joints_per_body = 0;

    int count = 0;

    void resize(int n) {
        count = n;
        type.resize(n); parent.resize(n); child.resize(n);
        X_p_pos.resize(n); X_p_quat.resize(n);
        X_c_pos.resize(n); X_c_quat.resize(n);
        penalty_k_lin.resize(n); penalty_k_ang.resize(n);
        lambda_lin.resize(n); lambda_ang.resize(n);
        C0_lin.resize(n); C0_ang.resize(n);
        is_hard.resize(n);
    }

    void resize_adjacency(int body_count, int max_jpb) {
        max_joints_per_body = max_jpb;
        body_joint_count.resize(body_count);
        body_joint_indices.resize(body_count * max_jpb);
    }
};

}  // namespace rigid
}  // namespace chysx
