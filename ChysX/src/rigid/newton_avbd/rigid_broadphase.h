// SPDX-License-Identifier: Apache-2.0
//
// Rigid-body broadphase using shape-level AABBs.
//
// Phase 1 implementation: brute-force O(N²) pairwise AABB overlap test
// in a single CUDA kernel.  Sufficient for scenes with a few hundred
// shapes; a QuantBvh-backed path can be added later.

#pragma once

#include "../../collision/bvh/aabb.cuh"
#include "../../math/quat.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace rigid {

class RigidBroadphase {
public:
    void build(int max_shapes, int max_pairs);

    // Compute shape AABBs from body poses + shape data, then find
    // overlapping pairs.  Writes results to pair_count / pair_list.
    void query(
        const math::Vec3f* body_pos,
        const math::Quatf* body_quat,
        const int*         shape_body,
        const int*         shape_geo_type,
        const math::Vec3f* shape_geo_scale,
        const math::Vec3f* shape_pos_local,
        const math::Quatf* shape_quat_local,
        const float*       shape_gap,
        int                n_shapes,
        std::uintptr_t     cuda_stream = 0);

    int*       pair_count_dev()       { return pair_count_.gpu_data(); }
    const int* pair_count_dev() const { return pair_count_.gpu_data(); }
    math::Vec2i*       pair_list_dev()       { return pair_list_.gpu_data(); }
    const math::Vec2i* pair_list_dev() const { return pair_list_.gpu_data(); }

    int host_pair_count(std::uintptr_t cuda_stream = 0);

private:
    int max_shapes_ = 0;
    int max_pairs_  = 0;

    CudaArray<collision::Aabb> shape_aabbs_;
    CudaArray<int>             pair_count_;
    CudaArray<math::Vec2i>     pair_list_;
};

}  // namespace rigid
}  // namespace chysx
