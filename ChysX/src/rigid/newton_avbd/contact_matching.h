// SPDX-License-Identifier: Apache-2.0
//
// Contact matching for warm-start across frames.
// Uses CUB radix sort + binary search + atomic_min.

#pragma once

#include "../../math/quat.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

#include <cstdint>

namespace chysx {
namespace rigid {

class ContactMatcher {
public:
    void resize(int max_contacts);

    // Sort current contacts by key and match against previous frame.
    // After this call, warm-start data has been written into the
    // AVBD contact arrays (penalty_k, lambda, C0, stick_flag).
    void match_and_warmstart(
        int cur_count,
        const int* cur_shape0, const int* cur_shape1,
        const math::Vec3f* cur_point0, const math::Vec3f* cur_point1,
        const math::Vec3f* cur_normal,
        const float* cur_margin0, const float* cur_margin1,
        const int* shape_body,
        const math::Vec3f* body_pos, const math::Quatf* body_quat,
        // AVBD state to write warm-started values
        float* penalty_k, math::Vec3f* lambda,
        math::Vec3f* C0, int* stick_flag,
        const float* material_ke,
        // Previous frame data
        int prev_count,
        const int* prev_shape0, const int* prev_shape1,
        const math::Vec3f* prev_point0, const math::Vec3f* prev_point1,
        const math::Vec3f* prev_normal,
        const math::Vec3f* prev_lambda,
        const float* prev_penalty_k,
        const int* prev_stick_flag,
        std::uintptr_t stream = 0);

    // Snapshot current frame for next-frame matching.
    void snapshot(
        int count,
        const int* shape0, const int* shape1,
        const math::Vec3f* point0, const math::Vec3f* point1,
        const math::Vec3f* normal,
        const math::Vec3f* lambda, const float* penalty_k,
        const int* stick_flag,
        // Outputs (previous frame buffers)
        int* prev_shape0, int* prev_shape1,
        math::Vec3f* prev_point0, math::Vec3f* prev_point1,
        math::Vec3f* prev_normal,
        math::Vec3f* prev_lambda, float* prev_penalty_k,
        int* prev_stick_flag,
        std::uintptr_t stream = 0);

private:
    int max_contacts_ = 0;

    // Sort keys & indices
    CudaArray<std::uint64_t> cur_keys_;
    CudaArray<std::uint64_t> cur_keys_sorted_;
    CudaArray<int>           cur_indices_;
    CudaArray<int>           cur_indices_sorted_;

    CudaArray<std::uint64_t> prev_keys_sorted_;

    // CUB sort temporary
    CudaArray<std::uint8_t>  cub_temp_;
    std::size_t              cub_temp_bytes_ = 0;

    // Match results
    CudaArray<int>           match_index_;    // [max_contacts]
};

}  // namespace rigid
}  // namespace chysx
