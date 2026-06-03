// SPDX-License-Identifier: Apache-2.0
//
// Contact matching: CUB radix sort + binary search + atomic_min.

#include "contact_matching.h"
#include "../../math/quat.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <stdexcept>

namespace chysx {
namespace rigid {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }
constexpr std::uint64_t SORT_KEY_SENTINEL = 0xFFFFFFFFFFFFFFFFull;

// Build 64-bit sort key: (shape_a:20 | shape_b:20 | sub_key:24)
// sub_key is unused for now (set to 0); the matching is by position.
__device__ std::uint64_t make_sort_key(int shape_a, int shape_b) {
    std::uint64_t a = static_cast<std::uint64_t>(shape_a) & 0xFFFFFull;
    std::uint64_t b = static_cast<std::uint64_t>(shape_b) & 0xFFFFFull;
    return (a << 44) | (b << 24);
}

__global__ void prepare_sort_keys_kernel(
    int n, int max_contacts,
    const int* __restrict__ shape0, const int* __restrict__ shape1,
    std::uint64_t* __restrict__ keys, int* __restrict__ indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = make_sort_key(shape0[i], shape1[i]);
        indices[i] = i;
    } else if (i < max_contacts) {
        keys[i] = SORT_KEY_SENTINEL;
        indices[i] = i;
    }
}

// Match current contacts against previous sorted contacts.
// For each current contact, find the (shape_a, shape_b) range in prev,
// then pick the closest by world-space midpoint distance.
__global__ void match_contacts_kernel(
    int cur_count, int prev_count,
    const std::uint64_t* __restrict__ cur_keys_sorted,
    const int*           __restrict__ cur_indices_sorted,
    const std::uint64_t* __restrict__ prev_keys_sorted,
    const math::Vec3f*   __restrict__ cur_point0,
    const math::Vec3f*   __restrict__ cur_point1,
    const math::Vec3f*   __restrict__ prev_point0,
    const math::Vec3f*   __restrict__ prev_point1,
    const int*           __restrict__ shape_body,
    const int*           __restrict__ cur_shape0,
    const math::Vec3f*   __restrict__ body_pos,
    const math::Quatf*   __restrict__ body_quat,
    int* __restrict__ match_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cur_count) return;

    int orig_idx = cur_indices_sorted[tid];
    std::uint64_t key = cur_keys_sorted[tid];

    // Mask out sub_key for shape-pair matching
    std::uint64_t pair_key = key & 0xFFFFFFFFFF000000ull;

    // Binary search for range start in prev_keys_sorted
    int lo = 0, hi = prev_count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((prev_keys_sorted[mid] & 0xFFFFFFFFFF000000ull) < pair_key)
            lo = mid + 1;
        else
            hi = mid;
    }
    int range_start = lo;

    // Find range end
    hi = prev_count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((prev_keys_sorted[mid] & 0xFFFFFFFFFF000000ull) <= pair_key)
            lo = mid + 1;
        else
            hi = mid;
    }
    int range_end = lo;

    if (range_start >= range_end) {
        match_index[orig_idx] = -1;
        return;
    }

    // Compute current contact midpoint in world space
    int bi = shape_body[cur_shape0[orig_idx]];
    math::Vec3f p0 = cur_point0[orig_idx];
    math::Vec3f p1 = cur_point1[orig_idx];
    math::Vec3f mid_cur;
    if (bi >= 0) {
        mid_cur = math::transform_point(body_pos[bi], body_quat[bi], p0);
    } else {
        mid_cur = p0;
    }

    // Find closest previous contact in the range
    float best_dist2 = 1e30f;
    int best_prev = -1;
    for (int pi = range_start; pi < range_end; ++pi) {
        math::Vec3f pp0 = prev_point0[pi];
        math::Vec3f pp_world;
        if (bi >= 0) {
            pp_world = math::transform_point(body_pos[bi], body_quat[bi], pp0);
        } else {
            pp_world = pp0;
        }
        math::Vec3f diff = mid_cur - pp_world;
        float d2 = math::dot(diff, diff);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_prev = pi;
        }
    }

    // Accept only if within reasonable distance
    match_index[orig_idx] = (best_dist2 < 0.1f * 0.1f) ? best_prev : -1;
}

// Apply warm-start from matched previous contacts.
__global__ void apply_warmstart_kernel(
    int n,
    const int* __restrict__ match_index,
    const math::Vec3f* __restrict__ prev_lambda,
    const float*       __restrict__ prev_penalty_k,
    const int*         __restrict__ prev_stick_flag,
    const float*       __restrict__ material_ke,
    float*       __restrict__ penalty_k,
    math::Vec3f* __restrict__ lambda,
    math::Vec3f* __restrict__ C0,
    int*         __restrict__ stick_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int mi = match_index[i];
    if (mi >= 0) {
        penalty_k[i] = prev_penalty_k[mi];
        lambda[i] = prev_lambda[mi];
        stick_flag[i] = prev_stick_flag[mi];
    } else {
        penalty_k[i] = material_ke[i];
        lambda[i] = math::Vec3f(0.f);
        stick_flag[i] = 0;
    }
    C0[i] = math::Vec3f(0.f);
}

// Copy current contact state for next-frame matching.
__global__ void snapshot_kernel(
    int n,
    const int*         __restrict__ shape0,  const int*         __restrict__ shape1,
    const math::Vec3f* __restrict__ point0,  const math::Vec3f* __restrict__ point1,
    const math::Vec3f* __restrict__ normal,
    const math::Vec3f* __restrict__ lam,     const float*       __restrict__ pk,
    const int*         __restrict__ sf,
    int*         __restrict__ o_shape0,  int*         __restrict__ o_shape1,
    math::Vec3f* __restrict__ o_point0,  math::Vec3f* __restrict__ o_point1,
    math::Vec3f* __restrict__ o_normal,
    math::Vec3f* __restrict__ o_lam,     float*       __restrict__ o_pk,
    int*         __restrict__ o_sf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o_shape0[i] = shape0[i]; o_shape1[i] = shape1[i];
    o_point0[i] = point0[i]; o_point1[i] = point1[i];
    o_normal[i] = normal[i];
    o_lam[i] = lam[i]; o_pk[i] = pk[i]; o_sf[i] = sf[i];
}

}  // namespace

void ContactMatcher::resize(int max_contacts) {
    max_contacts_ = max_contacts;
    cur_keys_.resize(max_contacts);
    cur_keys_sorted_.resize(max_contacts);
    cur_indices_.resize(max_contacts);
    cur_indices_sorted_.resize(max_contacts);
    prev_keys_sorted_.resize(max_contacts);
    match_index_.resize(max_contacts);

    // Query CUB sort temp size
    cub_temp_bytes_ = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, cub_temp_bytes_,
        cur_keys_.gpu_data(), cur_keys_sorted_.gpu_data(),
        cur_indices_.gpu_data(), cur_indices_sorted_.gpu_data(),
        max_contacts);
    cub_temp_.allocate_device(cub_temp_bytes_);
}

void ContactMatcher::match_and_warmstart(
    int cur_count,
    const int* cur_shape0, const int* cur_shape1,
    const math::Vec3f* cur_point0, const math::Vec3f* cur_point1,
    const math::Vec3f* cur_normal,
    const float* cur_margin0, const float* cur_margin1,
    const int* shape_body,
    const math::Vec3f* body_pos, const math::Quatf* body_quat,
    float* penalty_k, math::Vec3f* lambda,
    math::Vec3f* C0, int* stick_flag,
    const float* material_ke,
    int prev_count,
    const int* prev_shape0, const int* prev_shape1,
    const math::Vec3f* prev_point0, const math::Vec3f* prev_point1,
    const math::Vec3f* prev_normal,
    const math::Vec3f* prev_lambda,
    const float* prev_penalty_k,
    const int* prev_stick_flag,
    std::uintptr_t stream)
{
    auto s = reinterpret_cast<cudaStream_t>(stream);

    if (cur_count <= 0) return;

    // 1. Prepare sort keys for current contacts
    int n_pad = max_contacts_;
    prepare_sort_keys_kernel<<<grid(n_pad), kBlock, 0, s>>>(
        cur_count, n_pad, cur_shape0, cur_shape1,
        cur_keys_.gpu_data(), cur_indices_.gpu_data());

    // 2. CUB radix sort
    std::size_t temp = cub_temp_bytes_;
    cub::DeviceRadixSort::SortPairs(
        cub_temp_.gpu_data(), temp,
        cur_keys_.gpu_data(), cur_keys_sorted_.gpu_data(),
        cur_indices_.gpu_data(), cur_indices_sorted_.gpu_data(),
        n_pad, 0, 64, s);

    // 3. If no previous data, just init
    if (prev_count <= 0) {
        apply_warmstart_kernel<<<grid(cur_count), kBlock, 0, s>>>(
            cur_count, match_index_.gpu_data(),
            prev_lambda, prev_penalty_k, prev_stick_flag, material_ke,
            penalty_k, lambda, C0, stick_flag);
        return;
    }

    // 4. Match against previous sorted keys
    // (previous keys were prepared & sorted during last frame's snapshot)
    match_contacts_kernel<<<grid(cur_count), kBlock, 0, s>>>(
        cur_count, prev_count,
        cur_keys_sorted_.gpu_data(), cur_indices_sorted_.gpu_data(),
        prev_keys_sorted_.gpu_data(),
        cur_point0, cur_point1,
        prev_point0, prev_point1,
        shape_body, cur_shape0,
        body_pos, body_quat,
        match_index_.gpu_data());

    // 5. Apply warm-start
    apply_warmstart_kernel<<<grid(cur_count), kBlock, 0, s>>>(
        cur_count, match_index_.gpu_data(),
        prev_lambda, prev_penalty_k, prev_stick_flag, material_ke,
        penalty_k, lambda, C0, stick_flag);
}

void ContactMatcher::snapshot(
    int count,
    const int* shape0, const int* shape1,
    const math::Vec3f* point0, const math::Vec3f* point1,
    const math::Vec3f* normal,
    const math::Vec3f* lambda, const float* penalty_k,
    const int* stick_flag,
    int* prev_shape0, int* prev_shape1,
    math::Vec3f* prev_point0, math::Vec3f* prev_point1,
    math::Vec3f* prev_normal,
    math::Vec3f* prev_lambda, float* prev_penalty_k,
    int* prev_stick_flag,
    std::uintptr_t stream)
{
    if (count <= 0) return;
    auto s = reinterpret_cast<cudaStream_t>(stream);

    snapshot_kernel<<<grid(count), kBlock, 0, s>>>(
        count, shape0, shape1, point0, point1, normal,
        lambda, penalty_k, stick_flag,
        prev_shape0, prev_shape1, prev_point0, prev_point1, prev_normal,
        prev_lambda, prev_penalty_k, prev_stick_flag);

    // Also sort previous keys for next frame's binary search
    int n_pad = max_contacts_;
    prepare_sort_keys_kernel<<<grid(n_pad), kBlock, 0, s>>>(
        count, n_pad, shape0, shape1,
        cur_keys_.gpu_data(), cur_indices_.gpu_data());

    std::size_t temp = cub_temp_bytes_;
    cub::DeviceRadixSort::SortPairs(
        cub_temp_.gpu_data(), temp,
        cur_keys_.gpu_data(), prev_keys_sorted_.gpu_data(),
        cur_indices_.gpu_data(), cur_indices_sorted_.gpu_data(),
        n_pad, 0, 64, s);
}

}  // namespace rigid
}  // namespace chysx
