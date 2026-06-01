// SPDX-License-Identifier: Apache-2.0

#include "rigid_broadphase.h"
#include "rigid_shape.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace chysx {
namespace rigid {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("RigidBroadphase: ") + w +
                                 ": " + cudaGetErrorString(e));
}

// Build AABB for each shape from body pose + shape local transform + geometry.
__global__ void compute_shape_aabbs_kernel(
    const math::Vec3f* __restrict__ body_pos,
    const math::Quatf* __restrict__ body_quat,
    const int*         __restrict__ shape_body,
    const int*         __restrict__ shape_geo_type,
    const math::Vec3f* __restrict__ shape_geo_scale,
    const math::Vec3f* __restrict__ shape_pos_local,
    const math::Quatf* __restrict__ shape_quat_local,
    const float*       __restrict__ shape_gap,
    int n,
    collision::Aabb* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int b = shape_body[i];
    int gt = shape_geo_type[i];
    math::Vec3f sc = shape_geo_scale[i];
    float gap = shape_gap[i];

    // World-space center of shape
    math::Vec3f center;
    math::Quatf rot;
    if (b >= 0) {
        math::Vec3f bp = body_pos[b];
        math::Quatf bq = body_quat[b];
        center = math::transform_point(bp, bq, shape_pos_local[i]);
        rot = math::quat_multiply(bq, shape_quat_local[i]);
    } else {
        center = shape_pos_local[i];
        rot = shape_quat_local[i];
    }

    math::Vec3f half(0.f);
    switch (gt) {
        case GEO_SPHERE:
            half = math::Vec3f(sc.x, sc.x, sc.x);
            break;
        case GEO_BOX: {
            // Compute world-space extents of OBB
            math::Mat3f R = math::quat_to_matrix(rot);
            half.x = fabsf(R(0,0))*sc.x + fabsf(R(0,1))*sc.y + fabsf(R(0,2))*sc.z;
            half.y = fabsf(R(1,0))*sc.x + fabsf(R(1,1))*sc.y + fabsf(R(1,2))*sc.z;
            half.z = fabsf(R(2,0))*sc.x + fabsf(R(2,1))*sc.y + fabsf(R(2,2))*sc.z;
            break;
        }
        case GEO_CAPSULE: {
            // sc.x = radius, sc.y = half_height
            float r = sc.x;
            float hh = sc.y;
            math::Mat3f R = math::quat_to_matrix(rot);
            // Capsule axis is local Y
            math::Vec3f axis(R(0,1)*hh, R(1,1)*hh, R(2,1)*hh);
            half.x = fabsf(axis.x) + r;
            half.y = fabsf(axis.y) + r;
            half.z = fabsf(axis.z) + r;
            break;
        }
        case GEO_PLANE:
            // Infinite plane: use a very large AABB
            half = math::Vec3f(1e6f, 1e6f, 1e6f);
            break;
    }

    half.x += gap;
    half.y += gap;
    half.z += gap;

    collision::Aabb box;
    box.mn = center - half;
    box.mx = center + half;
    out[i] = box;
}

// Brute-force pairwise overlap.  Thread per shape pair (i, j) with i < j.
// Uses a 2D grid: each thread checks one (i, j) pair.
__global__ void overlap_pairs_kernel(
    const collision::Aabb* __restrict__ aabbs,
    const int*             __restrict__ shape_body,
    int n,
    int* __restrict__ pair_count,
    math::Vec2i* __restrict__ pair_list,
    int max_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Map linear index to (i, j) pair with i < j
    // Total pairs = n*(n-1)/2.  idx maps to the upper triangle.
    int total_pairs = n * (n - 1) / 2;
    if (idx >= total_pairs) return;

    // Recover (i, j) from linear index:
    // j = floor((1 + sqrt(1 + 8*idx)) / 2)
    // i = idx - j*(j-1)/2
    float fidx = (float)idx;
    int j = (int)floorf(0.5f + sqrtf(0.25f + 2.0f * fidx));
    int i = idx - j * (j - 1) / 2;
    if (j >= n) j = n - 1;
    while (j * (j - 1) / 2 > idx) --j;
    while (j * (j + 1) / 2 <= idx) ++j;
    i = idx - j * (j - 1) / 2;

    if (i >= j || i < 0 || j >= n) return;

    // Skip shapes on the same body
    int bi = shape_body[i];
    int bj = shape_body[j];
    if (bi >= 0 && bi == bj) return;

    if (aabbs[i].overlaps(aabbs[j])) {
        int slot = atomicAdd(pair_count, 1);
        if (slot < max_pairs) {
            pair_list[slot] = math::Vec2i(i, j);
        }
    }
}

__global__ void clear_int(int* p) { *p = 0; }

}  // namespace

void RigidBroadphase::build(int max_shapes, int max_pairs) {
    max_shapes_ = max_shapes;
    max_pairs_  = max_pairs;
    shape_aabbs_.resize(max_shapes);
    pair_count_.resize(1);
    pair_list_.resize(max_pairs);
}

void RigidBroadphase::query(
    const math::Vec3f* body_pos,
    const math::Quatf* body_quat,
    const int*         shape_body,
    const int*         shape_geo_type,
    const math::Vec3f* shape_geo_scale,
    const math::Vec3f* shape_pos_local,
    const math::Quatf* shape_quat_local,
    const float*       shape_gap,
    int                n_shapes,
    std::uintptr_t     cuda_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cuda_stream);

    // 1. Clear pair counter
    clear_int<<<1, 1, 0, s>>>(pair_count_.gpu_data());
    check(cudaGetLastError(), "clear_int");

    // 2. Compute AABBs
    if (n_shapes > 0) {
        compute_shape_aabbs_kernel<<<grid(n_shapes), kBlock, 0, s>>>(
            body_pos, body_quat, shape_body, shape_geo_type,
            shape_geo_scale, shape_pos_local, shape_quat_local,
            shape_gap, n_shapes, shape_aabbs_.gpu_data());
        check(cudaGetLastError(), "compute_shape_aabbs");
    }

    // 3. Pairwise overlap
    int total_pairs = n_shapes * (n_shapes - 1) / 2;
    if (total_pairs > 0) {
        overlap_pairs_kernel<<<grid(total_pairs), kBlock, 0, s>>>(
            shape_aabbs_.gpu_data(), shape_body, n_shapes,
            pair_count_.gpu_data(), pair_list_.gpu_data(), max_pairs_);
        check(cudaGetLastError(), "overlap_pairs");
    }
}

int RigidBroadphase::host_pair_count(std::uintptr_t cuda_stream) {
    pair_count_.copy_to_host(cuda_stream);
    if (cuda_stream == 0) {
        cudaDeviceSynchronize();
    } else {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
    }
    return pair_count_[0];
}

}  // namespace rigid
}  // namespace chysx
