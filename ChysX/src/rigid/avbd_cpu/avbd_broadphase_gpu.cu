// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU broadphase for AVBD: computes per-body AABBs from OBB data,
// builds a QuantBvh over them, then self-queries for overlapping pairs.

#include "avbd_broadphase_gpu.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace chysx {
namespace avbd {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("avbd::BroadphaseGPU: ") + w +
                                 ": " + cudaGetErrorString(e));
}

__global__ void compute_body_aabbs_kernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy, const float* __restrict__ qz,
    const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    int n,
    collision::Aabb* __restrict__ out_aabb,
    math::Vec3f*     __restrict__ out_center)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float cx = px[i], cy = py[i], cz = pz[i];

    float x = qx[i], y = qy[i], z = qz[i], w = qw[i];
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    float r00 = 1.f - (yy + zz), r01 = xy - wz,        r02 = xz + wy;
    float r10 = xy + wz,         r11 = 1.f - (xx + zz), r12 = yz - wx;
    float r20 = xz - wy,         r21 = yz + wx,         r22 = 1.f - (xx + yy);

    float ex = hx[i], ey = hy[i], ez = hz[i];

    float ax = fabsf(r00) * ex + fabsf(r01) * ey + fabsf(r02) * ez;
    float ay = fabsf(r10) * ex + fabsf(r11) * ey + fabsf(r12) * ez;
    float az = fabsf(r20) * ex + fabsf(r21) * ey + fabsf(r22) * ez;

    collision::Aabb box;
    box.mn = math::Vec3f(cx - ax, cy - ay, cz - az);
    box.mx = math::Vec3f(cx + ax, cy + ay, cz + az);
    out_aabb[i] = box;
    out_center[i] = math::Vec3f(cx, cy, cz);
}

__global__ void split_pairs_kernel(
    const math::Vec2i* __restrict__ pairs_in,
    int count,
    int* __restrict__ pair_a,
    int* __restrict__ pair_b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    pair_a[i] = pairs_in[i].x;
    pair_b[i] = pairs_in[i].y;
}

}  // namespace

void BroadphaseGPU::build(int max_bodies, int max_pairs) {
    max_bodies_ = max_bodies;
    max_pairs_  = max_pairs;

    pos_x_.resize(max_bodies); pos_y_.resize(max_bodies); pos_z_.resize(max_bodies);
    quat_x_.resize(max_bodies); quat_y_.resize(max_bodies);
    quat_z_.resize(max_bodies); quat_w_.resize(max_bodies);
    half_x_.resize(max_bodies); half_y_.resize(max_bodies); half_z_.resize(max_bodies);
    mass_.resize(max_bodies);
    friction_.resize(max_bodies);
    aabbs_.resize(max_bodies);
    centers_.resize(max_bodies);

    pair_a_split_.resize(max_pairs);
    pair_b_split_.resize(max_pairs);

    bvh_.build(max_bodies, max_pairs);
}

int BroadphaseGPU::query(
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* quat_x, const float* quat_y, const float* quat_z,
    const float* quat_w,
    const float* half_x, const float* half_y, const float* half_z,
    const float* mass, const float* friction,
    int n_bodies,
    int* pair_a, int* pair_b)
{
    if (n_bodies <= 1) return 0;
    if (n_bodies > max_bodies_) {
        int64_t n = n_bodies;
        int mp = static_cast<int>(std::min(n * 8, n * (n - 1) / 2));
        if (mp < 256) mp = 256;
        build(n_bodies, mp);
    }

    std::size_t fb = n_bodies * sizeof(float);
    auto upload = [&](CudaArray<float>& arr, const float* src) {
        memcpy(arr.cpu_data(), src, fb);
        arr.copy_to_device();
    };
    upload(pos_x_, pos_x);   upload(pos_y_, pos_y);   upload(pos_z_, pos_z);
    upload(quat_x_, quat_x); upload(quat_y_, quat_y);
    upload(quat_z_, quat_z); upload(quat_w_, quat_w);
    upload(half_x_, half_x); upload(half_y_, half_y); upload(half_z_, half_z);
    upload(mass_, mass);
    upload(friction_, friction);

    compute_body_aabbs_kernel<<<grid(n_bodies), kBlock>>>(
        pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
        quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
        half_x_.gpu_data(), half_y_.gpu_data(), half_z_.gpu_data(),
        n_bodies, aabbs_.gpu_data(), centers_.gpu_data());
    check(cudaGetLastError(), "compute_body_aabbs");

    bvh_.build(n_bodies, max_pairs_);
    bvh_.refit_full(aabbs_.gpu_data(), centers_.gpu_data());
    bvh_.query_self_aabb_full(aabbs_.gpu_data(), mass_.gpu_data());

    int count_host = 0;
    check(cudaMemcpy(&count_host, bvh_.query_count_dev(),
                     sizeof(int), cudaMemcpyDeviceToHost),
          "pair_count D2H");
    int count = std::min(count_host, max_pairs_);

    // Split Vec2i pairs into separate a/b arrays on GPU for narrowphase
    if (count > 0) {
        if (count > (int)pair_a_split_.gpu_size()) {
            pair_a_split_.resize(count);
            pair_b_split_.resize(count);
        }
        split_pairs_kernel<<<grid(count), kBlock>>>(
            reinterpret_cast<const math::Vec2i*>(bvh_.query_pairs_dev()),
            count,
            pair_a_split_.gpu_data(),
            pair_b_split_.gpu_data());
        check(cudaGetLastError(), "split_pairs");

        // Also download to CPU for the caller
        check(cudaMemcpy(pair_a, pair_a_split_.gpu_data(),
                         count * sizeof(int), cudaMemcpyDeviceToHost),
              "pair_a D2H");
        check(cudaMemcpy(pair_b, pair_b_split_.gpu_data(),
                         count * sizeof(int), cudaMemcpyDeviceToHost),
              "pair_b D2H");
    }

    return count;
}

int BroadphaseGPU::query_gpu(
    const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
    const float* quat_x_dev, const float* quat_y_dev,
    const float* quat_z_dev, const float* quat_w_dev,
    const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
    const float* mass_dev,
    int n_bodies)
{
    if (n_bodies <= 1) return 0;
    if (n_bodies > max_bodies_) {
        int64_t n = n_bodies;
        int mp = static_cast<int>(std::min(n * 8, n * (n - 1) / 2));
        if (mp < 256) mp = 256;
        build(n_bodies, mp);
    }

    compute_body_aabbs_kernel<<<grid(n_bodies), kBlock>>>(
        pos_x_dev, pos_y_dev, pos_z_dev,
        quat_x_dev, quat_y_dev, quat_z_dev, quat_w_dev,
        half_x_dev, half_y_dev, half_z_dev,
        n_bodies, aabbs_.gpu_data(), centers_.gpu_data());
    check(cudaGetLastError(), "compute_body_aabbs");

    bvh_.build(n_bodies, max_pairs_);
    bvh_.refit_full(aabbs_.gpu_data(), centers_.gpu_data());
    bvh_.query_self_aabb_full(aabbs_.gpu_data(), mass_dev);

    int count_host = 0;
    check(cudaMemcpy(&count_host, bvh_.query_count_dev(),
                     sizeof(int), cudaMemcpyDeviceToHost),
          "pair_count D2H");
    int count = std::min(count_host, max_pairs_);

    if (count > 0) {
        if (count > (int)pair_a_split_.gpu_size()) {
            pair_a_split_.resize(count);
            pair_b_split_.resize(count);
        }
        split_pairs_kernel<<<grid(count), kBlock>>>(
            reinterpret_cast<const math::Vec2i*>(bvh_.query_pairs_dev()),
            count,
            pair_a_split_.gpu_data(),
            pair_b_split_.gpu_data());
        check(cudaGetLastError(), "split_pairs");
    }

    return count;
}

}  // namespace avbd
}  // namespace chysx
