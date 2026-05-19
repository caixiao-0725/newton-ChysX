// SPDX-License-Identifier: Apache-2.0
//
// Device-side `__device__ inline` sampler for `SdfVolume`.  Kept
// separate from `sdf_volume.h` so the .h can stay free of CUDA
// `__device__` qualifiers (host-only consumers don't need it).

#pragma once

#include "../../math/vec.cuh"
#include "sdf_volume.h"

namespace chysx {
namespace collision {

// Read-only POD bundle the kernels take by value.  Pose lives in a
// SEPARATE device buffer (`pose` points at four contiguous Vec3f's:
// [pos, ex, ey, ez]) so the kernel reads whatever was last uploaded
// by `SdfVolume::set_pose(...)` — even after the kernel launch has
// been captured into a CUDA Graph.  Embedding the pose by value here
// would let `cudaGraphInstantiate` snapshot it at capture time, and
// any later `set_pose` would silently have no effect during replay.
struct SdfVolumeView {
    int                nx;
    int                ny;
    int                nz;
    float              voxel_size;
    math::Vec3f        origin_local;
    const math::Vec3f* pose;    // [pos, ex, ey, ez], 4 Vec3f's on device
    const float*       values;  // length nx*ny*nz, x fastest

    // Trilinear-sample the SDF at a world-space point.  Returns
    //   out_sd          : signed distance at `world_point` (positive
    //                     outside the body, negative inside).
    //   out_world_grad  : ∇_world sd at `world_point`.  Unit-length
    //                     (modulo trilinear discretisation error) on
    //                     the surface, so it can be used directly as
    //                     the contact normal.
    //
    // When the query falls outside the baked grid on any axis the
    // function sets `out_sd = +1e30f` and zeroes `out_world_grad`,
    // so callers can branch on `out_sd >= some_threshold` to skip.
    __device__ inline void sample(const math::Vec3f& world_point,
                                  float&             out_sd,
                                  math::Vec3f&       out_world_grad) const {
        // Read the latest pose from device memory.  pose[0] = body
        // origin in world; pose[1..3] = the orthonormal columns of
        // the world<-local rotation.
        const math::Vec3f pos = pose[0];
        const math::Vec3f ex  = pose[1];
        const math::Vec3f ey  = pose[2];
        const math::Vec3f ez  = pose[3];

        // ---- world -> local ----------------------------------------
        // R has columns (ex, ey, ez); R^T row k is the k-th column of
        // R (= ex/ey/ez).  Therefore
        //   local.x = ex · (world - pos)
        //   local.y = ey · (world - pos)
        //   local.z = ez · (world - pos)
        const math::Vec3f rel = world_point - pos;
        const float lx = ex.x * rel.x + ex.y * rel.y + ex.z * rel.z;
        const float ly = ey.x * rel.x + ey.y * rel.y + ey.z * rel.z;
        const float lz = ez.x * rel.x + ez.y * rel.y + ez.z * rel.z;

        // ---- local -> continuous grid index -----------------------
        const float inv_v = 1.0f / voxel_size;
        const float gx = (lx - origin_local.x) * inv_v;
        const float gy = (ly - origin_local.y) * inv_v;
        const float gz = (lz - origin_local.z) * inv_v;

        const int ix = static_cast<int>(floorf(gx));
        const int iy = static_cast<int>(floorf(gy));
        const int iz = static_cast<int>(floorf(gz));

        // Out-of-grid: signal "no contact" to the caller.
        if (ix < 0 || ix > nx - 2 ||
            iy < 0 || iy > ny - 2 ||
            iz < 0 || iz > nz - 2) {
            out_sd = 1.0e30f;
            out_world_grad = math::Vec3f(0.0f, 0.0f, 0.0f);
            return;
        }

        const float fx = gx - static_cast<float>(ix);
        const float fy = gy - static_cast<float>(iy);
        const float fz = gz - static_cast<float>(iz);

        // ---- gather eight corner samples --------------------------
        const int stride_y = nx;
        const int stride_z = nx * ny;
        const int base = iz * stride_z + iy * stride_y + ix;

        const float v000 = values[base];
        const float v100 = values[base + 1];
        const float v010 = values[base + stride_y];
        const float v110 = values[base + stride_y + 1];
        const float v001 = values[base + stride_z];
        const float v101 = values[base + stride_z + 1];
        const float v011 = values[base + stride_z + stride_y];
        const float v111 = values[base + stride_z + stride_y + 1];

        // ---- trilinear interpolation -----------------------------
        const float v00 = v000 * (1.0f - fx) + v100 * fx;
        const float v10 = v010 * (1.0f - fx) + v110 * fx;
        const float v01 = v001 * (1.0f - fx) + v101 * fx;
        const float v11 = v011 * (1.0f - fx) + v111 * fx;
        const float v0  = v00  * (1.0f - fy) + v10  * fy;
        const float v1  = v01  * (1.0f - fy) + v11  * fy;
        out_sd = v0 * (1.0f - fz) + v1 * fz;

        // ---- analytic ∂sd/∂local_axis ----------------------------
        // For the trilinear formula, ∂f/∂fx is the bilinear interp
        // of the x-edge differences (v100-v000, v110-v010, ...).
        const float gx_local = inv_v * (
            (v100 - v000) * (1.0f - fy) * (1.0f - fz) +
            (v110 - v010) *         fy  * (1.0f - fz) +
            (v101 - v001) * (1.0f - fy) *         fz  +
            (v111 - v011) *         fy  *         fz);
        const float gy_local = inv_v * (
            (v010 - v000) * (1.0f - fx) * (1.0f - fz) +
            (v110 - v100) *         fx  * (1.0f - fz) +
            (v011 - v001) * (1.0f - fx) *         fz  +
            (v111 - v101) *         fx  *         fz);
        const float gz_local = inv_v * (
            (v001 - v000) * (1.0f - fx) * (1.0f - fy) +
            (v101 - v100) *         fx  * (1.0f - fy) +
            (v011 - v010) * (1.0f - fx) *         fy  +
            (v111 - v110) *         fx  *         fy);

        // ---- local -> world gradient -----------------------------
        // ∇_world sd = R · ∇_local sd, with R columns (ex, ey, ez).
        out_world_grad.x = ex.x * gx_local + ey.x * gy_local + ez.x * gz_local;
        out_world_grad.y = ex.y * gx_local + ey.y * gy_local + ez.y * gz_local;
        out_world_grad.z = ex.z * gx_local + ey.z * gy_local + ez.z * gz_local;
    }
};

}  // namespace collision
}  // namespace chysx
