// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::SdfVolume
//
// Dense 3D signed-distance-field volume with a per-frame rigid-body
// transform.  Stores `nx · ny · nz` float samples on device, plus the
// local-frame voxel grid metadata (voxel size + local-space origin of
// cell (0,0,0)) and a separate world<-local pose `(pos, ex, ey, ez)`
// where `(ex, ey, ez)` are the orthonormal column vectors of the
// rotation matrix.  The pose is the ONLY part that changes per step
// (the SDF samples are baked once at setup), so animating the SDF
// body costs a single 4x4-ish host->device push per frame.
//
// Sampling
// --------
//
//   world  ->  local: local  = R^T · (world - pos)
//   local  ->  grid:  g      = (local - origin_local) / voxel_size
//                   (continuous index in [0, n-1])
//   sd, ∇sd: trilinear interpolation of the eight surrounding voxels;
//            the local gradient is the analytic derivative of the
//            trilinear formula (piecewise-bilinear-per-cell); the
//            world gradient is `R · ∇_local sd`.
//
// Out-of-grid queries (any axis falls outside [0, n-1]) return
// `sd = +∞` sentinel and a zero gradient, so any downstream penalty
// kernel treats them as "no contact" and skips.
//
// Construction
// ------------
//
//   * `bake_box(hx, hy, hz, voxel_size, padding, corner_radius)` -- bakes the
//     analytic box SDF into a grid sized to cover `(±hx±pad,
//     ±hy±pad, ±hz±pad)` with cubic voxels.  Padding defaults to
//     `2 · voxel_size` (just enough to give the penalty band a clean
//     gradient on the surface).  `corner_radius > 0` turns it into a
//     rounded box (Minkowski sum with a sphere), which smooths the
//     normal field near sharp edges / corners.
//
//   * `bake_from_host(values_xyz, nx, ny, nz, voxel_size, origin_local)`
//     -- generic upload from a precomputed host array.  Useful for
//     SDFs cooked offline (Newton's `SDF.create_from_mesh`, etc.).
//
// View
// ----
//
// The `SdfVolumeView` POD bundles every read-only field plus the
// `values` device pointer, so kernels can take it by value (16-byte
// pointer + small floats fit comfortably) and call its
// `__device__ sample(...)` member.  See `sdf_contact.cu` for usage.

#pragma once

#include <cstdint>
#include <vector>

#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace collision {

// Device-friendly POD with a `__device__ sample(...)` member.  Defined
// in `sdf_volume.cuh` so .cu translation units that consume it can
// inline the sampling kernel without going through a function call.
struct SdfVolumeView;

class SdfVolume {
public:
    SdfVolume() = default;

    SdfVolume(const SdfVolume&)            = delete;
    SdfVolume& operator=(const SdfVolume&) = delete;
    SdfVolume(SdfVolume&&) noexcept            = default;
    SdfVolume& operator=(SdfVolume&&) noexcept = default;

    // ---- construction (one-time) ------------------------------------

    // Bake an analytic axis-aligned box SDF centred at the local
    // origin with half-extents `(hx, hy, hz)` into a uniform grid of
    // cubic voxels.  `padding` controls how much extra room the grid
    // wraps around the box (so the SDF is monotonic past the surface);
    // when negative (the default `-1`), defaults to `2 * voxel_size`.
    // `corner_radius` controls optional edge rounding: 0 = sharp box.
    // Must satisfy 0 <= corner_radius < min(hx, hy, hz).
    void bake_box(float hx, float hy, float hz, float voxel_size,
                  float padding = -1.0f,
                  float corner_radius = 0.0f);

    // Generic host->device upload.  `values_xyz` is a flat row-major
    // array of length `nx * ny * nz`, indexed as
    // `values[(k*ny + j)*nx + i]` (x fastest).  Caller owns the
    // memory; this class deep-copies into its own device buffer.
    void bake_from_host(const float*       values_xyz,
                        int                nx,
                        int                ny,
                        int                nz,
                        float              voxel_size,
                        const math::Vec3f& origin_local);

    // ---- per-frame pose update -------------------------------------

    // World <- local: `world = R · local + pos`, where R has column
    // vectors `(ex, ey, ez)` (i.e. `ex` is the local +x axis expressed
    // in world coordinates).  The three vectors must be orthonormal;
    // we do not re-orthogonalise on the device.
    //
    // The pose lives in a small device buffer (4 × Vec3f) so the
    // value the sampling kernel reads stays in sync even when the
    // kernel launch is part of a captured CUDA Graph (graph capture
    // freezes by-value kernel arguments, so we'd otherwise be stuck
    // forever with the pose the volume had at capture time).  Each
    // call uploads 48 bytes async on the optional `cuda_stream`.
    void set_pose(const math::Vec3f& pos,
                  const math::Vec3f& ex,
                  const math::Vec3f& ey,
                  const math::Vec3f& ez,
                  std::uintptr_t cuda_stream = 0) noexcept;

    // Convenience for axis-aligned bodies (identity rotation).
    void set_pose_translation(const math::Vec3f& pos,
                              std::uintptr_t cuda_stream = 0) noexcept {
        set_pose(pos,
                 math::Vec3f(1.0f, 0.0f, 0.0f),
                 math::Vec3f(0.0f, 1.0f, 0.0f),
                 math::Vec3f(0.0f, 0.0f, 1.0f),
                 cuda_stream);
    }

    // ---- accessors --------------------------------------------------

    int   nx() const noexcept { return nx_; }
    int   ny() const noexcept { return ny_; }
    int   nz() const noexcept { return nz_; }
    int   n_voxels() const noexcept { return nx_ * ny_ * nz_; }
    float voxel_size() const noexcept { return voxel_size_; }

    const math::Vec3f& origin_local() const noexcept { return origin_local_; }
    const math::Vec3f& pos() const noexcept { return pos_; }
    const math::Vec3f& ex()  const noexcept { return ex_; }
    const math::Vec3f& ey()  const noexcept { return ey_; }
    const math::Vec3f& ez()  const noexcept { return ez_; }

    const float* values_device() const noexcept {
        return values_.gpu_data();
    }
    float* values_device() noexcept {
        return values_.gpu_data();
    }

    // True if the volume has been baked at least once.
    bool active() const noexcept { return nx_ > 0 && voxel_size_ > 0.0f; }

    // Build a read-only POD view for kernels.  Defined out-of-line so
    // we can include `sdf_volume.cuh` from the consumer .cu without
    // adding a circular dep here.
    SdfVolumeView make_view() const noexcept;

private:
    // Lazily allocate the 4-Vec3f device pose buffer the first time
    // we need it.  Mutable so we can do it from const `make_view()`
    // — but the function semantics stay const-correct (no observable
    // state change for callers, the buffer is just an opaque scratch
    // area used only by `set_pose`).
    void ensure_pose_buffer_() const;

    int                nx_ = 0;
    int                ny_ = 0;
    int                nz_ = 0;
    float              voxel_size_ = 0.0f;
    math::Vec3f        origin_local_;
    // Host-side cache of the latest (pos, ex, ey, ez) so accessors
    // stay cheap and `make_view()` can rebuild without bouncing
    // through device memory.
    math::Vec3f        pos_;
    math::Vec3f        ex_ = math::Vec3f(1.0f, 0.0f, 0.0f);
    math::Vec3f        ey_ = math::Vec3f(0.0f, 1.0f, 0.0f);
    math::Vec3f        ez_ = math::Vec3f(0.0f, 0.0f, 1.0f);
    CudaArray<float>   values_;
    // Device-side mirror of (pos, ex, ey, ez), 4 contiguous Vec3f's,
    // consumed by `SdfVolumeView`'s sample kernel.  We keep this
    // on-device buffer stable across set_pose calls so the device
    // pointer the view holds stays valid for the lifetime of the
    // volume — required for CUDA Graph capture / replay of the
    // SDF detector kernels.
    mutable CudaArray<math::Vec3f> pose_dev_;
};

}  // namespace collision
}  // namespace chysx
