// SPDX-License-Identifier: Apache-2.0
//
// Host-side implementation of `SdfVolume`: grid construction +
// CPU-baked SDF samples + upload to device.  Sampling itself lives
// in `sdf_volume.cuh` so consumer kernels can inline it.

#include "sdf_volume.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "sdf_volume.cuh"  // for SdfVolumeView definition

namespace chysx {
namespace collision {

namespace {

// Analytic SDF of an axis-aligned box centred at the origin with
// half-extents (hx, hy, hz).  Standard Inigo Quilez form: outside
// distance + (negative) inside distance.
inline float sdf_box_analytic(float x, float y, float z,
                              float hx, float hy, float hz) {
    const float qx = std::fabs(x) - hx;
    const float qy = std::fabs(y) - hy;
    const float qz = std::fabs(z) - hz;
    const float ox = std::max(qx, 0.0f);
    const float oy = std::max(qy, 0.0f);
    const float oz = std::max(qz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside = std::min(std::max(std::max(qx, qy), qz), 0.0f);
    return outside + inside;
}

// Rounded-box SDF (IQ style): equivalent to a sharp box with reduced
// half-extents, then offset by a sphere radius `r`.  For r=0 this
// reduces exactly to `sdf_box_analytic`.
inline float sdf_rounded_box_analytic(float x, float y, float z,
                                      float hx, float hy, float hz,
                                      float r) {
    if (r <= 0.0f) {
        return sdf_box_analytic(x, y, z, hx, hy, hz);
    }
    const float bx = hx - r;
    const float by = hy - r;
    const float bz = hz - r;
    const float qx = std::fabs(x) - bx;
    const float qy = std::fabs(y) - by;
    const float qz = std::fabs(z) - bz;
    const float ox = std::max(qx, 0.0f);
    const float oy = std::max(qy, 0.0f);
    const float oz = std::max(qz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside = std::min(std::max(std::max(qx, qy), qz), 0.0f);
    return outside + inside - r;
}

}  // namespace

void SdfVolume::bake_box(float hx, float hy, float hz,
                         float voxel_size, float padding,
                         float corner_radius) {
    if (!(hx > 0.0f && hy > 0.0f && hz > 0.0f)) {
        throw std::invalid_argument(
            "SdfVolume::bake_box: half-extents must be positive");
    }
    if (!(voxel_size > 0.0f)) {
        throw std::invalid_argument(
            "SdfVolume::bake_box: voxel_size must be positive");
    }
    if (padding < 0.0f) {
        padding = 2.0f * voxel_size;
    }
    if (corner_radius < 0.0f) {
        throw std::invalid_argument(
            "SdfVolume::bake_box: corner_radius must be >= 0");
    }
    const float hmin = std::min(hx, std::min(hy, hz));
    if (!(corner_radius < hmin)) {
        throw std::invalid_argument(
            "SdfVolume::bake_box: corner_radius must be smaller than "
            "min(hx, hy, hz)");
    }

    // Total local-space extent we need: [-(h + pad), +(h + pad)] per axis.
    // Round the cell count up so the grid strictly covers that range,
    // then place the (0,0,0) cell at -(h + pad) on each axis.
    auto cells = [&](float h_axis) {
        const float total = 2.0f * (h_axis + padding);
        // +1 for the closing sample (cell count = sample count - 1).
        return std::max(2, static_cast<int>(std::ceil(total / voxel_size)) + 1);
    };
    nx_ = cells(hx);
    ny_ = cells(hy);
    nz_ = cells(hz);
    voxel_size_ = voxel_size;
    origin_local_ = math::Vec3f(
        -(hx + padding),
        -(hy + padding),
        -(hz + padding));

    const std::size_t total = static_cast<std::size_t>(nx_) *
                              static_cast<std::size_t>(ny_) *
                              static_cast<std::size_t>(nz_);
    std::vector<float> host_vals(total);

    for (int k = 0; k < nz_; ++k) {
        const float z = origin_local_.z + voxel_size_ * static_cast<float>(k);
        for (int j = 0; j < ny_; ++j) {
            const float y = origin_local_.y + voxel_size_ * static_cast<float>(j);
            float* row = host_vals.data() +
                         static_cast<std::size_t>((k * ny_ + j) * nx_);
            for (int i = 0; i < nx_; ++i) {
                const float x = origin_local_.x +
                                voxel_size_ * static_cast<float>(i);
                row[i] = sdf_rounded_box_analytic(
                    x, y, z, hx, hy, hz, corner_radius);
            }
        }
    }

    values_.resize(total);
    std::memcpy(values_.cpu_data(), host_vals.data(), total * sizeof(float));
    values_.copy_to_device();

    // Reset pose to identity-at-origin; caller can then `set_pose(...)`.
    ensure_pose_buffer_();
    set_pose_translation(math::Vec3f(0.0f, 0.0f, 0.0f));
}

void SdfVolume::bake_from_host(const float*       values_xyz,
                               int                nx,
                               int                ny,
                               int                nz,
                               float              voxel_size,
                               const math::Vec3f& origin_local) {
    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::invalid_argument(
            "SdfVolume::bake_from_host: each grid dimension must be >= 2");
    }
    if (!(voxel_size > 0.0f)) {
        throw std::invalid_argument(
            "SdfVolume::bake_from_host: voxel_size must be positive");
    }
    if (values_xyz == nullptr) {
        throw std::invalid_argument(
            "SdfVolume::bake_from_host: values_xyz pointer is null");
    }

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    voxel_size_ = voxel_size;
    origin_local_ = origin_local;

    const std::size_t total = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    values_.resize(total);
    std::memcpy(values_.cpu_data(), values_xyz, total * sizeof(float));
    values_.copy_to_device();

    ensure_pose_buffer_();
    set_pose_translation(math::Vec3f(0.0f, 0.0f, 0.0f));
}

void SdfVolume::ensure_pose_buffer_() const {
    // Four contiguous Vec3f's: [pos, ex, ey, ez].  Allocated once;
    // never resized, so the device pointer the views hold stays
    // stable for the lifetime of the volume.
    if (pose_dev_.gpu_size() != 4u) {
        pose_dev_.resize(4u);
    }
}

void SdfVolume::set_pose(const math::Vec3f& pos,
                         const math::Vec3f& ex,
                         const math::Vec3f& ey,
                         const math::Vec3f& ez,
                         std::uintptr_t cuda_stream) noexcept {
    // Update host-side cache for cheap accessors.
    pos_ = pos;
    ex_  = ex;
    ey_  = ey;
    ez_  = ez;

    // Lazily allocate the device buffer (catches the rare path where
    // set_pose is called before any bake_* has run).
    ensure_pose_buffer_();

    // Stage [pos, ex, ey, ez] into the CudaArray's pinned-host slots
    // then async copy to device.  Pinned host -> device with stream
    // is fully overlap-friendly with whatever the solver is doing.
    pose_dev_[0] = pos_;
    pose_dev_[1] = ex_;
    pose_dev_[2] = ey_;
    pose_dev_[3] = ez_;
    pose_dev_.copy_to_device(cuda_stream);
}

SdfVolumeView SdfVolume::make_view() const noexcept {
    // If a caller asks for a view before any set_pose / bake_* call,
    // the device buffer is still empty; lazy-allocate it here so the
    // pointer the view carries is always a valid device address.
    // (The sample kernel will then read whatever default the buffer
    // was zero-init'd to — undefined contact normals, but the volume
    // also has no values so the sampler returns sd = +∞ anyway.)
    ensure_pose_buffer_();

    SdfVolumeView v;
    v.nx           = nx_;
    v.ny           = ny_;
    v.nz           = nz_;
    v.voxel_size   = voxel_size_;
    v.origin_local = origin_local_;
    v.pose         = pose_dev_.gpu_data();
    v.values       = values_.gpu_data();
    return v;
}

}  // namespace collision
}  // namespace chysx
