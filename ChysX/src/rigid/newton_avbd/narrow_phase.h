// SPDX-License-Identifier: Apache-2.0
//
// Narrow-phase contact detection for rigid shapes.

#pragma once

#include "../../math/quat.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"
#include "rigid_shape.h"

#include <cstdint>

namespace chysx {
namespace rigid {

// Run narrowphase on broadphase candidate pairs.  Writes detected
// contacts into the contact arrays via atomicAdd on *contact_count.
void narrow_phase_detect(
    const math::Vec2i* pair_list,
    int                pair_count,
    // Shape data (device pointers)
    const int*         shape_body,
    const int*         shape_geo_type,
    const math::Vec3f* shape_geo_scale,
    const math::Vec3f* shape_pos_local,
    const math::Quatf* shape_quat_local,
    const float*       shape_ke,
    const float*       shape_kd,
    const float*       shape_mu,
    const float*       shape_gap,
    // Body poses (device pointers)
    const math::Vec3f* body_pos,
    const math::Quatf* body_quat,
    // Excluded body pairs (sorted by (min,max), device pointer; null = no filter)
    const math::Vec2i* excluded_body_pairs,
    int                excluded_pair_count,
    // Output contact arrays (device pointers)
    int*         contact_shape0,
    int*         contact_shape1,
    math::Vec3f* contact_point0,
    math::Vec3f* contact_point1,
    math::Vec3f* contact_normal,
    float*       contact_margin0,
    float*       contact_margin1,
    float*       contact_material_ke,
    float*       contact_material_kd,
    float*       contact_material_mu,
    int*         contact_count,     // single int on device
    int          contact_max,
    std::uintptr_t cuda_stream = 0);

}  // namespace rigid
}  // namespace chysx
