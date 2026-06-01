// SPDX-License-Identifier: Apache-2.0
//
// SoA collision shape data for the AVBD solver.

#pragma once

#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace rigid {

enum GeoType : int {
    GEO_SPHERE  = 0,
    GEO_BOX     = 1,
    GEO_CAPSULE = 2,
    GEO_PLANE   = 3,
};

struct RigidShapeData {
    CudaArray<int>            body;      // owning body index (-1 = static ground)
    CudaArray<int>            geo_type;  // GeoType enum
    CudaArray<math::Vec3f>    geo_scale; // sphere: (r,0,0), box: half-extents,
                                         // capsule: (r, half_height, 0)
    CudaArray<math::Vec3f>    pos_local; // shape-frame offset from body origin
    CudaArray<math::Quatf>    quat_local;// shape-frame rotation

    CudaArray<float>          ke;        // contact stiffness
    CudaArray<float>          kd;        // contact damping
    CudaArray<float>          mu;        // friction coefficient
    CudaArray<float>          gap;       // contact margin

    int count = 0;

    void resize(int n) {
        count = n;
        body.resize(n); geo_type.resize(n); geo_scale.resize(n);
        pos_local.resize(n); quat_local.resize(n);
        ke.resize(n); kd.resize(n); mu.resize(n); gap.resize(n);
    }
};

}  // namespace rigid
}  // namespace chysx
