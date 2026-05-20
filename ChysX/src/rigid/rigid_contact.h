// SPDX-License-Identifier: Apache-2.0
//
// SoA contact data for the AVBD solver.

#pragma once

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace rigid {

enum StickFlag : int {
    STICK_NONE     = 0,
    STICK_ANCHOR   = 1,  // kinematic/static pair
    STICK_DEADZONE = 2,  // dynamic-dynamic below threshold
};

struct RigidContactData {
    // --- Geometry from narrowphase ---
    CudaArray<int>            shape0;    // shape index of body A
    CudaArray<int>            shape1;    // shape index of body B
    CudaArray<math::Vec3f>    point0;    // contact point, body-A local frame
    CudaArray<math::Vec3f>    point1;    // contact point, body-B local frame
    CudaArray<math::Vec3f>    normal;    // world-space contact normal (A→B)
    CudaArray<float>          margin0;   // contact gap for shape A
    CudaArray<float>          margin1;   // contact gap for shape B

    // --- AVBD state ---
    CudaArray<float>          penalty_k;
    CudaArray<math::Vec3f>    lambda;
    CudaArray<math::Vec3f>    C0;
    CudaArray<int>            stick_flag;
    CudaArray<float>          material_ke;
    CudaArray<float>          material_kd;
    CudaArray<float>          material_mu;

    // --- Per-body CSR contact list ---
    CudaArray<int>            body_contact_counts;   // [body_count]
    CudaArray<int>            body_contact_indices;  // [body_count * per_body_cap]

    int contact_count = 0;
    int contact_max   = 0;      // allocated capacity
    int per_body_cap  = 0;      // max contacts per body

    // --- Warm-start history (previous frame) ---
    CudaArray<math::Vec3f>    prev_lambda;
    CudaArray<math::Vec3f>    prev_point0;
    CudaArray<math::Vec3f>    prev_point1;
    CudaArray<math::Vec3f>    prev_normal;
    CudaArray<int>            prev_stick_flag;
    CudaArray<float>          prev_penalty_k;
    CudaArray<int>            prev_shape0;
    CudaArray<int>            prev_shape1;
    int prev_count = 0;

    void resize(int max_contacts, int body_count, int per_body) {
        contact_max = max_contacts;
        per_body_cap = per_body;
        contact_count = 0;

        shape0.resize(max_contacts); shape1.resize(max_contacts);
        point0.resize(max_contacts); point1.resize(max_contacts);
        normal.resize(max_contacts);
        margin0.resize(max_contacts); margin1.resize(max_contacts);

        penalty_k.resize(max_contacts);
        lambda.resize(max_contacts); C0.resize(max_contacts);
        stick_flag.resize(max_contacts);
        material_ke.resize(max_contacts);
        material_kd.resize(max_contacts);
        material_mu.resize(max_contacts);

        body_contact_counts.resize(body_count);
        body_contact_indices.resize(body_count * per_body);

        prev_lambda.resize(max_contacts);
        prev_point0.resize(max_contacts); prev_point1.resize(max_contacts);
        prev_normal.resize(max_contacts);
        prev_stick_flag.resize(max_contacts);
        prev_penalty_k.resize(max_contacts);
        prev_shape0.resize(max_contacts); prev_shape1.resize(max_contacts);
    }
};

}  // namespace rigid
}  // namespace chysx
