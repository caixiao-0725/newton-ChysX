// SPDX-License-Identifier: Apache-2.0
//
// Standalone collision detection module for body-particle soft contacts.
//
// Mirrors Newton's ``create_soft_contacts`` kernel: brute-force O(N*M)
// particle x shape SDF evaluation, output stored in SoftContactBuffer.

#pragma once

#include <cstdint>
#include <vector>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"

namespace chysx {
namespace collision {

enum GeoType : int {
    GEO_NONE = 0,
    GEO_PLANE = 1,
    GEO_HFIELD = 2,
    GEO_SPHERE = 3,
    GEO_CAPSULE = 4,
    GEO_ELLIPSOID = 5,
    GEO_CYLINDER = 6,
    GEO_BOX = 7,
    GEO_MESH = 8,
    GEO_CONE = 9,
    GEO_CONVEX_MESH = 10,
};

// Bit flags matching Newton's ParticleFlags / ShapeFlags.
enum ParticleFlag : int { PARTICLE_ACTIVE = 1 << 0 };
enum ShapeFlag : int { SHAPE_COLLIDE_PARTICLES = 1 << 2 };

struct ShapeDataHost {
    std::vector<int>      body;
    std::vector<int>      geo_type;
    std::vector<math::Vec3f> geo_scale;
    std::vector<float>    local_transform;   // 7 floats per shape
    std::vector<int>      flags;
    std::vector<uint64_t> mesh_id;
    // per-shape material (for init_contact_materials)
    std::vector<float>    material_ke;
    std::vector<float>    material_kd;
    std::vector<float>    material_mu;

    int count = 0;
};

struct ShapeDataDevice {
    CudaArray<int>        body;
    CudaArray<int>        geo_type;
    CudaArray<math::Vec3f> geo_scale;
    CudaArray<float>      local_transform;   // 7 * count
    CudaArray<int>        flags;
    CudaArray<uint64_t>   mesh_id;
    CudaArray<float>      material_ke;
    CudaArray<float>      material_kd;
    CudaArray<float>      material_mu;
    int count = 0;
};

struct SoftContactBuffer {
    CudaArray<int>        count;       // [1] atomic counter
    CudaArray<int>        particle;
    CudaArray<int>        shape;
    CudaArray<math::Vec3f> body_pos;   // body-local contact point
    CudaArray<math::Vec3f> body_vel;   // body-local surface velocity
    CudaArray<math::Vec3f> normal;     // world-space normal
    // per-contact material (filled by init_contact_materials)
    CudaArray<float>      ke;
    CudaArray<float>      kd;
    CudaArray<float>      mu;
    int max_contacts = 0;
};

class CollisionPipeline {
public:
    CollisionPipeline() = default;

    void add_shape(int body, int geo_type, math::Vec3f scale,
                   const float* local_tf_7, int flags, uint64_t mesh_id,
                   float mat_ke, float mat_kd, float mat_mu);

    void finalize(int max_soft_contacts = 0);

    void collide(
        const math::Vec3f* particle_q,
        const float*  particle_radius,
        const int*    particle_flags,
        int n_particles,
        const float*  body_q,          // 7 floats per body (Transform7)
        int n_bodies,
        float margin,
        std::uintptr_t cuda_stream);

    void init_contact_materials(
        float soft_contact_ke, float soft_contact_kd, float soft_contact_mu,
        std::uintptr_t cuda_stream);

    // Accessors for contact buffer device pointers
    int*   contact_count_ptr()    { return contacts_.count.gpu_data(); }
    int*   contact_particle_ptr() { return contacts_.particle.gpu_data(); }
    int*   contact_shape_ptr()    { return contacts_.shape.gpu_data(); }
    float* contact_body_pos_ptr() { return reinterpret_cast<float*>(contacts_.body_pos.gpu_data()); }
    float* contact_body_vel_ptr() { return reinterpret_cast<float*>(contacts_.body_vel.gpu_data()); }
    float* contact_normal_ptr()   { return reinterpret_cast<float*>(contacts_.normal.gpu_data()); }
    float* contact_ke_ptr()       { return contacts_.ke.gpu_data(); }
    float* contact_kd_ptr()       { return contacts_.kd.gpu_data(); }
    float* contact_mu_ptr()       { return contacts_.mu.gpu_data(); }
    int    contact_max() const    { return contacts_.max_contacts; }
    int    shape_count() const    { return shapes_gpu_.count; }

    // shape_body device pointer (needed by contact force kernel)
    int*   shape_body_ptr()       { return shapes_gpu_.body.gpu_data(); }

private:
    ShapeDataHost   shapes_host_;
    ShapeDataDevice shapes_gpu_;
    SoftContactBuffer contacts_;
};

}  // namespace collision
}  // namespace chysx
