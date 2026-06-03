// SPDX-License-Identifier: Apache-2.0
//
// chysx::coupled::CoupledSimulator
//
// A standalone VBD-based simulator for coupled rigid-soft scenes.
// Rigid bodies are driven externally (e.g. Featherstone or IK);
// this class only solves particles and accumulates body-particle
// contact forces into the VBD solve.
//
// Architecture:
//
//   Newton (Python)
//     ├─ Featherstone / IK → body_q, body_qd
//     ├─ CollisionPipeline  → contacts
//     └─ CoupledSimulator   → particle_q, particle_qd (VBD)
//
// For each substep:
//   1. forward_step (particles)
//   2. for iter in 0..iterations-1:
//        for each color group:
//          accumulate body-particle contact force → particle_forces/hessians
//          solve_elasticity (local 3x3 per vertex)
//   3. update_velocity (particles)

#pragma once

#include <cstdint>
#include <vector>

#include "../collision/collision_pipeline.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "../solver/vbd_solver.h"

namespace chysx {
namespace coupled {

// Per-contact data passed from Newton's CollisionPipeline.
// These are device pointers into Newton's Contacts object.
struct BodyParticleContacts {
    int* contact_particle = nullptr;     // [max_contacts] particle idx
    int* contact_count = nullptr;        // [1] actual count
    int  contact_max = 0;

    float* contact_ke = nullptr;         // [max_contacts] penalty stiffness
    float* contact_kd = nullptr;         // [max_contacts] damping coeff
    float* contact_mu = nullptr;         // [max_contacts] friction coeff

    int*   contact_shape = nullptr;      // [max_contacts] shape index
    float* contact_body_pos = nullptr;   // [max_contacts * 3] body-local point
    float* contact_body_vel = nullptr;   // [max_contacts * 3] body-local vel
    float* contact_normal = nullptr;     // [max_contacts * 3] world normal
};

// External rigid body state (from Featherstone or similar).
struct ExternalBodies {
    // body_q: [N_bodies] transform as (px,py,pz, qx,qy,qz,qw) float[7]
    float* body_q = nullptr;
    // body_q_prev: [N_bodies] previous step transform
    float* body_q_prev = nullptr;
    // body_qd: [N_bodies] spatial velocity as (vx,vy,vz, wx,wy,wz) float[6]
    float* body_qd = nullptr;
    // body_com: [N_bodies] COM offset in local frame (x,y,z)
    float* body_com = nullptr;
    // shape_body: [N_shapes] body index per shape
    int*   shape_body = nullptr;
    // particle_radius: [N_particles]
    float* particle_radius = nullptr;
    // particle_colors: [N_particles] color assignment
    int*   particle_colors = nullptr;

    int n_bodies = 0;
    int n_shapes = 0;
};

class CoupledSimulator {
public:
    CoupledSimulator() = default;

    // ---- VBD setup (delegated to internal VBDSolver) -----------------

    void build_coloring(
        const math::Vec4i* host_tets, int n_tets, int n_particles);

    void build_adjacency(
        const math::Vec4i* host_tets, int n_tets, int n_particles);

    void set_coloring(const int* host_colors, int n_particles);

    // ---- per-step interface ------------------------------------------

    void step(
        DeviceSpan<math::Vec3f> pos,
        DeviceSpan<math::Vec3f> vel,
        DeviceSpan<float> inv_mass,
        DeviceSpan<math::Vec4i> tet_indices,
        DeviceSpan<math::Mat3f> tet_poses,
        DeviceSpan<math::Vec3f> tet_materials,
        math::Vec3f gravity,
        float dt,
        int iterations,
        const BodyParticleContacts& contacts,
        const ExternalBodies& bodies,
        float friction_epsilon,
        std::uintptr_t cuda_stream = 0);

    // ---- collision pipeline (ChysX-native) ----------------------------

    void add_collision_shape(int body, int geo_type,
                             float sx, float sy, float sz,
                             const float* local_tf_7, int flags,
                             uint64_t mesh_id,
                             float mat_ke, float mat_kd, float mat_mu);

    void finalize_collision(int max_soft_contacts = 0);

    void step_with_collision(
        DeviceSpan<math::Vec3f> pos,
        DeviceSpan<math::Vec3f> vel,
        DeviceSpan<float> inv_mass,
        DeviceSpan<math::Vec4i> tet_indices,
        DeviceSpan<math::Mat3f> tet_poses,
        DeviceSpan<math::Vec3f> tet_materials,
        math::Vec3f gravity,
        float dt,
        int iterations,
        const float* body_q,
        const float* body_q_prev,
        int n_bodies,
        const float* particle_radius,
        const int*   particle_flags,
        float margin,
        float friction_epsilon,
        float soft_contact_ke,
        float soft_contact_kd,
        float soft_contact_mu,
        std::uintptr_t cuda_stream);

    // ---- accessors ---------------------------------------------------

    int num_colors() const noexcept { return vbd_.num_colors(); }
    solver::VBDSolver& vbd() noexcept { return vbd_; }
    collision::CollisionPipeline& collision_pipeline() noexcept { return collision_; }

private:
    solver::VBDSolver vbd_;
    collision::CollisionPipeline collision_;

    // Per-particle force/hessian accumulators for contacts
    CudaArray<math::Vec3f> particle_forces_;
    CudaArray<math::Mat3f> particle_hessians_;
};

}  // namespace coupled
}  // namespace chysx
