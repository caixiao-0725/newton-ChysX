// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::MeshContact
//
// Penalty contact between cloth particles and an *animated* rigid
// triangle-mesh body.  Functionally equivalent to `SdfContact` but
// replaces the trilinear SDF volume sample with a BVH-accelerated
// closest-point query on the mesh surface.
//
// The internal BVH is built once from the mesh topology and refitted
// each time the body pose changes (the mesh vertices are re-skinned
// from a rest pose + rigid transform on the GPU, then the triangle
// AABBs are recomputed and the LBVH is refitted bottom-up).
//
// Per-step pipeline (identical shape to SdfContact):
//
//   1. set_pose(...)        -- rigid transform the rest mesh to world.
//   2. detect(positions, n) -- BVH closest-point per particle ->
//                              cache (normal, depth) in Vec4f.
//   3. accumulate_gradient  -- penalty + optional IPC friction.
//   4. bake_diag            -- Hessian diagonal.
//   5. apply_coulomb_friction -- Coulomb-cone post-projection.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "bvh/aabb.cuh"
#include "bvh/quant_bvh.h"

namespace chysx {
namespace collision {

class MeshContact {
public:
    MeshContact() = default;

    MeshContact(const MeshContact&)            = delete;
    MeshContact& operator=(const MeshContact&) = delete;
    MeshContact(MeshContact&&) noexcept            = default;
    MeshContact& operator=(MeshContact&&) noexcept = default;

    // ---- mesh data (call once at setup) ---------------------------------

    // Upload a triangle mesh from host.  `vertices` is the rest-pose
    // vertex array (n_vertices Vec3f), `indices` is a flat triangle
    // index array of length `3 * n_triangles`.  The mesh is stored in
    // device memory; subsequent `set_pose` calls rigidly transform it.
    void set_mesh(const math::Vec3f* vertices, int n_vertices,
                  const int* indices, int n_triangles);

    int n_vertices()  const noexcept { return n_vertices_; }
    int n_triangles() const noexcept { return n_triangles_; }

    // ---- per-frame pose update ------------------------------------------

    // Rigidly transform the rest mesh to world space:
    //   world_p = R * local_p + pos
    // where R = [ex | ey | ez] column vectors.
    // Also rebuilds triangle AABBs and refits the BVH.
    void set_pose(const math::Vec3f& pos,
                  const math::Vec3f& ex,
                  const math::Vec3f& ey,
                  const math::Vec3f& ez,
                  std::uintptr_t cuda_stream = 0);

    // Convenience: identity rotation.
    void set_pose_translation(const math::Vec3f& pos,
                              std::uintptr_t cuda_stream = 0) {
        set_pose(pos,
                 math::Vec3f(1.0f, 0.0f, 0.0f),
                 math::Vec3f(0.0f, 1.0f, 0.0f),
                 math::Vec3f(0.0f, 0.0f, 1.0f),
                 cuda_stream);
    }

    // ---- material parameters (mirror SdfContact) ------------------------

    void set_thickness(float t) noexcept { thickness_ = t; }
    float thickness() const noexcept { return thickness_; }

    void set_search_radius(float r) noexcept { search_radius_ = r; }
    float search_radius() const noexcept { return search_radius_; }

    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    void set_friction(float mu) noexcept { friction_ = mu; }
    float friction() const noexcept { return friction_; }

    void set_friction_epsilon(float eps) noexcept { friction_epsilon_ = eps; }
    float friction_epsilon() const noexcept { return friction_epsilon_; }

    void set_contact_kd(float kd) noexcept { contact_kd_ = kd; }
    float contact_kd() const noexcept { return contact_kd_; }

    void set_ipc_friction_enabled(bool v) noexcept { ipc_friction_ = v; }
    bool ipc_friction_enabled() const noexcept { return ipc_friction_; }

    void set_body_velocity(const math::Vec3f& v,
                           std::uintptr_t cuda_stream = 0);
    const math::Vec3f& body_velocity() const noexcept { return body_velocity_; }

    bool active() const noexcept {
        return stiffness_ > 0.0f && n_triangles_ > 0 && bvh_built_;
    }

    // ---- per-step pipeline (same API as SdfContact) ---------------------

    void detect(const math::Vec3f* positions,
                int                n_particles,
                std::uintptr_t     cuda_stream = 0,
                const math::Vec3f* velocities  = nullptr,
                float              dt          = 0.0f);

    void accumulate_gradient(math::Vec3f*    rhs,
                             int             n_particles,
                             std::uintptr_t  cuda_stream = 0,
                             const math::Vec3f* positions = nullptr,
                             const math::Vec3f* prev_positions = nullptr,
                             float           dt = 0.0f) const;

    void bake_diag(math::Mat3f*    A_diag,
                   int             n_particles,
                   float           dt,
                   std::uintptr_t  cuda_stream = 0,
                   const math::Vec3f* positions = nullptr,
                   const math::Vec3f* prev_positions = nullptr) const;

    void apply_coulomb_friction(math::Vec3f*       rhs,
                                int                n_particles,
                                const float*       mass,
                                const math::Mat3f* diag,
                                const math::Vec3f& gravity,
                                float              inv_dt2,
                                std::uintptr_t     cuda_stream = 0) const;

private:
    // Rest-pose mesh (device).
    CudaArray<math::Vec3f> rest_vertices_;
    CudaArray<math::Vec3i> triangles_;
    int n_vertices_  = 0;
    int n_triangles_ = 0;

    // World-pose mesh (device) — written by set_pose.
    CudaArray<math::Vec3f> world_vertices_;

    // Per-triangle AABB + centroid for BVH.
    CudaArray<Aabb>        tri_aabbs_;
    CudaArray<math::Vec3f> tri_centers_;

    // QuantBvh over world-space triangles (stackless, 16-byte nodes).
    QuantBvh bvh_;
    bool bvh_built_ = false;

    // Material.
    float thickness_        = 0.0f;
    float search_radius_    = 0.0f;  // BVH query radius; 0 → use 10*thickness
    float stiffness_        = 0.0f;
    float friction_         = 0.0f;
    float friction_epsilon_  = 0.01f;
    float contact_kd_       = 1.0e-2f;
    bool  ipc_friction_     = false;

    math::Vec3f body_velocity_ = math::Vec3f(0.0f, 0.0f, 0.0f);

    int          cached_n_particles_ = 0;
    float        cached_dt_          = 0.0f;
    const math::Vec3f* cached_velocities_ = nullptr;

    // Per-particle (normal, depth) cache — same layout as SdfContact.
    CudaArray<math::Vec4f> contacts_;
    CudaArray<math::Vec3f> body_velocity_dev_;
};

}  // namespace collision
}  // namespace chysx
