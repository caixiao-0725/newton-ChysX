// SPDX-License-Identifier: Apache-2.0
//
// chysx::solver::VBDSolver
//
// Vertex Block Descent solver for tetrahedral FEM, ported 1:1 from
// Newton's SolverVBD particle pipeline.  VBD is a per-vertex
// Gauss-Seidel (block coordinate descent) solver that updates
// particles color-by-color — vertices sharing an element must have
// different colors so that parallel updates within one color are
// race-free.
//
// Algorithm (per substep):
//
//   1. forward_step — snapshot q_prev, compute inertia target
//   2. for iter in 0..iterations-1:
//        for each color group:
//          solve_elasticity — per-vertex local 3x3 solve
//          update positions from displacements
//   3. update_velocity — v = (q - q_prev) / dt
//
// This replaces PCGSolver for the VBD path.  ClothSimulator selects
// between the two via an enum at construction time.

#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"

namespace chysx {
namespace solver {

// CSR-style per-vertex adjacency to tet elements.
// For vertex i, adjacent tets are stored at indices
// [offsets[i], offsets[i+1]) in the flat `data` array.
// Each entry is a pair: (tet_id, vertex_order_in_tet).
// So actual adj count = (offsets[i+1] - offsets[i]) / 2.
struct TetAdjacency {
    CudaArray<int> offsets;  // [N+1]
    CudaArray<int> data;     // [2 * total_adj_entries]
};

// Per-color group: a device array of particle indices.
struct ColorGroup {
    CudaArray<int> indices;
    int count = 0;
};

class VBDSolver {
public:
    VBDSolver() = default;

    // ---- setup (call once) -------------------------------------------

    // Build graph coloring from tet topology.  Must be called before
    // the first step.  `host_tets` is (T, 4) int32, same as what
    // set_tet_mesh receives.  This runs on the host (CPU).
    void build_coloring(
        const math::Vec4i* host_tets,
        int n_tets,
        int n_particles);

    // Build per-vertex tet adjacency (CSR).  Call after build_coloring
    // or at the same time.
    void build_adjacency(
        const math::Vec4i* host_tets,
        int n_tets,
        int n_particles);

    // ---- per-step interface ------------------------------------------

    // Initialize work buffers for `n` particles.
    void initialize(int n);

    // Run one complete VBD substep.
    //
    // pos / vel: externally-owned device buffers (Newton particle_q / qd)
    // inv_mass: per-particle inverse mass (0 = pinned)
    // tet_indices: device (T, 4) int32
    // tet_poses: device (T,) Mat3f (Dm_inv)
    // tet_materials: device (T,) Vec3f (mu, lambda, k_damp)
    // gravity: (gx, gy, gz)
    // dt: timestep
    // iterations: number of VBD Gauss-Seidel iterations
    // cuda_stream: stream to launch on
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
        std::uintptr_t cuda_stream = 0);

    // Callback invoked before each color's solve_elasticity to allow
    // external contact force accumulation.  Signature:
    //   fn(color_index, q_prev, pos, mass, particle_forces, particle_hessians, stream)
    using ContactCallback = void(*)(
        int color,
        const math::Vec3f* q_prev,
        const math::Vec3f* pos,
        const float* mass,
        math::Vec3f* particle_forces,
        math::Mat3f* particle_hessians,
        cudaStream_t stream,
        void* user_data);

    // Like step() but accepts per-particle external force/hessian
    // accumulators and a callback for per-color contact accumulation.
    void step_with_contacts(
        DeviceSpan<math::Vec3f> pos,
        DeviceSpan<math::Vec3f> vel,
        DeviceSpan<float> inv_mass,
        DeviceSpan<math::Vec4i> tet_indices,
        DeviceSpan<math::Mat3f> tet_poses,
        DeviceSpan<math::Vec3f> tet_materials,
        math::Vec3f gravity,
        float dt,
        int iterations,
        ContactCallback contact_cb,
        void* contact_cb_data,
        math::Vec3f* particle_forces,
        math::Mat3f* particle_hessians,
        std::uintptr_t cuda_stream = 0);

    // ---- accessors ---------------------------------------------------

    int num_colors() const noexcept {
        return static_cast<int>(color_groups_.size());
    }
    int num_particles() const noexcept { return n_particles_; }
    math::Vec3f* q_prev_ptr() noexcept { return q_prev_.gpu_data(); }
    const math::Vec3f* q_prev_ptr() const noexcept { return q_prev_.gpu_data(); }
    const int* particle_colors_ptr() const noexcept { return particle_colors_.gpu_data(); }
    const std::vector<ColorGroup>& color_groups() const noexcept { return color_groups_; }

    // Import an externally-computed coloring instead of building one.
    // `host_colors` is a host array of length `n_particles` mapping each
    // particle to its color index.  Call this *instead of* build_coloring.
    void set_coloring(const int* host_colors, int n_particles);

private:
    int n_particles_ = 0;

    // Graph coloring
    std::vector<ColorGroup> color_groups_;
    CudaArray<int> particle_colors_;  // [N] color id per particle

    // Adjacency
    TetAdjacency tet_adj_;

    // Per-substep work buffers (length = n_particles)
    CudaArray<math::Vec3f> q_prev_;
    CudaArray<math::Vec3f> inertia_;
    CudaArray<math::Vec3f> displacements_;
    CudaArray<float> mass_;  // m = 1/inv_mass
};

}  // namespace solver
}  // namespace chysx
