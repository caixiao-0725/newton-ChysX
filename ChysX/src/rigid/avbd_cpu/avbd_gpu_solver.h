// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU AVBD solver: graph-colored Gauss-Seidel iterations on GPU.
// Body state lives on GPU; primal update runs per-color-group;
// dual update runs all manifolds in parallel.

#pragma once

#include "avbd_narrowphase_gpu.h"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace avbd {

class GraphColoringGPU;

class GpuSolver {
public:
    GpuSolver() = default;
    ~GpuSolver() = default;

    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    /// Upload body state from CPU SoA arrays to GPU.
    /// Must be called once per frame before solve().
    void upload_bodies(
        const float* pos_x, const float* pos_y, const float* pos_z,
        const float* quat_x, const float* quat_y, const float* quat_z, const float* quat_w,
        const float* vel_x, const float* vel_y, const float* vel_z,
        const float* velang_x, const float* velang_y, const float* velang_z,
        const float* prevvel_x, const float* prevvel_y, const float* prevvel_z,
        const float* mass, const float* moment_x, const float* moment_y, const float* moment_z,
        const float* half_x, const float* half_y, const float* half_z,
        const float* friction,
        int n_bodies);

    /// Hybrid upload: copy pose/half/mass/friction D2D from device pointers
    /// (e.g. broadphase GPU buffers), upload velocity/moment H2D from CPU.
    /// Avoids redundant H2D for data already on GPU.
    void upload_bodies_hybrid(
        const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
        const float* quat_x_dev, const float* quat_y_dev,
        const float* quat_z_dev, const float* quat_w_dev,
        const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
        const float* mass_dev, const float* friction_dev,
        const float* vel_x, const float* vel_y, const float* vel_z,
        const float* velang_x, const float* velang_y, const float* velang_z,
        const float* prevvel_x, const float* prevvel_y, const float* prevvel_z,
        const float* moment_x, const float* moment_y, const float* moment_z,
        int n_bodies);

    /// Run the full GPU solver: init bodies, colored GS iterations, velocity update.
    void solve(
        GpuManifold* manifolds_dev, GpuContact* contacts_dev,
        int n_manifolds,
        const int* vtx_counts_dev, const VertexEntry* vtx_table_dev, int vtx_stride,
        const int* colors_dev, int num_colors,
        int iterations, float dt, float gravity,
        float alpha, float beta_lin, float gamma);

    /// Download updated positions back to CPU arrays (for rendering / CPU forces).
    void download_positions(
        float* pos_x, float* pos_y, float* pos_z,
        float* quat_x, float* quat_y, float* quat_z, float* quat_w,
        float* vel_x, float* vel_y, float* vel_z,
        float* velang_x, float* velang_y, float* velang_z,
        int n_bodies);

    // GPU pointers for external use (e.g., broadphase reads position, renderer reads state)
    float* pos_x_dev() { return pos_x_.gpu_data(); }
    float* pos_y_dev() { return pos_y_.gpu_data(); }
    float* pos_z_dev() { return pos_z_.gpu_data(); }
    float* quat_x_dev() { return quat_x_.gpu_data(); }
    float* quat_y_dev() { return quat_y_.gpu_data(); }
    float* quat_z_dev() { return quat_z_.gpu_data(); }
    float* quat_w_dev() { return quat_w_.gpu_data(); }
    float* half_x_dev() { return half_x_.gpu_data(); }
    float* half_y_dev() { return half_y_.gpu_data(); }
    float* half_z_dev() { return half_z_.gpu_data(); }
    float* mass_dev() { return mass_.gpu_data(); }
    float* friction_dev() { return friction_.gpu_data(); }
    int body_count() const { return n_bodies_; }

    /// Download position and quaternion of a single body from the GPU.
    void download_body_pose(int idx,
                            float& px, float& py, float& pz,
                            float& qx, float& qy, float& qz, float& qw);

    /// Set up the ground body at slot n_bodies (the virtual n+1th body).
    /// Must be called after upload_bodies or after solve() leaves state valid.
    /// Ensures capacity includes the extra slot, writes mass=0, pos=(0,0,ground_z),
    /// identity quat, and zero velocity/moment into that slot.
    void setup_ground_body(float ground_z, float ground_friction);

private:
    void ensure_capacity(int n_bodies);

    int n_bodies_ = 0;       // active body count (set each frame)
    int capacity_  = 0;       // allocated buffer capacity

    // Position / orientation (read-write during solver)
    CudaArray<float> pos_x_, pos_y_, pos_z_;
    CudaArray<float> quat_x_, quat_y_, quat_z_, quat_w_;

    // Velocity
    CudaArray<float> vel_x_, vel_y_, vel_z_;
    CudaArray<float> velang_x_, velang_y_, velang_z_;
    CudaArray<float> prevvel_x_, prevvel_y_, prevvel_z_;

    // Body properties (constant within a frame)
    CudaArray<float> mass_, moment_x_, moment_y_, moment_z_;
    CudaArray<float> half_x_, half_y_, half_z_;
    CudaArray<float> friction_;

    // Solver temporaries
    CudaArray<float> initial_x_, initial_y_, initial_z_;
    CudaArray<float> initial_qx_, initial_qy_, initial_qz_, initial_qw_;
    CudaArray<float> inertial_x_, inertial_y_, inertial_z_;
    CudaArray<float> inertial_qx_, inertial_qy_, inertial_qz_, inertial_qw_;
};

}  // namespace avbd
}  // namespace chysx
