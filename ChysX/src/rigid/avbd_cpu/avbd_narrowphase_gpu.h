// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU-accelerated narrowphase (OBB-OBB SAT) for the AVBD solver.
// Takes broadphase pair list + SoA body data on GPU, outputs contacts.

#pragma once

#include "../../memory/cuda_array.h"
#include "../../math/vec.cuh"

namespace chysx {
namespace avbd {

constexpr int VERTEX_TABLE_MAX_NEIGHBORS = 8;

struct GpuContact {
    int feature_key;
    float rA_x, rA_y, rA_z;
    float rB_x, rB_y, rB_z;

    // Warm-start data (carried from prev frame, zeroed for fresh contacts)
    float lambda_x, lambda_y, lambda_z;
    float penalty_x, penalty_y, penalty_z;
    float C0_x, C0_y, C0_z;
    int stick;  // 0 or 1
};

struct GpuManifold {
    int body_a;
    int body_b;
    int num_contacts;
    int contact_offset;    // index into flat contact array
    float basis[9];        // 3x3 row-major
    float friction;
};

struct VertexEntry {
    int other_body;        // which body this body collides with
    int manifold_idx;      // index into the manifold array
};

class NarrowphaseGPU {
public:
    NarrowphaseGPU() = default;
    ~NarrowphaseGPU() = default;

    NarrowphaseGPU(const NarrowphaseGPU&) = delete;
    NarrowphaseGPU& operator=(const NarrowphaseGPU&) = delete;

    void build(int max_pairs);

    /// Run SAT narrowphase on GPU for all broadphase pairs.
    /// Body SoA data must already reside on the GPU (device pointers).
    /// pair_a_dev / pair_b_dev are device arrays of broadphase pair indices.
    /// Returns the number of manifolds that had contacts (written into manifolds/contacts).
    int query(const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
              const float* quat_x_dev, const float* quat_y_dev,
              const float* quat_z_dev, const float* quat_w_dev,
              const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
              const float* friction_dev,
              const int* pair_a_dev, const int* pair_b_dev,
              int n_pairs, int n_bodies,
              GpuManifold* manifolds_out, GpuContact* contacts_out,
              int& total_contacts_out);

    /// GPU-only variant: no D2H download of manifolds/contacts/vtx.
    /// Returns {n_manifolds, n_contacts} via the output params.
    void query_gpu(const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
                   const float* quat_x_dev, const float* quat_y_dev,
                   const float* quat_z_dev, const float* quat_w_dev,
                   const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
                   const float* friction_dev,
                   const int* pair_a_dev, const int* pair_b_dev,
                   int n_pairs, int n_bodies,
                   int& n_manifolds_out, int& n_contacts_out);

    /// Run warm-start kernel on GPU: match current manifolds/contacts against
    /// previous frame's data using the vertex table, transfer lambda/penalty/C0/stick.
    /// Must be called after query(). Re-downloads contacts to contacts_out.
    void warmstart(int n_manifolds, int n_contacts, int n_bodies,
                   GpuContact* contacts_out);

    /// GPU-only warm-start: no D2H re-download of contacts.
    void warmstart_gpu(int n_manifolds, int n_contacts, int n_bodies);

    /// Append ground-plane contacts to the existing narrowphase results.
    /// Must be called after query_gpu() and before warmstart.
    /// ground_body_idx is the virtual body index for the ground (typically n_bodies).
    /// n_bodies_with_ground = n_bodies + 1 (used for vtx table sizing).
    void append_ground_plane_gpu(
        const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
        const float* quat_x_dev, const float* quat_y_dev,
        const float* quat_z_dev, const float* quat_w_dev,
        const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
        const float* friction_dev, const float* mass_dev,
        int n_bodies, float ground_z, float ground_friction,
        int ground_body_idx,
        int& n_manifolds_inout, int& n_contacts_inout);

    /// Append sphere-collider contacts to existing narrowphase results.
    /// Must be called after query_gpu() (and optionally after append_ground_plane_gpu)
    /// and before warmstart.  Generates box-sphere contacts for all n_box_bodies,
    /// plus an optional sphere-ground contact.
    void append_sphere_gpu(
        const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
        const float* quat_x_dev, const float* quat_y_dev,
        const float* quat_z_dev, const float* quat_w_dev,
        const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
        const float* friction_dev, const float* mass_dev,
        int n_box_bodies, int sphere_body_idx,
        float sphere_radius, float sphere_friction,
        bool has_ground, float ground_z, float ground_friction,
        int ground_body_idx,
        int& n_manifolds_inout, int& n_contacts_inout);

    /// Save current frame's GPU data as "previous frame" for next frame's warm-start.
    /// Call this AFTER the solver loop, with final contact state uploaded.
    void snapshot_for_next_frame(int n_manifolds, int n_contacts, int n_bodies);

    /// Upload post-solver contact data back to GPU for snapshotting.
    void upload_contacts(const GpuContact* contacts, int n_contacts);

    // Per-body neighbor count / table (downloaded after query)
    const int* vertex_counts() const { return vtx_counts_.cpu_data(); }
    const VertexEntry* vertex_table() const { return vtx_table_.cpu_data(); }
    int vertex_table_stride() const { return VERTEX_TABLE_MAX_NEIGHBORS; }

    // GPU device pointers for direct use by graph coloring kernels
    const int* vtx_counts_dev() const { return vtx_counts_.gpu_data(); }
    const VertexEntry* vtx_table_dev() const { return vtx_table_.gpu_data(); }

    // Current frame GPU data (for other kernels that need them)
    GpuManifold* manifolds_dev() { return manifolds_.gpu_data(); }
    GpuContact* contacts_dev() { return contacts_.gpu_data(); }

private:
    int max_pairs_  = 0;
    int max_bodies_ = 0;

    CudaArray<GpuManifold> manifolds_;
    CudaArray<GpuContact>  contacts_;
    CudaArray<int>         manifold_count_;   // single int on GPU
    CudaArray<int>         contact_count_;    // single int on GPU

    // Per-body vertex table: vtx_table_[body * 8 + slot] = {other, manifold_idx}
    CudaArray<int>         vtx_counts_;       // [n_bodies], neighbor count per body
    CudaArray<VertexEntry> vtx_table_;        // [n_bodies * 8]

    // Previous frame data for warm-start (GPU only, no D2H needed)
    CudaArray<GpuManifold> prev_manifolds_;
    CudaArray<GpuContact>  prev_contacts_;
    CudaArray<int>         prev_vtx_counts_;
    CudaArray<VertexEntry> prev_vtx_table_;
    int prev_n_manifolds_ = 0;
    int prev_n_bodies_    = 0;
};

}  // namespace avbd
}  // namespace chysx
