// SPDX-License-Identifier: Apache-2.0
//
// VBD (Vertex Block Descent) solver — CUDA implementation.
//
// Ported 1:1 from Newton's SolverVBD / particle_vbd_kernels.py.

#include "vbd_solver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"

namespace chysx {
namespace solver {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::solver::VBDSolver: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// -----------------------------------------------------------------------
// Cofactor (same as in tet_fem_constraint.cu)
// -----------------------------------------------------------------------
__device__ __forceinline__ math::Mat3f vbd_cofactor(const math::Mat3f& F) {
    const float F11 = F(0,0), F21 = F(1,0), F31 = F(2,0);
    const float F12 = F(0,1), F22 = F(1,1), F32 = F(2,1);
    const float F13 = F(0,2), F23 = F(1,2), F33 = F(2,2);
    return math::Mat3f(
        F22*F33 - F23*F32, F23*F31 - F21*F33, F21*F32 - F22*F31,
        F13*F32 - F12*F33, F11*F33 - F13*F31, F12*F31 - F11*F32,
        F12*F23 - F13*F22, F13*F21 - F11*F23, F11*F22 - F12*F21);
}

__device__ __forceinline__ math::Mat3f vbd_mat3_from_cols(
    const math::Vec3f& c0, const math::Vec3f& c1, const math::Vec3f& c2) {
    return math::Mat3f(
        c0.x, c1.x, c2.x,
        c0.y, c1.y, c2.y,
        c0.z, c1.z, c2.z);
}

// -----------------------------------------------------------------------
// forward_step kernel — snapshot q_prev, compute inertia target
// -----------------------------------------------------------------------
__global__ void vbd_forward_step_kernel(
    math::Vec3f* __restrict__ pos,
    math::Vec3f* __restrict__ vel,
    const float* __restrict__ inv_mass,
    math::Vec3f* __restrict__ q_prev,
    math::Vec3f* __restrict__ inertia,
    math::Vec3f* __restrict__ displacements,
    float* __restrict__ mass_out,
    math::Vec3f gravity,
    float dt,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    q_prev[i] = pos[i];
    const float w = inv_mass[i];
    mass_out[i] = (w > 1.0e-12f) ? (1.0f / w) : 0.0f;

    if (w == 0.0f) {
        inertia[i] = pos[i];
        displacements[i] = math::Vec3f(0.0f, 0.0f, 0.0f);
        return;
    }

    const math::Vec3f vel_new = vel[i] + gravity * dt;
    inertia[i] = pos[i] + vel_new * dt;
    displacements[i] = vel_new * dt;
    pos[i] = pos[i] + vel_new * dt;
}

// -----------------------------------------------------------------------
// solve_elasticity kernel — per-vertex local 3x3 VBD solve
//
// For each particle in the current color group:
//   f = m/dt^2 * (inertia - pos) + sum(f_tet)
//   H = m/dt^2 * I + sum(H_tet)
//   displacement += H^{-1} * f
//   pos = q_prev + displacement
// -----------------------------------------------------------------------
__global__ void vbd_solve_elasticity_kernel(
    const int* __restrict__ color_ids,
    int n_color,
    float dt,
    math::Vec3f* __restrict__ pos,
    const math::Vec3f* __restrict__ q_prev,
    const float* __restrict__ mass,
    const math::Vec3f* __restrict__ inertia,
    const math::Vec4i* __restrict__ tet_indices,
    const math::Mat3f* __restrict__ tet_poses,
    const math::Vec3f* __restrict__ tet_materials,
    const int* __restrict__ adj_offsets,
    const int* __restrict__ adj_data,
    math::Vec3f* __restrict__ displacements,
    const math::Vec3f* __restrict__ ext_forces,     // optional, may be null
    const math::Mat3f* __restrict__ ext_hessians) {  // optional, may be null

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_color) return;

    const int pi = color_ids[tid];
    const float m = mass[pi];
    if (m == 0.0f) {
        displacements[pi] = math::Vec3f(0.0f, 0.0f, 0.0f);
        return;
    }

    const float inv_dt2 = 1.0f / (dt * dt);

    // Inertia force and Hessian
    math::Vec3f f = (inertia[pi] - pos[pi]) * (m * inv_dt2);
    math::Mat3f h = math::Mat3f::identity() * (m * inv_dt2);

    // Iterate over adjacent tets
    const int adj_begin = adj_offsets[pi];
    const int adj_end = adj_offsets[pi + 1];
    for (int a = adj_begin; a < adj_end; a += 2) {
        const int tet_id = adj_data[a];
        const int v_order = adj_data[a + 1];

        const math::Vec3f& mat = tet_materials[tet_id];
        const float mu = mat.x;
        const float lmbd = mat.y;
        const float k_damp = mat.z;

        if (mu <= 0.0f && lmbd <= 0.0f) continue;

        const math::Vec4i& ti = tet_indices[tet_id];
        const math::Vec3f v0 = pos[ti.x], v1 = pos[ti.y];
        const math::Vec3f v2 = pos[ti.z], v3 = pos[ti.w];
        const math::Mat3f& Bm = tet_poses[tet_id];

        // Rest volume
        const float det_Bm = determinant(Bm);
        const float rest_volume = 1.0f / (det_Bm * 6.0f);

        // Deformation gradient F = Ds * Dm_inv
        const math::Mat3f Ds = vbd_mat3_from_cols(v1 - v0, v2 - v0, v3 - v0);
        const math::Mat3f F = Ds * Bm;
        const float J = determinant(F);

        // Stable Neo-Hookean parameters
        const float mu_nh = mu;
        const float lmbd_nh = lmbd + mu;
        const float lmbd_safe = (lmbd_nh >= 0.0f ? 1.0f : -1.0f)
                                * fmaxf(fabsf(lmbd_nh), 1.0e-6f);
        const float alpha = 1.0f + mu_nh / lmbd_safe;

        // Cofactor
        const math::Mat3f cof = vbd_cofactor(F);

        // Flatten to vec9 (column-major)
        float f_vec[9], cof_vec[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                f_vec[c * 3 + r] = F(r, c);
                cof_vec[c * 3 + r] = cof(r, c);
            }
        }

        // P_vec = V0 * (mu_nh * f + s * cof_vec)
        const float s = lmbd_nh * (J - alpha);
        float P_vec[9];
        for (int k = 0; k < 9; ++k)
            P_vec[k] = rest_volume * (mu_nh * f_vec[k] + s * cof_vec[k]);

        // Material constants for Hessian
        const float mu_V = rest_volume * mu_nh;
        const float lmbd_V = rest_volume * lmbd_nh;

        // m-vector for this vertex order
        float mv[3];
        if (v_order == 0) {
            mv[0] = -(Bm(0,0) + Bm(1,0) + Bm(2,0));
            mv[1] = -(Bm(0,1) + Bm(1,1) + Bm(2,1));
            mv[2] = -(Bm(0,2) + Bm(1,2) + Bm(2,2));
        } else if (v_order == 1) {
            mv[0] = Bm(0,0); mv[1] = Bm(0,1); mv[2] = Bm(0,2);
        } else if (v_order == 2) {
            mv[0] = Bm(1,0); mv[1] = Bm(1,1); mv[2] = Bm(1,2);
        } else {
            mv[0] = Bm(2,0); mv[1] = Bm(2,1); mv[2] = Bm(2,2);
        }

        // force = -G^T * P_vec
        math::Vec3f f_tet;
        f_tet.x = -(P_vec[0]*mv[0] + P_vec[3]*mv[1] + P_vec[6]*mv[2]);
        f_tet.y = -(P_vec[1]*mv[0] + P_vec[4]*mv[1] + P_vec[7]*mv[2]);
        f_tet.z = -(P_vec[2]*mv[0] + P_vec[5]*mv[1] + P_vec[8]*mv[2]);

        // G^T * cof_vec
        float gc[3];
        gc[0] = cof_vec[0]*mv[0] + cof_vec[3]*mv[1] + cof_vec[6]*mv[2];
        gc[1] = cof_vec[1]*mv[0] + cof_vec[4]*mv[1] + cof_vec[7]*mv[2];
        gc[2] = cof_vec[2]*mv[0] + cof_vec[5]*mv[1] + cof_vec[8]*mv[2];

        // H_tet = mu_V*(m.m)*I + lmbd_V*(gc outer gc)
        const float m_sq = mv[0]*mv[0] + mv[1]*mv[1] + mv[2]*mv[2];
        const float diag_mu = mu_V * m_sq;
        math::Mat3f h_tet(
            diag_mu + lmbd_V*gc[0]*gc[0], lmbd_V*gc[0]*gc[1], lmbd_V*gc[0]*gc[2],
            lmbd_V*gc[1]*gc[0], diag_mu + lmbd_V*gc[1]*gc[1], lmbd_V*gc[1]*gc[2],
            lmbd_V*gc[2]*gc[0], lmbd_V*gc[2]*gc[1], diag_mu + lmbd_V*gc[2]*gc[2]);

        // Damping
        if (k_damp > 0.0f) {
            const float inv_dt = 1.0f / dt;
            const math::Vec3f v0p = q_prev[ti.x], v1p = q_prev[ti.y];
            const math::Vec3f v2p = q_prev[ti.z], v3p = q_prev[ti.w];
            const math::Vec3f d0 = (v0 - v0p) * inv_dt;
            const math::Vec3f d1 = (v1 - v1p) * inv_dt;
            const math::Vec3f d2 = (v2 - v2p) * inv_dt;
            const math::Vec3f d3 = (v3 - v3p) * inv_dt;
            const math::Mat3f Ds_dot = vbd_mat3_from_cols(d1-d0, d2-d0, d3-d0);
            const math::Mat3f F_dot = Ds_dot * Bm;

            float f_dot[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    f_dot[c*3+r] = F_dot(r, c);

            // H*f_dot = V0*(mu*f_dot + lambda*(cof.f_dot)*cof)
            float cof_dot_fdot = 0.0f;
            for (int k = 0; k < 9; ++k)
                cof_dot_fdot += cof_vec[k] * f_dot[k];

            float P_damp[9];
            for (int k = 0; k < 9; ++k)
                P_damp[k] = k_damp * rest_volume * (mu_nh*f_dot[k] + lmbd_nh*cof_dot_fdot*cof_vec[k]);

            math::Vec3f f_d;
            f_d.x = -(P_damp[0]*mv[0] + P_damp[3]*mv[1] + P_damp[6]*mv[2]);
            f_d.y = -(P_damp[1]*mv[0] + P_damp[4]*mv[1] + P_damp[7]*mv[2]);
            f_d.z = -(P_damp[2]*mv[0] + P_damp[5]*mv[1] + P_damp[8]*mv[2]);
            f_tet.x += f_d.x; f_tet.y += f_d.y; f_tet.z += f_d.z;
            h_tet *= (1.0f + k_damp * inv_dt);
        }

        // VBD uses force (= -grad E), so f_tet is force direction
        f = f + f_tet;
        h = h + h_tet;
    }

    // Add external contact forces/hessians (from body-particle contacts)
    if (ext_forces != nullptr)  f = f + ext_forces[pi];
    if (ext_hessians != nullptr) h = h + ext_hessians[pi];

    // Local 3x3 solve: displacement += H^{-1} * f
    const float det_h = determinant(h);
    if (fabsf(det_h) > 1.0e-8f) {
        const math::Mat3f h_inv = inverse(h);
        displacements[pi] = displacements[pi] + h_inv * f;
    }

    // Update position
    pos[pi] = q_prev[pi] + displacements[pi];
}

// -----------------------------------------------------------------------
// update_velocity kernel
// -----------------------------------------------------------------------
__global__ void vbd_update_velocity_kernel(
    const math::Vec3f* __restrict__ pos,
    const math::Vec3f* __restrict__ q_prev,
    math::Vec3f* __restrict__ vel,
    float inv_dt,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i] = (pos[i] - q_prev[i]) * inv_dt;
}

}  // namespace

// ====================================================================
// VBDSolver host methods
// ====================================================================

void VBDSolver::build_coloring(
    const math::Vec4i* host_tets, int n_tets, int n_particles) {

    n_particles_ = n_particles;

    // Build graph edges from tet connectivity: 6 edges per tet
    // (all C(4,2) pairs), canonicalized (u < v), deduplicated.
    std::set<std::pair<int,int>> edge_set;
    for (int t = 0; t < n_tets; ++t) {
        const int v[4] = { host_tets[t].x, host_tets[t].y,
                           host_tets[t].z, host_tets[t].w };
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                int u = v[a], w = v[b];
                if (u > w) std::swap(u, w);
                edge_set.insert({u, w});
            }
        }
    }

    // Build adjacency list for greedy coloring
    std::vector<std::vector<int>> adj(n_particles);
    for (auto& [u, v] : edge_set) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Greedy graph coloring (largest-degree-first ordering)
    std::vector<int> order(n_particles);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });

    std::vector<int> colors(n_particles, -1);
    int num_colors = 0;

    for (int idx : order) {
        std::set<int> neighbor_colors;
        for (int nb : adj[idx]) {
            if (colors[nb] >= 0)
                neighbor_colors.insert(colors[nb]);
        }
        int c = 0;
        while (neighbor_colors.count(c)) ++c;
        colors[idx] = c;
        if (c >= num_colors) num_colors = c + 1;
    }

    // Build color groups
    color_groups_.resize(num_colors);
    std::vector<std::vector<int>> groups(num_colors);
    for (int i = 0; i < n_particles; ++i) {
        groups[colors[i]].push_back(i);
    }
    for (int c = 0; c < num_colors; ++c) {
        color_groups_[c].count = static_cast<int>(groups[c].size());
        color_groups_[c].indices.resize(groups[c].size());
        std::memcpy(color_groups_[c].indices.cpu_data(),
                     groups[c].data(),
                     groups[c].size() * sizeof(int));
        color_groups_[c].indices.copy_to_device();
    }

    // Upload particle_colors
    particle_colors_.resize(n_particles);
    std::memcpy(particle_colors_.cpu_data(), colors.data(),
                n_particles * sizeof(int));
    particle_colors_.copy_to_device();

    // Pre-allocate work buffers so step() never allocates during
    // CUDA graph capture.
    initialize(n_particles);
}

void VBDSolver::set_coloring(const int* host_colors, int n_particles) {
    n_particles_ = n_particles;

    int num_colors = 0;
    for (int i = 0; i < n_particles; ++i) {
        if (host_colors[i] >= num_colors)
            num_colors = host_colors[i] + 1;
    }

    std::vector<std::vector<int>> groups(num_colors);
    for (int i = 0; i < n_particles; ++i) {
        groups[host_colors[i]].push_back(i);
    }

    color_groups_.resize(num_colors);
    for (int c = 0; c < num_colors; ++c) {
        color_groups_[c].count = static_cast<int>(groups[c].size());
        color_groups_[c].indices.resize(groups[c].size());
        std::memcpy(color_groups_[c].indices.cpu_data(),
                     groups[c].data(),
                     groups[c].size() * sizeof(int));
        color_groups_[c].indices.copy_to_device();
    }

    particle_colors_.resize(n_particles);
    std::memcpy(particle_colors_.cpu_data(), host_colors,
                n_particles * sizeof(int));
    particle_colors_.copy_to_device();

    initialize(n_particles);
}

void VBDSolver::build_adjacency(
    const math::Vec4i* host_tets, int n_tets, int n_particles) {

    // Count adjacent tets per vertex
    std::vector<int> counts(n_particles, 0);
    for (int t = 0; t < n_tets; ++t) {
        ++counts[host_tets[t].x];
        ++counts[host_tets[t].y];
        ++counts[host_tets[t].z];
        ++counts[host_tets[t].w];
    }

    // Build offsets (each entry stores 2 ints: tet_id + v_order)
    std::vector<int> offsets(n_particles + 1);
    offsets[0] = 0;
    for (int i = 0; i < n_particles; ++i) {
        offsets[i + 1] = offsets[i] + counts[i] * 2;
    }

    // Fill adjacency data
    const int total = offsets[n_particles];
    std::vector<int> data(total);
    std::vector<int> fill(n_particles, 0);

    for (int t = 0; t < n_tets; ++t) {
        const int v[4] = { host_tets[t].x, host_tets[t].y,
                           host_tets[t].z, host_tets[t].w };
        for (int k = 0; k < 4; ++k) {
            const int vi = v[k];
            const int pos = offsets[vi] + fill[vi] * 2;
            data[pos] = t;
            data[pos + 1] = k;
            ++fill[vi];
        }
    }

    // Upload
    tet_adj_.offsets.resize(n_particles + 1);
    std::memcpy(tet_adj_.offsets.cpu_data(), offsets.data(),
                (n_particles + 1) * sizeof(int));
    tet_adj_.offsets.copy_to_device();

    tet_adj_.data.resize(total);
    if (total > 0) {
        std::memcpy(tet_adj_.data.cpu_data(), data.data(),
                     total * sizeof(int));
        tet_adj_.data.copy_to_device();
    }
}

void VBDSolver::initialize(int n) {
    const auto sz = static_cast<std::size_t>(n);
    if (q_prev_.gpu_size() != sz) q_prev_.resize(sz);
    if (inertia_.gpu_size() != sz) inertia_.resize(sz);
    if (displacements_.gpu_size() != sz) displacements_.resize(sz);
    if (mass_.gpu_size() != sz) mass_.resize(sz);
}

void VBDSolver::step(
    DeviceSpan<math::Vec3f> pos,
    DeviceSpan<math::Vec3f> vel,
    DeviceSpan<float> inv_mass,
    DeviceSpan<math::Vec4i> tet_indices,
    DeviceSpan<math::Mat3f> tet_poses,
    DeviceSpan<math::Vec3f> tet_materials,
    math::Vec3f gravity,
    float dt,
    int iterations,
    std::uintptr_t cuda_stream) {

    const int n = static_cast<int>(pos.size());
    if (n <= 0) return;

    initialize(n);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    constexpr int block = 256;
    const int grid = (n + block - 1) / block;

    // Phase 1: forward_step
    vbd_forward_step_kernel<<<grid, block, 0, stream>>>(
        pos.data(), vel.data(), inv_mass.data(),
        q_prev_.gpu_data(), inertia_.gpu_data(),
        displacements_.gpu_data(), mass_.gpu_data(),
        gravity, dt, n);
    check_cuda(cudaGetLastError(), "vbd_forward_step_kernel");

    // Phase 2: VBD iterations
    for (int iter = 0; iter < iterations; ++iter) {
        for (int c = 0; c < static_cast<int>(color_groups_.size()); ++c) {
            const int n_color = color_groups_[c].count;
            if (n_color == 0) continue;

            const int grid_c = (n_color + block - 1) / block;
            vbd_solve_elasticity_kernel<<<grid_c, block, 0, stream>>>(
                color_groups_[c].indices.gpu_data(),
                n_color,
                dt,
                pos.data(),
                q_prev_.gpu_data(),
                mass_.gpu_data(),
                inertia_.gpu_data(),
                tet_indices.data(),
                tet_poses.data(),
                tet_materials.data(),
                tet_adj_.offsets.gpu_data(),
                tet_adj_.data.gpu_data(),
                displacements_.gpu_data(),
                nullptr, nullptr);
            check_cuda(cudaGetLastError(), "vbd_solve_elasticity_kernel");
        }
    }

    // Phase 3: update velocity
    const float inv_dt = 1.0f / dt;
    vbd_update_velocity_kernel<<<grid, block, 0, stream>>>(
        pos.data(), q_prev_.gpu_data(), vel.data(), inv_dt, n);
    check_cuda(cudaGetLastError(), "vbd_update_velocity_kernel");
}

void VBDSolver::step_with_contacts(
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
    std::uintptr_t cuda_stream) {

    const int n = static_cast<int>(pos.size());
    if (n <= 0) return;

    initialize(n);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    constexpr int block = 256;
    const int grid = (n + block - 1) / block;

    // Phase 1: forward_step
    vbd_forward_step_kernel<<<grid, block, 0, stream>>>(
        pos.data(), vel.data(), inv_mass.data(),
        q_prev_.gpu_data(), inertia_.gpu_data(),
        displacements_.gpu_data(), mass_.gpu_data(),
        gravity, dt, n);
    check_cuda(cudaGetLastError(), "vbd_forward_step_kernel (contacts)");

    const bool has_contacts = (contact_cb != nullptr)
                           && (particle_forces != nullptr)
                           && (particle_hessians != nullptr);

    // Phase 2: VBD iterations with per-color contact accumulation
    for (int iter = 0; iter < iterations; ++iter) {
        for (int c = 0; c < static_cast<int>(color_groups_.size()); ++c) {
            const int n_color = color_groups_[c].count;
            if (n_color == 0) continue;

            // Zero and accumulate contacts for this color
            if (has_contacts) {
                contact_cb(c, q_prev_.gpu_data(), pos.data(), mass_.gpu_data(),
                          particle_forces, particle_hessians,
                          stream, contact_cb_data);
            }

            const int grid_c = (n_color + block - 1) / block;
            vbd_solve_elasticity_kernel<<<grid_c, block, 0, stream>>>(
                color_groups_[c].indices.gpu_data(),
                n_color,
                dt,
                pos.data(),
                q_prev_.gpu_data(),
                mass_.gpu_data(),
                inertia_.gpu_data(),
                tet_indices.data(),
                tet_poses.data(),
                tet_materials.data(),
                tet_adj_.offsets.gpu_data(),
                tet_adj_.data.gpu_data(),
                displacements_.gpu_data(),
                particle_forces,
                particle_hessians);
            check_cuda(cudaGetLastError(), "vbd_solve_elasticity_kernel (contacts)");
        }
    }

    // Phase 3: update velocity
    const float inv_dt = 1.0f / dt;
    vbd_update_velocity_kernel<<<grid, block, 0, stream>>>(
        pos.data(), q_prev_.gpu_data(), vel.data(), inv_dt, n);
    check_cuda(cudaGetLastError(), "vbd_update_velocity_kernel (contacts)");
}

}  // namespace solver
}  // namespace chysx
