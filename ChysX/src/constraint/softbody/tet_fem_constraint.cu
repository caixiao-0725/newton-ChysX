// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::constraint::TetFEMConstraint.
//
// Stable Neo-Hookean constitutive model (Smith et al. 2018), ported
// 1:1 from Newton's VBD kernel `evaluate_volumetric_neo_hookean_force_and_hessian`.

#include "tet_fem_constraint.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "../../sparse/block_csr_atomic.cuh"

namespace chysx {
namespace constraint {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::constraint::TetFEMConstraint: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// -----------------------------------------------------------------------
// Cofactor (adjugate) matrix — stable even when det(F) ~ 0.
// Matches Newton's `compute_cofactor` exactly.
// -----------------------------------------------------------------------
__device__ __forceinline__ math::Mat3f cofactor(const math::Mat3f& F) {
    const float F11 = F(0,0), F21 = F(1,0), F31 = F(2,0);
    const float F12 = F(0,1), F22 = F(1,1), F32 = F(2,1);
    const float F13 = F(0,2), F23 = F(1,2), F33 = F(2,2);
    return math::Mat3f(
        F22*F33 - F23*F32, F23*F31 - F21*F33, F21*F32 - F22*F31,
        F13*F32 - F12*F33, F11*F33 - F13*F31, F12*F31 - F11*F32,
        F12*F23 - F13*F22, F13*F21 - F11*F23, F11*F22 - F12*F21);
}

// -----------------------------------------------------------------------
// mat3_from_cols: build 3x3 matrix from three column vectors.
// ChysX Mat3 is row-major, so col c, row r = data[r*3+c].
// -----------------------------------------------------------------------
__device__ __forceinline__ math::Mat3f mat3_from_cols(
    const math::Vec3f& c0, const math::Vec3f& c1, const math::Vec3f& c2) {
    return math::Mat3f(
        c0.x, c1.x, c2.x,
        c0.y, c1.y, c2.y,
        c0.z, c1.z, c2.z);
}

// -----------------------------------------------------------------------
// vec9: column-major flattening of a 3x3 matrix, matching Warp's layout.
//   idx 0,1,2 = col 0; idx 3,4,5 = col 1; idx 6,7,8 = col 2.
// -----------------------------------------------------------------------
struct Vec9f { float v[9]; };

__device__ __forceinline__ Vec9f flatten_col_major(const math::Mat3f& M) {
    Vec9f r;
    r.v[0] = M(0,0); r.v[1] = M(1,0); r.v[2] = M(2,0);
    r.v[3] = M(0,1); r.v[4] = M(1,1); r.v[5] = M(2,1);
    r.v[6] = M(0,2); r.v[7] = M(1,2); r.v[8] = M(2,2);
    return r;
}

// -----------------------------------------------------------------------
// assemble_tet_vertex_force_and_hessian
//
// Given P_vec (9-vec), H (conceptual 9x9 = mu*I + lambda*cof outer cof),
// and the m-vector (m0, m1, m2) for this vertex order, compute:
//   force  = -G^T * P_vec
//   hessian = G^T * H * G
// where G = [m0*I3; m1*I3; m2*I3] (9x3).
//
// Instead of forming the full 9x9 H, we exploit its rank-1+diagonal
// structure: H = mu*I9 + lambda*(cof outer cof). Then:
//   G^T * (mu*I9) * G = mu * (m0^2+m1^2+m2^2) * I3
//   G^T * (cof outer cof) * G = (G^T cof) outer (G^T cof)
//
// But we keep the expanded form for clarity and numerical fidelity.
// -----------------------------------------------------------------------

__device__ __forceinline__ void assemble_vertex(
    const Vec9f& P_vec,
    float mu_nh_V, float lmbd_nh_V,
    const Vec9f& cof_vec,
    float m0, float m1, float m2,
    math::Vec3f& force_out,
    math::Mat3f& hessian_out) {

    // force = -G^T * P_vec
    force_out.x = -(P_vec.v[0]*m0 + P_vec.v[3]*m1 + P_vec.v[6]*m2);
    force_out.y = -(P_vec.v[1]*m0 + P_vec.v[4]*m1 + P_vec.v[7]*m2);
    force_out.z = -(P_vec.v[2]*m0 + P_vec.v[5]*m1 + P_vec.v[8]*m2);

    // G^T * cof_vec (3-vector)
    const float gc0 = cof_vec.v[0]*m0 + cof_vec.v[3]*m1 + cof_vec.v[6]*m2;
    const float gc1 = cof_vec.v[1]*m0 + cof_vec.v[4]*m1 + cof_vec.v[7]*m2;
    const float gc2 = cof_vec.v[2]*m0 + cof_vec.v[5]*m1 + cof_vec.v[8]*m2;

    // hessian = mu_V * (m0^2+m1^2+m2^2)*I3 + lmbd_V * (gc outer gc)
    const float m_sq = m0*m0 + m1*m1 + m2*m2;
    const float diag_mu = mu_nh_V * m_sq;
    hessian_out = math::Mat3f(
        diag_mu + lmbd_nh_V * gc0*gc0,
                  lmbd_nh_V * gc0*gc1,
                  lmbd_nh_V * gc0*gc2,
                  lmbd_nh_V * gc1*gc0,
        diag_mu + lmbd_nh_V * gc1*gc1,
                  lmbd_nh_V * gc1*gc2,
                  lmbd_nh_V * gc2*gc0,
                  lmbd_nh_V * gc2*gc1,
        diag_mu + lmbd_nh_V * gc2*gc2);
}

// -----------------------------------------------------------------------
// assemble_all_vertices: compute force/hessian for all 4 vertices of a tet.
// Used by both gradient and Hessian kernels.
// -----------------------------------------------------------------------
struct TetForceHessian {
    math::Vec3f f[4];
    math::Mat3f h[4];
};

__device__ __forceinline__ TetForceHessian compute_tet_force_hessian(
    const math::Vec3f& v0, const math::Vec3f& v1,
    const math::Vec3f& v2, const math::Vec3f& v3,
    const math::Mat3f& Dm_inv,
    float mu, float lmbd) {

    TetForceHessian out;

    // rest volume
    const float det_Dm_inv = determinant(Dm_inv);
    const float rest_volume = 1.0f / (det_Dm_inv * 6.0f);

    // deformation gradient
    const math::Mat3f Ds = mat3_from_cols(v1 - v0, v2 - v0, v3 - v0);
    const math::Mat3f F = Ds * Dm_inv;

    // flatten to vec9 (column-major)
    const Vec9f f_vec = flatten_col_major(F);

    // determinant J
    const float J = determinant(F);

    // stable Neo-Hookean Lamé conversion
    const float mu_nh = mu;
    const float lmbd_nh = lmbd + mu;
    const float lmbd_safe = (lmbd_nh >= 0.0f ? 1.0f : -1.0f)
                             * fmaxf(fabsf(lmbd_nh), 1.0e-6f);
    const float alpha = 1.0f + mu_nh / lmbd_safe;

    // cofactor
    const math::Mat3f cof = cofactor(F);
    const Vec9f cof_vec = flatten_col_major(cof);

    // stress (P_vec = V0 * (mu_nh * f + s * cof_vec))
    const float s = lmbd_nh * (J - alpha);
    Vec9f P_vec;
    for (int i = 0; i < 9; ++i)
        P_vec.v[i] = rest_volume * (mu_nh * f_vec.v[i] + s * cof_vec.v[i]);

    // scaled material constants for Hessian
    const float mu_V = rest_volume * mu_nh;
    const float lmbd_V = rest_volume * lmbd_nh;

    // m-vectors for each vertex order
    const float m_vecs[4][3] = {
        { -(Dm_inv(0,0) + Dm_inv(1,0) + Dm_inv(2,0)),
          -(Dm_inv(0,1) + Dm_inv(1,1) + Dm_inv(2,1)),
          -(Dm_inv(0,2) + Dm_inv(1,2) + Dm_inv(2,2)) },
        {  Dm_inv(0,0), Dm_inv(0,1), Dm_inv(0,2) },
        {  Dm_inv(1,0), Dm_inv(1,1), Dm_inv(1,2) },
        {  Dm_inv(2,0), Dm_inv(2,1), Dm_inv(2,2) },
    };

    for (int k = 0; k < 4; ++k) {
        assemble_vertex(P_vec, mu_V, lmbd_V, cof_vec,
                        m_vecs[k][0], m_vecs[k][1], m_vecs[k][2],
                        out.f[k], out.h[k]);
    }
    return out;
}

// -----------------------------------------------------------------------
// Damping: Rayleigh (stiffness-proportional), matching Newton exactly.
// -----------------------------------------------------------------------
__device__ __forceinline__ void apply_damping(
    TetForceHessian& fh,
    const math::Vec3f& v0, const math::Vec3f& v1,
    const math::Vec3f& v2, const math::Vec3f& v3,
    const math::Vec3f& v0_prev, const math::Vec3f& v1_prev,
    const math::Vec3f& v2_prev, const math::Vec3f& v3_prev,
    const math::Mat3f& Dm_inv,
    float mu, float lmbd, float k_damp, float dt) {

    if (k_damp <= 0.0f) return;

    const float inv_dt = 1.0f / dt;
    const float rest_volume = 1.0f / (determinant(Dm_inv) * 6.0f);
    const float mu_nh = mu;
    const float lmbd_nh = lmbd + mu;

    // F_dot = Ds_dot * Dm_inv
    const math::Mat3f Ds = mat3_from_cols(v1 - v0, v2 - v0, v3 - v0);
    const math::Mat3f F = Ds * Dm_inv;
    const math::Mat3f cof = cofactor(F);
    const Vec9f cof_vec = flatten_col_major(cof);

    const math::Vec3f d0 = (v0 - v0_prev) * inv_dt;
    const math::Vec3f d1 = (v1 - v1_prev) * inv_dt;
    const math::Vec3f d2 = (v2 - v2_prev) * inv_dt;
    const math::Vec3f d3 = (v3 - v3_prev) * inv_dt;
    const math::Mat3f Ds_dot = mat3_from_cols(d1 - d0, d2 - d0, d3 - d0);
    const math::Mat3f F_dot = Ds_dot * Dm_inv;
    const Vec9f f_dot = flatten_col_major(F_dot);

    // H_9x9 * f_dot (H = V0*(mu*I + lambda*cof outer cof))
    // = V0 * (mu*f_dot + lambda*(cof . f_dot)*cof)
    float cof_dot_fdot = 0.0f;
    for (int i = 0; i < 9; ++i)
        cof_dot_fdot += cof_vec.v[i] * f_dot.v[i];

    Vec9f P_damp;
    for (int i = 0; i < 9; ++i) {
        const float H_fdot_i = rest_volume * (mu_nh * f_dot.v[i]
                              + lmbd_nh * cof_dot_fdot * cof_vec.v[i]);
        P_damp.v[i] = k_damp * H_fdot_i;
    }

    // m-vectors
    const float m_vecs[4][3] = {
        { -(Dm_inv(0,0) + Dm_inv(1,0) + Dm_inv(2,0)),
          -(Dm_inv(0,1) + Dm_inv(1,1) + Dm_inv(2,1)),
          -(Dm_inv(0,2) + Dm_inv(1,2) + Dm_inv(2,2)) },
        {  Dm_inv(0,0), Dm_inv(0,1), Dm_inv(0,2) },
        {  Dm_inv(1,0), Dm_inv(1,1), Dm_inv(1,2) },
        {  Dm_inv(2,0), Dm_inv(2,1), Dm_inv(2,2) },
    };

    const float hessian_scale = 1.0f + k_damp * inv_dt;
    for (int k = 0; k < 4; ++k) {
        const float m0 = m_vecs[k][0], m1 = m_vecs[k][1], m2 = m_vecs[k][2];
        math::Vec3f f_d;
        f_d.x = -(P_damp.v[0]*m0 + P_damp.v[3]*m1 + P_damp.v[6]*m2);
        f_d.y = -(P_damp.v[1]*m0 + P_damp.v[4]*m1 + P_damp.v[7]*m2);
        f_d.z = -(P_damp.v[2]*m0 + P_damp.v[5]*m1 + P_damp.v[8]*m2);
        fh.f[k].x += f_d.x;
        fh.f[k].y += f_d.y;
        fh.f[k].z += f_d.z;
        fh.h[k] *= hessian_scale;
    }
}

// ---------------------------------------------------------------------------
// Dm_inv init kernel: compute inverse rest shape matrix from positions.
// ---------------------------------------------------------------------------
__global__ void tet_dm_inv_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Vec3f* __restrict__ positions,
    math::Mat3f* __restrict__ Dm_inv_out,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const math::Vec3f p0 = positions[id.x];
    const math::Vec3f p1 = positions[id.y];
    const math::Vec3f p2 = positions[id.z];
    const math::Vec3f p3 = positions[id.w];

    const math::Mat3f Dm = mat3_from_cols(p1 - p0, p2 - p0, p3 - p0);
    Dm_inv_out[e] = inverse(Dm);
}

// ---------------------------------------------------------------------------
// Energy kernel
// ---------------------------------------------------------------------------
__global__ void tet_energy_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Mat3f* __restrict__ Dm_inv,
    const math::Vec3f* __restrict__ materials,
    const math::Vec3f* __restrict__ positions,
    float* __restrict__ out,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const math::Vec3f v0 = positions[id.x], v1 = positions[id.y];
    const math::Vec3f v2 = positions[id.z], v3 = positions[id.w];
    const math::Mat3f& Bm = Dm_inv[e];

    const float rest_volume = 1.0f / (determinant(Bm) * 6.0f);
    const math::Mat3f Ds = mat3_from_cols(v1 - v0, v2 - v0, v3 - v0);
    const math::Mat3f F = Ds * Bm;

    const float mu = materials[e].x;
    const float lmbd = materials[e].y;
    const float mu_nh = mu;
    const float lmbd_nh = lmbd + mu;
    const float lmbd_safe = (lmbd_nh >= 0.0f ? 1.0f : -1.0f)
                            * fmaxf(fabsf(lmbd_nh), 1.0e-6f);
    const float alpha = 1.0f + mu_nh / lmbd_safe;

    float F_sq = 0.0f;
    for (int i = 0; i < 9; ++i) F_sq += F.data[i] * F.data[i];
    const float J = determinant(F);
    const float psi = 0.5f * mu_nh * (F_sq - 3.0f)
                    + 0.5f * lmbd_nh * (J - alpha) * (J - alpha);

    atomicAdd(out, rest_volume * psi);
}

// ---------------------------------------------------------------------------
// Gradient scatter kernel
// ---------------------------------------------------------------------------
__global__ void tet_gradient_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Mat3f* __restrict__ Dm_inv_buf,
    const math::Vec3f* __restrict__ materials,
    const math::Vec3f* __restrict__ positions,
    const math::Vec3f* __restrict__ pos_prev,
    float dt,
    math::Vec3f* __restrict__ out_grad,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const math::Vec3f v0 = positions[id.x], v1 = positions[id.y];
    const math::Vec3f v2 = positions[id.z], v3 = positions[id.w];
    const math::Mat3f& Bm = Dm_inv_buf[e];
    const float mu = materials[e].x;
    const float lmbd = materials[e].y;
    const float k_damp = materials[e].z;

    TetForceHessian fh = compute_tet_force_hessian(v0, v1, v2, v3, Bm, mu, lmbd);

    if (k_damp > 0.0f && pos_prev != nullptr) {
        const math::Vec3f v0p = pos_prev[id.x], v1p = pos_prev[id.y];
        const math::Vec3f v2p = pos_prev[id.z], v3p = pos_prev[id.w];
        apply_damping(fh, v0, v1, v2, v3, v0p, v1p, v2p, v3p,
                      Bm, mu, lmbd, k_damp, dt);
    }

    const int idx[4] = { id.x, id.y, id.z, id.w };
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        // fh.f[k] = -dE/dx (force).  The constraint protocol
        // accumulates +dE/dx (gradient), so we negate.
        atomicAdd(&out_grad[idx[k]].x, -fh.f[k].x);
        atomicAdd(&out_grad[idx[k]].y, -fh.f[k].y);
        atomicAdd(&out_grad[idx[k]].z, -fh.f[k].z);
    }
}

// ---------------------------------------------------------------------------
// Hessian scatter kernel
//
// Each tet contributes 16 = 4x4 local 3x3 Hessian blocks.
// Off-diagonal (a != b) blocks:
//   H_{a,b} = G_a^T * H_9x9 * G_b
// where G_a, G_b use their respective m-vectors.
//
// This is computed by exploiting H's structure:
//   G_a^T (mu*I + lambda*(c outer c)) G_b
//   = mu * (m_a . m_b) * I3 + lambda * (G_a^T c)(G_b^T c)^T
// ---------------------------------------------------------------------------
__global__ void tet_hessian_scatter_kernel(
    const math::Vec4i* __restrict__ verts,
    const math::Mat3f* __restrict__ Dm_inv_buf,
    const math::Vec3f* __restrict__ materials,
    const math::Vec3f* __restrict__ positions,
    const math::Vec3f* __restrict__ pos_prev,
    float dt,
    const int* __restrict__ slots,
    math::Mat3f* __restrict__ A_diag,
    math::Mat3f* __restrict__ A_values,
    int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;

    const math::Vec4i id = verts[e];
    const math::Vec3f v0 = positions[id.x], v1 = positions[id.y];
    const math::Vec3f v2 = positions[id.z], v3 = positions[id.w];
    const math::Mat3f& Bm = Dm_inv_buf[e];
    const float mu = materials[e].x;
    const float lmbd = materials[e].y;
    const float k_damp = materials[e].z;

    // Rest volume and deformation gradient
    const float rest_volume = 1.0f / (determinant(Bm) * 6.0f);
    const math::Mat3f Ds = mat3_from_cols(v1 - v0, v2 - v0, v3 - v0);
    const math::Mat3f F = Ds * Bm;

    const float mu_nh = mu;
    const float lmbd_nh = lmbd + mu;
    const math::Mat3f cof = cofactor(F);
    const Vec9f cof_vec = flatten_col_major(cof);

    const float mu_V = rest_volume * mu_nh;
    const float lmbd_V = rest_volume * lmbd_nh;

    // m-vectors for each vertex
    float m[4][3];
    m[0][0] = -(Bm(0,0) + Bm(1,0) + Bm(2,0));
    m[0][1] = -(Bm(0,1) + Bm(1,1) + Bm(2,1));
    m[0][2] = -(Bm(0,2) + Bm(1,2) + Bm(2,2));
    m[1][0] = Bm(0,0); m[1][1] = Bm(0,1); m[1][2] = Bm(0,2);
    m[2][0] = Bm(1,0); m[2][1] = Bm(1,1); m[2][2] = Bm(1,2);
    m[3][0] = Bm(2,0); m[3][1] = Bm(2,1); m[3][2] = Bm(2,2);

    // G_k^T * cof_vec for each vertex
    float gc[4][3];
    for (int k = 0; k < 4; ++k) {
        gc[k][0] = cof_vec.v[0]*m[k][0] + cof_vec.v[3]*m[k][1] + cof_vec.v[6]*m[k][2];
        gc[k][1] = cof_vec.v[1]*m[k][0] + cof_vec.v[4]*m[k][1] + cof_vec.v[7]*m[k][2];
        gc[k][2] = cof_vec.v[2]*m[k][0] + cof_vec.v[5]*m[k][1] + cof_vec.v[8]*m[k][2];
    }

    // Damping scale
    float hessian_scale = 1.0f;
    if (k_damp > 0.0f) {
        hessian_scale = 1.0f + k_damp / dt;
    }

    const int base = 16 * e;

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            // H_{a,b} = mu_V * (m_a . m_b) * I3 + lmbd_V * gc_a outer gc_b
            const float m_dot = m[a][0]*m[b][0] + m[a][1]*m[b][1] + m[a][2]*m[b][2];
            const float diag_val = mu_V * m_dot;

            math::Mat3f H_ab(
                diag_val + lmbd_V * gc[a][0]*gc[b][0],
                           lmbd_V * gc[a][0]*gc[b][1],
                           lmbd_V * gc[a][0]*gc[b][2],
                           lmbd_V * gc[a][1]*gc[b][0],
                diag_val + lmbd_V * gc[a][1]*gc[b][1],
                           lmbd_V * gc[a][1]*gc[b][2],
                           lmbd_V * gc[a][2]*gc[b][0],
                           lmbd_V * gc[a][2]*gc[b][1],
                diag_val + lmbd_V * gc[a][2]*gc[b][2]);

            if (hessian_scale != 1.0f)
                H_ab *= hessian_scale;

            sparse::scatter_hessian_block(
                slots[base + 4*a + b], A_diag, A_values, H_ab);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// TetFEMConstraint public methods
// ---------------------------------------------------------------------------

void TetFEMConstraint::set_tets_from_positions(
    const math::Vec4i* host_tets,
    const math::Vec3f* host_materials,
    int n,
    DeviceSpan<math::Vec3f> positions,
    std::uintptr_t cuda_stream) {
    if (n < 0) {
        throw std::invalid_argument(
            "TetFEMConstraint::set_tets_from_positions: negative count");
    }

    indices_.resize(static_cast<std::size_t>(n));
    Dm_inv_.resize(static_cast<std::size_t>(n));
    materials_.resize(static_cast<std::size_t>(n));

    if (n == 0) return;

    std::memcpy(indices_.cpu_data(), host_tets, n * sizeof(math::Vec4i));
    indices_.copy_to_device(cuda_stream);

    std::memcpy(materials_.cpu_data(), host_materials, n * sizeof(math::Vec3f));
    materials_.copy_to_device(cuda_stream);

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    tet_dm_inv_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        positions.data(),
        Dm_inv_.gpu_data(),
        n);
    check_cuda(cudaGetLastError(), "tet_dm_inv_kernel launch");
}

float TetFEMConstraint::compute_energy(
    DeviceSpan<math::Vec3f> positions,
    std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return 0.0f;

    if (energy_buffer_.gpu_size() < 1) {
        energy_buffer_.resize(1);
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemsetAsync(energy_buffer_.gpu_data(), 0, sizeof(float), stream),
               "cudaMemsetAsync(energy)");

    tet_energy_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        materials_.gpu_data(),
        positions.data(),
        energy_buffer_.gpu_data(),
        n);
    check_cuda(cudaGetLastError(), "tet_energy_kernel launch");

    energy_buffer_.copy_to_host(cuda_stream);
    return energy_buffer_.cpu_data()[0];
}

void TetFEMConstraint::accumulate_gradient(
    DeviceSpan<math::Vec3f> positions,
    DeviceSpan<math::Vec3f> out_grad,
    std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    tet_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        materials_.gpu_data(),
        positions.data(),
        pos_prev_.data(),
        dt_,
        out_grad.data(),
        n);
    check_cuda(cudaGetLastError(), "tet_gradient_kernel launch");
}

void TetFEMConstraint::accumulate_hessian(
    DeviceSpan<math::Vec3f> positions,
    sparse::BlockCSR3& A,
    std::uintptr_t cuda_stream) const {
    const int n = size();
    if (n == 0) return;

    if (static_cast<int>(hessian_slots_.gpu_size()) < 16 * n) {
        throw std::runtime_error(
            "TetFEMConstraint::accumulate_hessian: bind_hessian_layout(A) "
            "must be called before stepping");
    }

    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    tet_hessian_scatter_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        indices_.gpu_data(),
        Dm_inv_.gpu_data(),
        materials_.gpu_data(),
        positions.data(),
        pos_prev_.data(),
        dt_,
        hessian_slots_.gpu_data(),
        A.diag.gpu_data(),
        A.values.gpu_data(),
        n);
    check_cuda(cudaGetLastError(), "tet_hessian_scatter_kernel launch");
}

}  // namespace constraint
}  // namespace chysx
