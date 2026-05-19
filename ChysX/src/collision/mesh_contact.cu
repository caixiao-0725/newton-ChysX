// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of `MeshContact`.
//
// The detect kernel uses the LinearBvh (built over world-space triangle
// AABBs) with a per-particle stack-based traversal to find the closest
// triangle.  Signed distance is determined via pseudo-normal sign
// (accumulated vertex/edge/face normals at the closest feature).
//
// The gradient / Hessian / friction kernels are identical to those in
// sdf_contact.cu — they operate on the same per-particle
// (nx, ny, nz, depth) cache format.

#include "mesh_contact.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "bvh/quant_bvh.h"

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::MeshContact: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// =====================================================================
// Rigid-transform mesh vertices: world_v = R * rest_v + pos
// =====================================================================

__global__ void transform_vertices_kernel(
    const math::Vec3f* __restrict__ rest,
    math::Vec3f*       __restrict__ world,
    int n,
    math::Vec3f pos, math::Vec3f ex, math::Vec3f ey, math::Vec3f ez)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const math::Vec3f v = rest[i];
    world[i] = math::Vec3f(
        pos.x + ex.x * v.x + ey.x * v.y + ez.x * v.z,
        pos.y + ex.y * v.x + ey.y * v.y + ez.y * v.z,
        pos.z + ex.z * v.x + ey.z * v.y + ez.z * v.z);
}

// =====================================================================
// Compute per-triangle AABB + centroid (for BVH build/refit)
// =====================================================================

__global__ void compute_tri_aabbs_kernel(
    const math::Vec3f* __restrict__ verts,
    const math::Vec3i* __restrict__ tris,
    int n_tris,
    float thickness,
    Aabb*         __restrict__ aabbs,
    math::Vec3f*  __restrict__ centers)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tris) return;
    const math::Vec3i t = tris[i];
    const math::Vec3f a = verts[t.x];
    const math::Vec3f b = verts[t.y];
    const math::Vec3f c = verts[t.z];
    Aabb box;
    box.add(a);
    box.add(b);
    box.add(c);
    box.enlarge(thickness);
    aabbs[i] = box;
    centers[i] = math::Vec3f(
        (a.x + b.x + c.x) * (1.0f / 3.0f),
        (a.y + b.y + c.y) * (1.0f / 3.0f),
        (a.z + b.z + c.z) * (1.0f / 3.0f));
}

// =====================================================================
// Point-triangle closest point (returns barycentric coords + distance²)
// =====================================================================

__device__ __forceinline__ float point_triangle_dist2(
    const math::Vec3f& p,
    const math::Vec3f& a,
    const math::Vec3f& b,
    const math::Vec3f& c,
    math::Vec3f& closest,
    math::Vec3f& out_normal)
{
    const math::Vec3f ab(b.x - a.x, b.y - a.y, b.z - a.z);
    const math::Vec3f ac(c.x - a.x, c.y - a.y, c.z - a.z);
    const math::Vec3f ap(p.x - a.x, p.y - a.y, p.z - a.z);

    const float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
    const float d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - a.x, p.y - a.y, p.z - a.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const math::Vec3f bp(p.x - b.x, p.y - b.y, p.z - b.z);
    const float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
    const float d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - b.x, p.y - b.y, p.z - b.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        closest = math::Vec3f(a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z);
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - closest.x, p.y - closest.y, p.z - closest.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const math::Vec3f cp(p.x - c.x, p.y - c.y, p.z - c.z);
    const float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
    const float d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - c.x, p.y - c.y, p.z - c.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        closest = math::Vec3f(a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z);
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - closest.x, p.y - closest.y, p.z - closest.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = math::Vec3f(b.x + w * (c.x - b.x),
                               b.y + w * (c.y - b.y),
                               b.z + w * (c.z - b.z));
        float nx = ab.y * ac.z - ab.z * ac.y;
        float ny = ab.z * ac.x - ab.x * ac.z;
        float nz = ab.x * ac.y - ab.y * ac.x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
        out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
        math::Vec3f d(p.x - closest.x, p.y - closest.y, p.z - closest.z);
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    closest = math::Vec3f(a.x + ab.x * v + ac.x * w,
                           a.y + ab.y * v + ac.y * w,
                           a.z + ab.z * v + ac.z * w);
    float nx = ab.y * ac.z - ab.z * ac.y;
    float ny = ab.z * ac.x - ab.x * ac.z;
    float nz = ab.x * ac.y - ab.y * ac.x;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    float inv = (len > 1e-30f) ? 1.0f / len : 0.0f;
    out_normal = math::Vec3f(nx * inv, ny * inv, nz * inv);
    math::Vec3f d(p.x - closest.x, p.y - closest.y, p.z - closest.z);
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

// =====================================================================
// QuantBvh stackless point query: find closest triangle per particle
// =====================================================================

__global__ void detect_mesh_kernel(
    const float3* __restrict__       positions,
    int                              n_particles,
    const Ull2*   __restrict__       nodes,
    const PackedFace* __restrict__   ext_face,
    const Aabb*  __restrict__        scene_box,
    int                              int_size,
    const math::Vec3f* __restrict__  mesh_verts,
    float                            thickness,
    float                            search_radius,
    math::Vec4f* __restrict__        contacts)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const float3 pf = positions[p];
    const math::Vec3f query(pf.x, pf.y, pf.z);

    if (int_size <= 0) {
        contacts[p] = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const Aabb sb = scene_box[0];
    const float bucket = static_cast<float>((1 << QuantBvh::aabb_bits) - 2);
    const float dx = (sb.mx.x - sb.mn.x) / bucket;
    const float dy = (sb.mx.y - sb.mn.y) / bucket;
    const float dz = (sb.mx.z - sb.mn.z) / bucket;
    const float idx_q = 1.0f / fmaxf(dx, 1e-30f);
    const float idy_q = 1.0f / fmaxf(dy, 1e-30f);
    const float idz_q = 1.0f / fmaxf(dz, 1e-30f);

    // Use search_radius (>> thickness) so deeply penetrated particles
    // still find their closest triangle.
    const float r = search_radius;
    int qmn_x = (int)((query.x - r - sb.mn.x) * idx_q);
    int qmn_y = (int)((query.y - r - sb.mn.y) * idy_q);
    int qmn_z = (int)((query.z - r - sb.mn.z) * idz_q);
    int qmx_x = (int)ceilf((query.x + r - sb.mn.x) * idx_q);
    int qmx_y = (int)ceilf((query.y + r - sb.mn.y) * idy_q);
    int qmx_z = (int)ceilf((query.z + r - sb.mn.z) * idz_q);

    float best_dist2 = search_radius * search_radius;
    math::Vec3f best_closest(0.0f, 0.0f, 0.0f);
    math::Vec3f best_face_normal(0.0f, 0.0f, 1.0f);
    bool found = false;

    std::uint32_t st = 0u;
    while (st != QuantBvh::max_index) {
        Ull2 node = nodes[st];
        const std::uint32_t lc     = (std::uint32_t)(node.x >> QuantBvh::offset3);
        const std::uint32_t escape = (std::uint32_t)(node.y >> QuantBvh::offset3);

        constexpr std::uint64_t MASK = QuantBvh::aabb_mask;
        int v;
        bool hit = true;
        v = (int)((node.x >> QuantBvh::offset2) & MASK); if (v > qmx_x) hit = false;
        if (hit) { v = (int)((node.y >> QuantBvh::offset2) & MASK); if (v < qmn_x) hit = false; }
        if (hit) { v = (int)((node.x >> QuantBvh::offset1) & MASK); if (v > qmx_y) hit = false; }
        if (hit) { v = (int)((node.y >> QuantBvh::offset1) & MASK); if (v < qmn_y) hit = false; }
        if (hit) { v = (int)(node.x & MASK); if (v > qmx_z) hit = false; }
        if (hit) { v = (int)(node.y & MASK); if (v < qmn_z) hit = false; }

        if (hit) {
            if (lc == QuantBvh::max_index) {
                const int leaf_slot = static_cast<int>(st) - int_size;
                const PackedFace fd = ext_face[leaf_slot];
                const math::Vec3f a = mesh_verts[fd.x];
                const math::Vec3f b = mesh_verts[fd.y];
                const math::Vec3f c = mesh_verts[fd.z];

                math::Vec3f closest, face_n;
                float d2 = point_triangle_dist2(query, a, b, c, closest, face_n);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_closest = closest;
                    best_face_normal = face_n;
                    found = true;
                    float nr = sqrtf(d2);
                    qmn_x = (int)((query.x - nr - sb.mn.x) * idx_q);
                    qmn_y = (int)((query.y - nr - sb.mn.y) * idy_q);
                    qmn_z = (int)((query.z - nr - sb.mn.z) * idz_q);
                    qmx_x = (int)ceilf((query.x + nr - sb.mn.x) * idx_q);
                    qmx_y = (int)ceilf((query.y + nr - sb.mn.y) * idy_q);
                    qmx_z = (int)ceilf((query.z + nr - sb.mn.z) * idz_q);
                }
                st = escape;
            } else {
                st = lc;
            }
        } else {
            st = escape;
        }
    }

    if (!found) {
        contacts[p] = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float dist = sqrtf(best_dist2);
    math::Vec3f diff(query.x - best_closest.x,
                     query.y - best_closest.y,
                     query.z - best_closest.z);
    float dot_sign = diff.x * best_face_normal.x +
                     diff.y * best_face_normal.y +
                     diff.z * best_face_normal.z;

    float sd = (dot_sign >= 0.0f) ? dist : -dist;
    float depth = thickness - sd;

    math::Vec3f n(0.0f, 0.0f, 0.0f);
    if (depth > 0.0f && dist > 1e-20f) {
        float inv_dist = 1.0f / dist;
        n.x = diff.x * inv_dist;
        n.y = diff.y * inv_dist;
        n.z = diff.z * inv_dist;
        if (dot_sign < 0.0f) {
            n.x = -n.x; n.y = -n.y; n.z = -n.z;
        }
    } else {
        depth = 0.0f;
    }

    contacts[p] = math::Vec4f(n.x, n.y, n.z, depth);
}

// =====================================================================
// Gradient / Hessian / friction kernels — identical to sdf_contact.cu.
// They only depend on the contacts_[] cache, not on how it was filled.
// =====================================================================

// IPC gradient
__global__ void scatter_gradient_ipc_kernel(
    const math::Vec4f* __restrict__ contacts, int n,
    float k, float mu, float eps, float dt,
    const float3* __restrict__ vel,
    const math::Vec3f* __restrict__ bv_dev,
    float kd,
    math::Vec3f* __restrict__ rhs)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const math::Vec4f c = contacts[p];
    if (c.w <= 0.0f) return;
    const float nx = c.x, ny = c.y, nz = c.z;
    const float f_n = k * c.w;
    rhs[p].x += -f_n * nx;
    rhs[p].y += -f_n * ny;
    rhs[p].z += -f_n * nz;
    const float3 vp = vel[p];
    const math::Vec3f vb = bv_dev[0];
    const float rx = (vp.x - vb.x) * dt, ry = (vp.y - vb.y) * dt, rz = (vp.z - vb.z) * dt;
    const float rdn = rx * nx + ry * ny + rz * nz;
    if (kd > 0.0f && rdn < 0.0f) {
        const float d = kd * k;
        rhs[p].x -= d * rdn * nx; rhs[p].y -= d * rdn * ny; rhs[p].z -= d * rdn * nz;
    }
    const float utx = rx - rdn * nx, uty = ry - rdn * ny, utz = rz - rdn * nz;
    const float un = sqrtf(utx * utx + uty * uty + utz * utz);
    if (mu > 0.0f && f_n > 0.0f && un > 0.0f) {
        const float eu = eps * dt;
        float f1 = (un > eu) ? 1.0f / un : (-un / eu + 2.0f) / eu;
        float s = mu * f_n * f1;
        rhs[p].x -= s * utx; rhs[p].y -= s * uty; rhs[p].z -= s * utz;
    }
}

// IPC Hessian diagonal
__global__ void bake_diag_ipc_kernel(
    const math::Vec4f* __restrict__ contacts, int n,
    float k, float mu, float eps, float dt,
    const float3* __restrict__ vel,
    const math::Vec3f* __restrict__ bv_dev,
    float kd,
    math::Mat3f* __restrict__ diag)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const math::Vec4f c = contacts[p];
    if (c.w <= 0.0f) return;
    const float nx = c.x, ny = c.y, nz = c.z;
    math::Mat3f& A = diag[p];
    A.data[0] += k*nx*nx; A.data[1] += k*nx*ny; A.data[2] += k*nx*nz;
    A.data[3] += k*nx*ny; A.data[4] += k*ny*ny; A.data[5] += k*ny*nz;
    A.data[6] += k*nx*nz; A.data[7] += k*ny*nz; A.data[8] += k*nz*nz;

    const float3 vp = vel[p];
    const math::Vec3f vb = bv_dev[0];
    const float rx = (vp.x-vb.x)*dt, ry = (vp.y-vb.y)*dt, rz = (vp.z-vb.z)*dt;
    const float rdn = rx*nx + ry*ny + rz*nz;
    if (kd > 0.0f && rdn < 0.0f) {
        const float dc = kd * k / fmaxf(dt, 1e-30f);
        A.data[0]+=dc*nx*nx; A.data[1]+=dc*nx*ny; A.data[2]+=dc*nx*nz;
        A.data[3]+=dc*nx*ny; A.data[4]+=dc*ny*ny; A.data[5]+=dc*ny*nz;
        A.data[6]+=dc*nx*nz; A.data[7]+=dc*ny*nz; A.data[8]+=dc*nz*nz;
    }
    const float f_n = k * c.w;
    if (mu > 0.0f && f_n > 0.0f) {
        const float utx = rx-rdn*nx, uty = ry-rdn*ny, utz = rz-rdn*nz;
        const float un = sqrtf(utx*utx + uty*uty + utz*utz);
        if (un > 0.0f) {
            const float eu = eps * dt;
            float f1 = (un > eu) ? 1.0f / un : (-un/eu + 2.0f) / eu;
            float s = mu * f_n * f1;
            A.data[0]+=s*(1.0f-nx*nx); A.data[1]+=s*(0.0f-nx*ny); A.data[2]+=s*(0.0f-nx*nz);
            A.data[3]+=s*(0.0f-nx*ny); A.data[4]+=s*(1.0f-ny*ny); A.data[5]+=s*(0.0f-ny*nz);
            A.data[6]+=s*(0.0f-nx*nz); A.data[7]+=s*(0.0f-ny*nz); A.data[8]+=s*(1.0f-nz*nz);
        }
    }
}

// Penalty-only gradient
__global__ void scatter_gradient_kernel(
    const math::Vec4f* __restrict__ contacts, int n,
    float k, math::Vec3f* __restrict__ rhs)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const math::Vec4f c = contacts[p];
    if (c.w <= 0.0f) return;
    const float s = -k * c.w;
    rhs[p].x += s * c.x; rhs[p].y += s * c.y; rhs[p].z += s * c.z;
}

// Penalty-only Hessian diagonal
__global__ void bake_diag_kernel(
    const math::Vec4f* __restrict__ contacts, int n,
    float k, math::Mat3f* __restrict__ diag)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const math::Vec4f c = contacts[p];
    if (c.w <= 0.0f) return;
    const float nx = c.x, ny = c.y, nz = c.z;
    math::Mat3f& A = diag[p];
    A.data[0]+=k*nx*nx; A.data[1]+=k*nx*ny; A.data[2]+=k*nx*nz;
    A.data[3]+=k*nx*ny; A.data[4]+=k*ny*ny; A.data[5]+=k*ny*nz;
    A.data[6]+=k*nx*nz; A.data[7]+=k*ny*nz; A.data[8]+=k*nz*nz;
}

// Coulomb-cone post-projection
__global__ void apply_coulomb_friction_kernel(
    const math::Vec4f* __restrict__ contacts, int n,
    float k, float mu, float thickness,
    const math::Vec3f* __restrict__ bv_dev,
    const float* __restrict__ mass,
    const float3* __restrict__ vel, float dt,
    math::Vec3f* __restrict__ rhs)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const math::Vec4f c = contacts[p];
    if (c.w <= 0.0f || mu <= 0.0f) return;
    const float nx = c.x, ny = c.y, nz = c.z;
    const float fn_push = k * c.w;
    const math::Vec3f r = rhs[p];
    const float F0x = r.x - fn_push*nx, F0y = r.y - fn_push*ny, F0z = r.z - fn_push*nz;
    const float F0n = F0x*nx + F0y*ny + F0z*nz;
    const float cone = mu * k * fmaxf(c.w, 0.1f * thickness);
    const float ftx = F0x - F0n*nx, fty = F0y - F0n*ny, ftz = F0z - F0n*nz;
    const float ft2 = ftx*ftx + fty*fty + ftz*ftz;
    if (ft2 <= 0.0f) return;
    const float ft = sqrtf(ft2);
    if (ft <= cone) {
        const math::Vec3f vb = bv_dev[0];
        const float3 vp = vel[p];
        const float vbn = vb.x*nx + vb.y*ny + vb.z*nz;
        const float vpn = vp.x*nx + vp.y*ny + vp.z*nz;
        const float dvx = (vb.x - vbn*nx) - (vp.x - vpn*nx);
        const float dvy = (vb.y - vbn*ny) - (vp.y - vpn*ny);
        const float dvz = (vb.z - vbn*nz) - (vp.z - vpn*nz);
        const float m = mass[p];
        const float m_dt = m / fmaxf(dt, 1e-30f);
        rhs[p].x += m_dt * dvx; rhs[p].y += m_dt * dvy; rhs[p].z += m_dt * dvz;
    } else {
        const float s = cone / ft;
        rhs[p].x -= s * ftx; rhs[p].y -= s * fty; rhs[p].z -= s * ftz;
    }
}

}  // namespace

// =====================================================================
// MeshContact implementation
// =====================================================================

void MeshContact::set_mesh(const math::Vec3f* vertices, int n_vertices,
                           const int* indices, int n_triangles) {
    n_vertices_  = n_vertices;
    n_triangles_ = n_triangles;

    rest_vertices_.resize(static_cast<std::size_t>(n_vertices));
    std::memcpy(rest_vertices_.cpu_data(), vertices,
                sizeof(math::Vec3f) * n_vertices);
    rest_vertices_.copy_to_device();

    triangles_.resize(static_cast<std::size_t>(n_triangles));
    std::memcpy(triangles_.cpu_data(),
                reinterpret_cast<const math::Vec3i*>(indices),
                sizeof(math::Vec3i) * n_triangles);
    triangles_.copy_to_device();

    world_vertices_.allocate_device(static_cast<std::size_t>(n_vertices));
    tri_aabbs_.allocate_device(static_cast<std::size_t>(n_triangles));
    tri_centers_.allocate_device(static_cast<std::size_t>(n_triangles));

    // QuantBvh sized for triangle count; max_query_pairs not used for
    // point queries but build() needs a positive value.
    bvh_.build(n_triangles, 1);
    bvh_built_ = false;
}

void MeshContact::set_pose(const math::Vec3f& pos,
                           const math::Vec3f& ex,
                           const math::Vec3f& ey,
                           const math::Vec3f& ez,
                           std::uintptr_t cuda_stream) {
    if (n_vertices_ <= 0 || n_triangles_ <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    transform_vertices_kernel<<<grid_for(n_vertices_), kBlockDim, 0, stream>>>(
        rest_vertices_.gpu_data(), world_vertices_.gpu_data(),
        n_vertices_, pos, ex, ey, ez);
    check_cuda(cudaGetLastError(), "transform_vertices_kernel");

    compute_tri_aabbs_kernel<<<grid_for(n_triangles_), kBlockDim, 0, stream>>>(
        world_vertices_.gpu_data(), triangles_.gpu_data(),
        n_triangles_, thickness_,
        tri_aabbs_.gpu_data(), tri_centers_.gpu_data());
    check_cuda(cudaGetLastError(), "compute_tri_aabbs_kernel");

    // QuantBvh::refit needs the face index array for PackedFace packing.
    bvh_.refit(tri_aabbs_.gpu_data(), tri_centers_.gpu_data(),
               triangles_.gpu_data(), cuda_stream);
    bvh_built_ = true;
}

void MeshContact::set_body_velocity(const math::Vec3f& v,
                                    std::uintptr_t cuda_stream) {
    body_velocity_ = v;
    if (body_velocity_dev_.gpu_size() != 1u) {
        body_velocity_dev_.resize(1u);
    }
    body_velocity_dev_[0] = v;
    body_velocity_dev_.copy_to_device(cuda_stream);
}

void MeshContact::detect(const math::Vec3f* positions,
                         int n_particles,
                         std::uintptr_t cuda_stream,
                         const math::Vec3f* velocities,
                         float dt) {
    if (!active() || n_particles <= 0) return;
    if (!positions) {
        throw std::invalid_argument(
            "chysx::collision::MeshContact::detect: positions must be non-null");
    }

    if (cached_n_particles_ != n_particles) {
        contacts_.allocate_device(static_cast<std::size_t>(n_particles));
        cached_n_particles_ = n_particles;
    }
    cached_dt_ = dt;
    cached_velocities_ = velocities;

    if (body_velocity_dev_.gpu_size() != 1u) {
        body_velocity_dev_.resize(1u);
        body_velocity_dev_[0] = body_velocity_;
        body_velocity_dev_.copy_to_device(cuda_stream);
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    const float sr = (search_radius_ > 0.0f) ? search_radius_
                                               : 10.0f * thickness_;
    detect_mesh_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        reinterpret_cast<const float3*>(positions),
        n_particles,
        bvh_.nodes_dev(),
        bvh_.ext_face_dev(),
        bvh_.scene_bbox_dev(),
        bvh_.int_size(),
        world_vertices_.gpu_data(),
        thickness_,
        sr,
        contacts_.gpu_data());
    check_cuda(cudaGetLastError(), "detect_mesh_kernel launch");
}

void MeshContact::accumulate_gradient(math::Vec3f* rhs, int n,
                                      std::uintptr_t cuda_stream,
                                      const math::Vec3f*, const math::Vec3f*,
                                      float dt) const {
    if (!active() || n <= 0 || cached_n_particles_ != n) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (ipc_friction_ && cached_velocities_ && dt > 0.0f) {
        scatter_gradient_ipc_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(), n, stiffness_, friction_,
            friction_epsilon_, dt,
            reinterpret_cast<const float3*>(cached_velocities_),
            body_velocity_dev_.gpu_data(), contact_kd_, rhs);
    } else {
        scatter_gradient_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(), n, stiffness_, rhs);
    }
    check_cuda(cudaGetLastError(), "MeshContact::accumulate_gradient");
}

void MeshContact::bake_diag(math::Mat3f* diag, int n, float dt,
                            std::uintptr_t cuda_stream,
                            const math::Vec3f*, const math::Vec3f*) const {
    if (!active() || n <= 0 || cached_n_particles_ != n) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (ipc_friction_ && cached_velocities_ && dt > 0.0f) {
        bake_diag_ipc_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(), n, stiffness_, friction_,
            friction_epsilon_, dt,
            reinterpret_cast<const float3*>(cached_velocities_),
            body_velocity_dev_.gpu_data(), contact_kd_, diag);
    } else {
        bake_diag_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
            contacts_.gpu_data(), n, stiffness_, diag);
    }
    check_cuda(cudaGetLastError(), "MeshContact::bake_diag");
}

void MeshContact::apply_coulomb_friction(math::Vec3f* rhs, int n,
                                         const float* mass,
                                         const math::Mat3f*,
                                         const math::Vec3f&,
                                         float,
                                         std::uintptr_t cuda_stream) const {
    if (!active() || n <= 0 || cached_n_particles_ != n) return;
    if (friction_ <= 0.0f || !cached_velocities_) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    apply_coulomb_friction_kernel<<<grid_for(n), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(), n, stiffness_, friction_, thickness_,
        body_velocity_dev_.gpu_data(), mass,
        reinterpret_cast<const float3*>(cached_velocities_),
        cached_dt_, rhs);
    check_cuda(cudaGetLastError(), "MeshContact::apply_coulomb_friction");
}

}  // namespace collision
}  // namespace chysx
