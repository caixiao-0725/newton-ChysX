// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU AVBD solver kernels: initialize bodies, colored Gauss-Seidel primal,
// parallel dual update, BDF1 velocity update.

#include "avbd_gpu_solver.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace chysx {
namespace avbd {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GpuSolver: ") + w +
                                 ": " + cudaGetErrorString(e));
}

// -----------------------------------------------------------------------
// Device math (self-contained, mirrors avbd_maths.h for GPU)
// -----------------------------------------------------------------------

struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };
struct M33 { V3 r[3]; };

__device__ V3 v3(float a, float b, float c) { return {a, b, c}; }
__device__ V3 operator+(V3 a, V3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ V3 operator-(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ V3 operator*(V3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ V3 operator/(V3 a, float s) { return {a.x/s, a.y/s, a.z/s}; }
__device__ V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
__device__ float dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ V3 cross(V3 a, V3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__device__ Q4 qconj(Q4 q) { return {-q.x, -q.y, -q.z, q.w}; }
__device__ float qlensq(Q4 q) { return q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w; }
__device__ Q4 qscale(Q4 q, float s) { return {q.x*s, q.y*s, q.z*s, q.w*s}; }
__device__ Q4 qadd(Q4 a, Q4 b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
__device__ Q4 qmul(Q4 a, Q4 b) {
    return {
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
}
__device__ Q4 qinv(Q4 q) { float s = 1.0f / qlensq(q); return qscale(qconj(q), s); }
__device__ Q4 qnorm(Q4 q) { return qscale(q, rsqrtf(qlensq(q))); }

__device__ V3 qrot(Q4 q, V3 v) {
    V3 u = {q.x, q.y, q.z};
    V3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

// quat difference -> angular error (vec3), mirrors operator-(quat a, quat b)
__device__ V3 qdiff(Q4 a, Q4 b) {
    Q4 d = qmul(a, qinv(b));
    return v3(d.x, d.y, d.z) * 2.0f;
}

// quat + angular increment, mirrors operator+(quat a, float3 b)
__device__ Q4 qaddv(Q4 a, V3 b) {
    Q4 bq = {b.x, b.y, b.z, 0.0f};
    Q4 r = qadd(a, qscale(qmul(bq, a), 0.5f));
    return qnorm(r);
}

__device__ V3 transform(V3 pos, Q4 ang, V3 v) {
    return qrot(ang, v) + pos;
}

// 3x3 matrix ops
__device__ M33 m33diag(float a, float b, float c) {
    M33 m; m.r[0] = {a,0,0}; m.r[1] = {0,b,0}; m.r[2] = {0,0,c}; return m;
}
__device__ V3 m33mulv(M33 m, V3 v) {
    return {dot(m.r[0], v), dot(m.r[1], v), dot(m.r[2], v)};
}
__device__ M33 m33T(M33 m) {
    M33 t;
    t.r[0] = {m.r[0].x, m.r[1].x, m.r[2].x};
    t.r[1] = {m.r[0].y, m.r[1].y, m.r[2].y};
    t.r[2] = {m.r[0].z, m.r[1].z, m.r[2].z};
    return t;
}
__device__ M33 m33mul(M33 a, M33 b) {
    M33 bt = m33T(b);
    M33 r;
    for (int i = 0; i < 3; i++)
        r.r[i] = {dot(a.r[i], bt.r[0]), dot(a.r[i], bt.r[1]), dot(a.r[i], bt.r[2])};
    return r;
}
__device__ M33 m33add(M33 a, M33 b) {
    M33 r; for (int i = 0; i < 3; i++) r.r[i] = a.r[i] + b.r[i]; return r;
}
__device__ M33 m33neg(M33 m) { M33 r; for (int i = 0; i < 3; i++) r.r[i] = -m.r[i]; return r; }
__device__ M33 m33scale(M33 m, float s) { M33 r; for (int i = 0; i < 3; i++) r.r[i] = m.r[i] * s; return r; }

__device__ M33 m33from_basis(const float b[9]) {
    M33 m;
    m.r[0] = {b[0], b[1], b[2]};
    m.r[1] = {b[3], b[4], b[5]};
    m.r[2] = {b[6], b[7], b[8]};
    return m;
}

// 6x6 LDL^T solver (direct port of avbd_maths.h solve())
__device__ void solve6x6(M33 aLin, M33 aAng, M33 aCross,
                         V3 bLin, V3 bAng, V3& xLin, V3& xAng) {
    float A11 = aLin.r[0].x;
    float A21 = aLin.r[1].x, A22 = aLin.r[1].y;
    float A31 = aLin.r[2].x, A32 = aLin.r[2].y, A33 = aLin.r[2].z;
    float A41 = aCross.r[0].x, A42 = aCross.r[0].y, A43 = aCross.r[0].z, A44 = aAng.r[0].x;
    float A51 = aCross.r[1].x, A52 = aCross.r[1].y, A53 = aCross.r[1].z, A54 = aAng.r[1].x, A55 = aAng.r[1].y;
    float A61 = aCross.r[2].x, A62 = aCross.r[2].y, A63 = aCross.r[2].z, A64 = aAng.r[2].x, A65 = aAng.r[2].y, A66 = aAng.r[2].z;

    float L21 = A21/A11, L31 = A31/A11, L41 = A41/A11, L51 = A51/A11, L61 = A61/A11;
    float D1 = A11;
    float D2 = A22 - L21*L21*D1;
    float L32 = (A32 - L21*L31*D1)/D2;
    float L42 = (A42 - L21*L41*D1)/D2;
    float L52 = (A52 - L21*L51*D1)/D2;
    float L62 = (A62 - L21*L61*D1)/D2;
    float D3 = A33 - (L31*L31*D1 + L32*L32*D2);
    float L43 = (A43 - L31*L41*D1 - L32*L42*D2)/D3;
    float L53 = (A53 - L31*L51*D1 - L32*L52*D2)/D3;
    float L63 = (A63 - L31*L61*D1 - L32*L62*D2)/D3;
    float D4 = A44 - (L41*L41*D1 + L42*L42*D2 + L43*L43*D3);
    float L54 = (A54 - L41*L51*D1 - L42*L52*D2 - L43*L53*D3)/D4;
    float L64 = (A64 - L41*L61*D1 - L42*L62*D2 - L43*L63*D3)/D4;
    float D5 = A55 - (L51*L51*D1 + L52*L52*D2 + L53*L53*D3 + L54*L54*D4);
    float L65 = (A65 - L51*L61*D1 - L52*L62*D2 - L53*L63*D3 - L54*L64*D4)/D5;
    float D6 = A66 - (L61*L61*D1 + L62*L62*D2 + L63*L63*D3 + L64*L64*D4 + L65*L65*D5);

    float y1 = bLin.x;
    float y2 = bLin.y - L21*y1;
    float y3 = bLin.z - L31*y1 - L32*y2;
    float y4 = bAng.x - L41*y1 - L42*y2 - L43*y3;
    float y5 = bAng.y - L51*y1 - L52*y2 - L53*y3 - L54*y4;
    float y6 = bAng.z - L61*y1 - L62*y2 - L63*y3 - L64*y4 - L65*y5;

    float z1=y1/D1, z2=y2/D2, z3=y3/D3, z4=y4/D4, z5=y5/D5, z6=y6/D6;

    xAng.z = z6;
    xAng.y = z5 - L65*xAng.z;
    xAng.x = z4 - L54*xAng.y - L64*xAng.z;
    xLin.z = z3 - L43*xAng.x - L53*xAng.y - L63*xAng.z;
    xLin.y = z2 - L32*xLin.z - L42*xAng.x - L52*xAng.y - L62*xAng.z;
    xLin.x = z1 - L21*xLin.y - L31*xLin.z - L41*xAng.x - L51*xAng.y - L61*xAng.z;
}

// -----------------------------------------------------------------------
// Kernel 1: Initialize bodies (replaces solver.cpp:428-447)
// -----------------------------------------------------------------------
__global__ void initialize_bodies_kernel(
    float* px, float* py, float* pz,
    float* qx, float* qy, float* qz, float* qw,
    const float* vx, const float* vy, const float* vz,
    const float* vax, const float* vay, const float* vaz,
    const float* pvx, const float* pvy, const float* pvz,
    const float* mass,
    float* ix, float* iy, float* iz,        // initialLin
    float* iqx, float* iqy, float* iqz, float* iqw,  // initialAng
    float* inx, float* iny, float* inz,     // inertialLin
    float* inqx, float* inqy, float* inqz, float* inqw, // inertialAng
    int n, float dt, float gravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    V3 posLin = {px[i], py[i], pz[i]};
    Q4 posAng = {qx[i], qy[i], qz[i], qw[i]};
    V3 velLin = {vx[i], vy[i], vz[i]};
    V3 velAng = {vax[i], vay[i], vaz[i]};
    V3 prevVelLin = {pvx[i], pvy[i], pvz[i]};
    float m = mass[i];

    // inertialLin
    V3 inertLin = posLin + velLin * dt;
    if (m > 0.0f) inertLin = inertLin + v3(0, 0, gravity) * (dt * dt);
    // inertialAng
    Q4 inertAng = qaddv(posAng, velAng * dt);

    inx[i] = inertLin.x; iny[i] = inertLin.y; inz[i] = inertLin.z;
    inqx[i] = inertAng.x; inqy[i] = inertAng.y; inqz[i] = inertAng.z; inqw[i] = inertAng.w;

    // accelWeight
    V3 accel = (velLin - prevVelLin) / dt;
    float accelExt = accel.z * (gravity < 0 ? -1.0f : gravity > 0 ? 1.0f : 0.0f);
    float absGrav = fabsf(gravity);
    float accelWeight = 0.0f;
    if (absGrav > 0.0f) {
        accelWeight = accelExt / absGrav;
        accelWeight = fmaxf(0.0f, fminf(1.0f, accelWeight));
    }
    if (!isfinite(accelWeight)) accelWeight = 0.0f;

    // initialLin/Ang
    ix[i] = posLin.x; iy[i] = posLin.y; iz[i] = posLin.z;
    iqx[i] = posAng.x; iqy[i] = posAng.y; iqz[i] = posAng.z; iqw[i] = posAng.w;

    // predicted position
    if (m > 0.0f) {
        V3 newPos = posLin + velLin * dt + v3(0, 0, gravity) * (accelWeight * dt * dt);
        Q4 newAng = qaddv(posAng, velAng * dt);
        px[i] = newPos.x; py[i] = newPos.y; pz[i] = newPos.z;
        qx[i] = newAng.x; qy[i] = newAng.y; qz[i] = newAng.z; qw[i] = newAng.w;
    }
}

// -----------------------------------------------------------------------
// Kernel 2: Initialize contacts (compute C0, scale lambda/penalty)
// Replaces the Manifold::initialize() contact loop on CPU.
// -----------------------------------------------------------------------
__global__ void initialize_contacts_kernel(
    GpuManifold* manifolds, GpuContact* contacts, int n_manifolds,
    const float* px, const float* py, const float* pz,
    const float* qx, const float* qy, const float* qz, const float* qw,
    float alpha, float gamma_, float collision_margin)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_manifolds) return;

    const GpuManifold& m = manifolds[tid];
    int ba = m.body_a, bb = m.body_b;
    V3 posA = {px[ba], py[ba], pz[ba]};
    Q4 angA = {qx[ba], qy[ba], qz[ba], qw[ba]};
    V3 posB = {px[bb], py[bb], pz[bb]};
    Q4 angB = {qx[bb], qy[bb], qz[bb], qw[bb]};
    M33 basis = m33from_basis(m.basis);

    for (int c = 0; c < m.num_contacts; c++) {
        GpuContact& ct = contacts[m.contact_offset + c];
        V3 rA = {ct.rA_x, ct.rA_y, ct.rA_z};
        V3 rB = {ct.rB_x, ct.rB_y, ct.rB_z};
        V3 xA = transform(posA, angA, rA);
        V3 xB = transform(posB, angB, rB);
        V3 diff = xA - xB;
        V3 C0 = m33mulv(basis, diff);
        C0.x += collision_margin;
        ct.C0_x = C0.x; ct.C0_y = C0.y; ct.C0_z = C0.z;

        ct.lambda_x *= alpha * gamma_;
        ct.lambda_y *= alpha * gamma_;
        ct.lambda_z *= alpha * gamma_;

        float pen_min = 1.0f, pen_max = 10000000000.0f;
        ct.penalty_x = fminf(fmaxf(ct.penalty_x * gamma_, pen_min), pen_max);
        ct.penalty_y = fminf(fmaxf(ct.penalty_y * gamma_, pen_min), pen_max);
        ct.penalty_z = fminf(fmaxf(ct.penalty_z * gamma_, pen_min), pen_max);
    }
}

// -----------------------------------------------------------------------
// Kernel 3: Primal update (colored Gauss-Seidel)
// One thread per body; only processes bodies with colors[body] == color_id.
// -----------------------------------------------------------------------
__global__ void primal_update_kernel(
    int color_id,
    const int* __restrict__ colors,
    float* px, float* py, float* pz,
    float* qx_, float* qy_, float* qz_, float* qw_,
    const float* ix, const float* iy, const float* iz,
    const float* iqx, const float* iqy, const float* iqz, const float* iqw,
    const float* inx, const float* iny, const float* inz,
    const float* inqx, const float* inqy, const float* inqz, const float* inqw,
    const float* mass, const float* mom_x, const float* mom_y, const float* mom_z,
    const GpuManifold* __restrict__ manifolds,
    GpuContact* __restrict__ contacts,
    const int* __restrict__ vtx_counts,
    const VertexEntry* __restrict__ vtx_table, int vtx_stride,
    int n_bodies, float dt, float alpha)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_bodies) return;
    if (colors[body] != color_id) return;

    float m = mass[body];
    if (m <= 0.0f) return;

    float invDt2 = 1.0f / (dt * dt);

    // M / dt^2
    M33 MLin = m33diag(m, m, m);
    M33 MAng = m33diag(mom_x[body], mom_y[body], mom_z[body]);

    M33 lhsLin = m33scale(MLin, invDt2);
    M33 lhsAng = m33scale(MAng, invDt2);
    M33 lhsCross = m33diag(0, 0, 0);

    V3 posLin = {px[body], py[body], pz[body]};
    Q4 posAng = {qx_[body], qy_[body], qz_[body], qw_[body]};
    V3 inertLin = {inx[body], iny[body], inz[body]};
    Q4 inertAng = {inqx[body], inqy[body], inqz[body], inqw[body]};

    V3 rhsLin = m33mulv(m33scale(MLin, invDt2), posLin - inertLin);
    V3 rhsAng = m33mulv(m33scale(MAng, invDt2), qdiff(posAng, inertAng));

    // Iterate over manifold neighbors from vertex table
    int cnt = vtx_counts[body];
    if (cnt > vtx_stride) cnt = vtx_stride;

    for (int s = 0; s < cnt; s++) {
        const VertexEntry& ve = vtx_table[body * vtx_stride + s];
        int midx = ve.manifold_idx;

        const GpuManifold& mf = manifolds[midx];
        M33 basis = m33from_basis(mf.basis);
        float friction = mf.friction;

        bool isA = (mf.body_a == body);
        int ba = mf.body_a, bb = mf.body_b;

        V3 posALin = {px[ba], py[ba], pz[ba]};
        Q4 posAAng = {qx_[ba], qy_[ba], qz_[ba], qw_[ba]};
        V3 posBLin = {px[bb], py[bb], pz[bb]};
        Q4 posBAng = {qx_[bb], qy_[bb], qz_[bb], qw_[bb]};
        V3 initALin = {ix[ba], iy[ba], iz[ba]};
        Q4 initAAng = {iqx[ba], iqy[ba], iqz[ba], iqw[ba]};
        V3 initBLin = {ix[bb], iy[bb], iz[bb]};
        Q4 initBAng = {iqx[bb], iqy[bb], iqz[bb], iqw[bb]};

        V3 dqALin = posALin - initALin;
        V3 dqAAng = qdiff(posAAng, initAAng);
        V3 dqBLin = posBLin - initBLin;
        V3 dqBAng = qdiff(posBAng, initBAng);

        for (int ci = 0; ci < mf.num_contacts; ci++) {
            const GpuContact& ct = contacts[mf.contact_offset + ci];
            V3 rA = {ct.rA_x, ct.rA_y, ct.rA_z};
            V3 rB = {ct.rB_x, ct.rB_y, ct.rB_z};
            V3 rAWorld = qrot(posAAng, rA);
            V3 rBWorld = qrot(posBAng, rB);

            M33 jALin = basis;
            M33 jBLin = m33neg(basis);

            // jAAng[row] = cross(rAWorld, jALin[row])
            M33 jAAng, jBAng;
            for (int r = 0; r < 3; r++) {
                jAAng.r[r] = cross(rAWorld, jALin.r[r]);
                jBAng.r[r] = cross(rBWorld, jBLin.r[r]);
            }

            M33 K = m33diag(ct.penalty_x, ct.penalty_y, ct.penalty_z);
            V3 C0 = {ct.C0_x, ct.C0_y, ct.C0_z};
            V3 C = C0 * (1.0f - alpha)
                 + m33mulv(jALin, dqALin) + m33mulv(jBLin, dqBLin)
                 + m33mulv(jAAng, dqAAng) + m33mulv(jBAng, dqBAng);

            V3 lambda = {ct.lambda_x, ct.lambda_y, ct.lambda_z};
            V3 F = m33mulv(K, C) + lambda;
            F.x = fminf(F.x, 0.0f);

            float bounds = fabsf(F.x) * friction;
            float fScale = sqrtf(F.y * F.y + F.z * F.z);
            if (fScale > bounds && fScale > 0.0f) {
                float ratio = bounds / fScale;
                F.y *= ratio;
                F.z *= ratio;
            }

            M33 jLin = isA ? jALin : jBLin;
            M33 jAng = isA ? jAAng : jBAng;
            M33 jLinT = m33T(jLin);
            M33 jAngT = m33T(jAng);
            M33 jAngTk = m33mul(jAngT, K);

            lhsLin  = m33add(lhsLin,  m33mul(m33mul(jLinT, K), jLin));
            lhsAng  = m33add(lhsAng,  m33mul(jAngTk, jAng));
            lhsCross = m33add(lhsCross, m33mul(jAngTk, jLin));

            rhsLin = rhsLin + m33mulv(jLinT, F);
            rhsAng = rhsAng + m33mulv(jAngT, F);
        }
    }

    // Solve 6x6 system
    V3 dxLin, dxAng;
    solve6x6(lhsLin, lhsAng, lhsCross, v3(0,0,0)-rhsLin, v3(0,0,0)-rhsAng, dxLin, dxAng);

    // Update position
    V3 newPos = posLin + dxLin;
    Q4 newAng = qaddv(posAng, dxAng);
    px[body] = newPos.x; py[body] = newPos.y; pz[body] = newPos.z;
    qx_[body] = newAng.x; qy_[body] = newAng.y; qz_[body] = newAng.z; qw_[body] = newAng.w;
}

// -----------------------------------------------------------------------
// Kernel 4: Dual update (all manifolds in parallel)
// -----------------------------------------------------------------------
__global__ void dual_update_kernel(
    const GpuManifold* __restrict__ manifolds,
    GpuContact* __restrict__ contacts,
    int n_manifolds,
    const float* px, const float* py, const float* pz,
    const float* qx_, const float* qy_, const float* qz_, const float* qw_,
    const float* ix, const float* iy, const float* iz,
    const float* iqx, const float* iqy, const float* iqz, const float* iqw,
    float alpha, float beta_lin)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_manifolds) return;

    const GpuManifold& mf = manifolds[tid];
    int ba = mf.body_a, bb = mf.body_b;
    M33 basis = m33from_basis(mf.basis);
    float friction = mf.friction;

    V3 posALin = {px[ba], py[ba], pz[ba]};
    Q4 posAAng = {qx_[ba], qy_[ba], qz_[ba], qw_[ba]};
    V3 posBLin = {px[bb], py[bb], pz[bb]};
    Q4 posBAng = {qx_[bb], qy_[bb], qz_[bb], qw_[bb]};
    V3 initALin = {ix[ba], iy[ba], iz[ba]};
    Q4 initAAng = {iqx[ba], iqy[ba], iqz[ba], iqw[ba]};
    V3 initBLin = {ix[bb], iy[bb], iz[bb]};
    Q4 initBAng = {iqx[bb], iqy[bb], iqz[bb], iqw[bb]};

    V3 dqALin = posALin - initALin;
    V3 dqAAng = qdiff(posAAng, initAAng);
    V3 dqBLin = posBLin - initBLin;
    V3 dqBAng = qdiff(posBAng, initBAng);

    for (int ci = 0; ci < mf.num_contacts; ci++) {
        GpuContact& ct = contacts[mf.contact_offset + ci];
        V3 rA = {ct.rA_x, ct.rA_y, ct.rA_z};
        V3 rB = {ct.rB_x, ct.rB_y, ct.rB_z};
        V3 rAWorld = qrot(posAAng, rA);
        V3 rBWorld = qrot(posBAng, rB);

        M33 jALin = basis;
        M33 jBLin = m33neg(basis);
        M33 jAAng, jBAng;
        for (int r = 0; r < 3; r++) {
            jAAng.r[r] = cross(rAWorld, jALin.r[r]);
            jBAng.r[r] = cross(rBWorld, jBLin.r[r]);
        }

        M33 K = m33diag(ct.penalty_x, ct.penalty_y, ct.penalty_z);
        V3 C0 = {ct.C0_x, ct.C0_y, ct.C0_z};
        V3 C = C0 * (1.0f - alpha)
             + m33mulv(jALin, dqALin) + m33mulv(jBLin, dqBLin)
             + m33mulv(jAAng, dqAAng) + m33mulv(jBAng, dqBAng);

        V3 lambda = {ct.lambda_x, ct.lambda_y, ct.lambda_z};
        V3 F = m33mulv(K, C) + lambda;
        F.x = fminf(F.x, 0.0f);

        float bounds = fabsf(F.x) * friction;
        float fScale = sqrtf(F.y * F.y + F.z * F.z);
        if (fScale > bounds && fScale > 0.0f) {
            float ratio = bounds / fScale;
            F.y *= ratio;
            F.z *= ratio;
        }

        ct.lambda_x = F.x; ct.lambda_y = F.y; ct.lambda_z = F.z;

        float pen_max = 10000000000.0f;
        if (F.x < 0.0f)
            ct.penalty_x = fminf(ct.penalty_x + beta_lin * fabsf(C.x), pen_max);

        if (fScale <= bounds) {
            ct.penalty_y = fminf(ct.penalty_y + beta_lin * fabsf(C.y), pen_max);
            ct.penalty_z = fminf(ct.penalty_z + beta_lin * fabsf(C.z), pen_max);
            ct.stick = (sqrtf(C.y * C.y + C.z * C.z) < 0.00001f) ? 1 : 0;
        }
    }
}

// -----------------------------------------------------------------------
// Kernel 5: Velocity update (BDF1)
// -----------------------------------------------------------------------
__global__ void velocity_update_kernel(
    const float* px, const float* py, const float* pz,
    const float* qx_, const float* qy_, const float* qz_, const float* qw_,
    const float* ix, const float* iy, const float* iz,
    const float* iqx, const float* iqy, const float* iqz, const float* iqw,
    float* vx, float* vy, float* vz,
    float* vax, float* vay, float* vaz,
    float* pvx, float* pvy, float* pvz,
    const float* mass, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pvx[i] = vx[i]; pvy[i] = vy[i]; pvz[i] = vz[i];

    if (mass[i] > 0.0f) {
        float invDt = 1.0f / dt;
        vx[i] = (px[i] - ix[i]) * invDt;
        vy[i] = (py[i] - iy[i]) * invDt;
        vz[i] = (pz[i] - iz[i]) * invDt;

        Q4 posAng = {qx_[i], qy_[i], qz_[i], qw_[i]};
        Q4 initAng = {iqx[i], iqy[i], iqz[i], iqw[i]};
        V3 da = qdiff(posAng, initAng);
        vax[i] = da.x * invDt;
        vay[i] = da.y * invDt;
        vaz[i] = da.z * invDt;
    }
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Host-side implementation
// -----------------------------------------------------------------------

void GpuSolver::ensure_capacity(int n) {
    if (n <= capacity_) return;
    capacity_ = n;
    pos_x_.resize(n); pos_y_.resize(n); pos_z_.resize(n);
    quat_x_.resize(n); quat_y_.resize(n); quat_z_.resize(n); quat_w_.resize(n);
    vel_x_.resize(n); vel_y_.resize(n); vel_z_.resize(n);
    velang_x_.resize(n); velang_y_.resize(n); velang_z_.resize(n);
    prevvel_x_.resize(n); prevvel_y_.resize(n); prevvel_z_.resize(n);
    mass_.resize(n); moment_x_.resize(n); moment_y_.resize(n); moment_z_.resize(n);
    half_x_.resize(n); half_y_.resize(n); half_z_.resize(n);
    friction_.resize(n);
    initial_x_.resize(n); initial_y_.resize(n); initial_z_.resize(n);
    initial_qx_.resize(n); initial_qy_.resize(n); initial_qz_.resize(n); initial_qw_.resize(n);
    inertial_x_.resize(n); inertial_y_.resize(n); inertial_z_.resize(n);
    inertial_qx_.resize(n); inertial_qy_.resize(n); inertial_qz_.resize(n); inertial_qw_.resize(n);
}

#define UPLOAD_ARRAY(dst, src, n) \
    memcpy(dst.cpu_data(), src, (n) * sizeof(float)); \
    dst.copy_to_device()

void GpuSolver::upload_bodies(
    const float* px, const float* py, const float* pz,
    const float* qx, const float* qy, const float* qz, const float* qw,
    const float* vx, const float* vy, const float* vz,
    const float* vax, const float* vay, const float* vaz,
    const float* pvx, const float* pvy, const float* pvz,
    const float* mass, const float* mom_x, const float* mom_y, const float* mom_z,
    const float* hx, const float* hy, const float* hz,
    const float* fric,
    int n)
{
    ensure_capacity(n + 1);
    n_bodies_ = n;
    UPLOAD_ARRAY(pos_x_, px, n); UPLOAD_ARRAY(pos_y_, py, n); UPLOAD_ARRAY(pos_z_, pz, n);
    UPLOAD_ARRAY(quat_x_, qx, n); UPLOAD_ARRAY(quat_y_, qy, n);
    UPLOAD_ARRAY(quat_z_, qz, n); UPLOAD_ARRAY(quat_w_, qw, n);
    UPLOAD_ARRAY(vel_x_, vx, n); UPLOAD_ARRAY(vel_y_, vy, n); UPLOAD_ARRAY(vel_z_, vz, n);
    UPLOAD_ARRAY(velang_x_, vax, n); UPLOAD_ARRAY(velang_y_, vay, n); UPLOAD_ARRAY(velang_z_, vaz, n);
    UPLOAD_ARRAY(prevvel_x_, pvx, n); UPLOAD_ARRAY(prevvel_y_, pvy, n); UPLOAD_ARRAY(prevvel_z_, pvz, n);
    UPLOAD_ARRAY(mass_, mass, n);
    UPLOAD_ARRAY(moment_x_, mom_x, n); UPLOAD_ARRAY(moment_y_, mom_y, n); UPLOAD_ARRAY(moment_z_, mom_z, n);
    UPLOAD_ARRAY(half_x_, hx, n); UPLOAD_ARRAY(half_y_, hy, n); UPLOAD_ARRAY(half_z_, hz, n);
    UPLOAD_ARRAY(friction_, fric, n);
}
#undef UPLOAD_ARRAY

void GpuSolver::upload_bodies_hybrid(
    const float* px_d, const float* py_d, const float* pz_d,
    const float* qx_d, const float* qy_d, const float* qz_d, const float* qw_d,
    const float* hx_d, const float* hy_d, const float* hz_d,
    const float* mass_d, const float* fric_d,
    const float* vx, const float* vy, const float* vz,
    const float* vax, const float* vay, const float* vaz,
    const float* pvx, const float* pvy, const float* pvz,
    const float* mom_x, const float* mom_y, const float* mom_z,
    int n)
{
    ensure_capacity(n);
    n_bodies_ = n;
    size_t fb = n * sizeof(float);

    // D2D copies for data already on GPU
    #define D2D(dst, src) check(cudaMemcpy(dst.gpu_data(), src, fb, cudaMemcpyDeviceToDevice), #dst " D2D")
    D2D(pos_x_, px_d);  D2D(pos_y_, py_d);  D2D(pos_z_, pz_d);
    D2D(quat_x_, qx_d); D2D(quat_y_, qy_d); D2D(quat_z_, qz_d); D2D(quat_w_, qw_d);
    D2D(half_x_, hx_d); D2D(half_y_, hy_d); D2D(half_z_, hz_d);
    D2D(mass_, mass_d);
    D2D(friction_, fric_d);
    #undef D2D

    // H2D for data only available on CPU
    #define UPLOAD_ARRAY(dst, src, n) \
        memcpy(dst.cpu_data(), src, (n) * sizeof(float)); \
        dst.copy_to_device()
    UPLOAD_ARRAY(vel_x_, vx, n);    UPLOAD_ARRAY(vel_y_, vy, n);    UPLOAD_ARRAY(vel_z_, vz, n);
    UPLOAD_ARRAY(velang_x_, vax, n); UPLOAD_ARRAY(velang_y_, vay, n); UPLOAD_ARRAY(velang_z_, vaz, n);
    UPLOAD_ARRAY(prevvel_x_, pvx, n); UPLOAD_ARRAY(prevvel_y_, pvy, n); UPLOAD_ARRAY(prevvel_z_, pvz, n);
    UPLOAD_ARRAY(moment_x_, mom_x, n); UPLOAD_ARRAY(moment_y_, mom_y, n); UPLOAD_ARRAY(moment_z_, mom_z, n);
    #undef UPLOAD_ARRAY
}

void GpuSolver::setup_ground_body(float ground_z, float ground_friction) {
    int idx = n_bodies_;
    ensure_capacity(idx + 1);

    auto setOne = [&](CudaArray<float>& arr, float val) {
        float v = val;
        check(cudaMemcpy(arr.gpu_data() + idx, &v, sizeof(float),
                         cudaMemcpyHostToDevice), "ground body slot");
    };

    setOne(pos_x_, 0.0f);  setOne(pos_y_, 0.0f);  setOne(pos_z_, ground_z);
    setOne(quat_x_, 0.0f); setOne(quat_y_, 0.0f);
    setOne(quat_z_, 0.0f); setOne(quat_w_, 1.0f);
    setOne(vel_x_, 0.0f);  setOne(vel_y_, 0.0f);  setOne(vel_z_, 0.0f);
    setOne(velang_x_, 0.0f); setOne(velang_y_, 0.0f); setOne(velang_z_, 0.0f);
    setOne(prevvel_x_, 0.0f); setOne(prevvel_y_, 0.0f); setOne(prevvel_z_, 0.0f);
    setOne(mass_, 0.0f);
    setOne(moment_x_, 1.0f); setOne(moment_y_, 1.0f); setOne(moment_z_, 1.0f);
    setOne(half_x_, 1000.0f); setOne(half_y_, 1000.0f); setOne(half_z_, 0.0f);
    setOne(friction_, ground_friction);
    setOne(initial_x_, 0.0f);  setOne(initial_y_, 0.0f);  setOne(initial_z_, ground_z);
    setOne(initial_qx_, 0.0f); setOne(initial_qy_, 0.0f);
    setOne(initial_qz_, 0.0f); setOne(initial_qw_, 1.0f);
    setOne(inertial_x_, 0.0f);  setOne(inertial_y_, 0.0f);  setOne(inertial_z_, ground_z);
    setOne(inertial_qx_, 0.0f); setOne(inertial_qy_, 0.0f);
    setOne(inertial_qz_, 0.0f); setOne(inertial_qw_, 1.0f);
}

void GpuSolver::solve(
    GpuManifold* manifolds_dev, GpuContact* contacts_dev,
    int n_manifolds,
    const int* vtx_counts_dev, const VertexEntry* vtx_table_dev, int vtx_stride,
    const int* colors_dev, int num_colors,
    int iterations, float dt, float gravity,
    float alpha, float beta_lin, float gamma_)
{
    int n = n_bodies_;
    if (n <= 0) return;

    // Kernel 1: initialize bodies
    initialize_bodies_kernel<<<grid(n), kBlock>>>(
        pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
        quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
        vel_x_.gpu_data(), vel_y_.gpu_data(), vel_z_.gpu_data(),
        velang_x_.gpu_data(), velang_y_.gpu_data(), velang_z_.gpu_data(),
        prevvel_x_.gpu_data(), prevvel_y_.gpu_data(), prevvel_z_.gpu_data(),
        mass_.gpu_data(),
        initial_x_.gpu_data(), initial_y_.gpu_data(), initial_z_.gpu_data(),
        initial_qx_.gpu_data(), initial_qy_.gpu_data(), initial_qz_.gpu_data(), initial_qw_.gpu_data(),
        inertial_x_.gpu_data(), inertial_y_.gpu_data(), inertial_z_.gpu_data(),
        inertial_qx_.gpu_data(), inertial_qy_.gpu_data(), inertial_qz_.gpu_data(), inertial_qw_.gpu_data(),
        n, dt, gravity);
    check(cudaGetLastError(), "initialize_bodies_kernel");

    // Kernel 2: initialize contacts (C0, lambda/penalty scaling)
    if (n_manifolds > 0) {
        initialize_contacts_kernel<<<grid(n_manifolds), kBlock>>>(
            manifolds_dev, contacts_dev, n_manifolds,
            pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
            quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
            alpha, gamma_, 0.01f);
        check(cudaGetLastError(), "initialize_contacts_kernel");
    }

    // Main solver iterations
    for (int it = 0; it < iterations; it++) {
        // Colored Gauss-Seidel primal update
        for (int c = 0; c < num_colors; c++) {
            primal_update_kernel<<<grid(n), kBlock>>>(
                c, colors_dev,
                pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
                quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
                initial_x_.gpu_data(), initial_y_.gpu_data(), initial_z_.gpu_data(),
                initial_qx_.gpu_data(), initial_qy_.gpu_data(), initial_qz_.gpu_data(), initial_qw_.gpu_data(),
                inertial_x_.gpu_data(), inertial_y_.gpu_data(), inertial_z_.gpu_data(),
                inertial_qx_.gpu_data(), inertial_qy_.gpu_data(), inertial_qz_.gpu_data(), inertial_qw_.gpu_data(),
                mass_.gpu_data(), moment_x_.gpu_data(), moment_y_.gpu_data(), moment_z_.gpu_data(),
                manifolds_dev, contacts_dev,
                vtx_counts_dev, vtx_table_dev, vtx_stride,
                n, dt, alpha);
        }

        // Dual update: all manifolds parallel
        if (n_manifolds > 0) {
            dual_update_kernel<<<grid(n_manifolds), kBlock>>>(
                manifolds_dev, contacts_dev, n_manifolds,
                pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
                quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
                initial_x_.gpu_data(), initial_y_.gpu_data(), initial_z_.gpu_data(),
                initial_qx_.gpu_data(), initial_qy_.gpu_data(), initial_qz_.gpu_data(), initial_qw_.gpu_data(),
                alpha, beta_lin);
        }

    }
    check(cudaGetLastError(), "solver iterations");

    // Kernel 5: velocity update
    velocity_update_kernel<<<grid(n), kBlock>>>(
        pos_x_.gpu_data(), pos_y_.gpu_data(), pos_z_.gpu_data(),
        quat_x_.gpu_data(), quat_y_.gpu_data(), quat_z_.gpu_data(), quat_w_.gpu_data(),
        initial_x_.gpu_data(), initial_y_.gpu_data(), initial_z_.gpu_data(),
        initial_qx_.gpu_data(), initial_qy_.gpu_data(), initial_qz_.gpu_data(), initial_qw_.gpu_data(),
        vel_x_.gpu_data(), vel_y_.gpu_data(), vel_z_.gpu_data(),
        velang_x_.gpu_data(), velang_y_.gpu_data(), velang_z_.gpu_data(),
        prevvel_x_.gpu_data(), prevvel_y_.gpu_data(), prevvel_z_.gpu_data(),
        mass_.gpu_data(), n, dt);
    check(cudaGetLastError(), "velocity_update_kernel");
}

#define DOWNLOAD_ARRAY(dst, src, n) \
    src.copy_to_host(); \
    memcpy(dst, src.cpu_data(), (n) * sizeof(float))

void GpuSolver::download_positions(
    float* px, float* py, float* pz,
    float* qx, float* qy, float* qz, float* qw,
    float* vx, float* vy, float* vz,
    float* vax, float* vay, float* vaz,
    int n)
{
    DOWNLOAD_ARRAY(px, pos_x_, n); DOWNLOAD_ARRAY(py, pos_y_, n); DOWNLOAD_ARRAY(pz, pos_z_, n);
    DOWNLOAD_ARRAY(qx, quat_x_, n); DOWNLOAD_ARRAY(qy, quat_y_, n);
    DOWNLOAD_ARRAY(qz, quat_z_, n); DOWNLOAD_ARRAY(qw, quat_w_, n);
    DOWNLOAD_ARRAY(vx, vel_x_, n); DOWNLOAD_ARRAY(vy, vel_y_, n); DOWNLOAD_ARRAY(vz, vel_z_, n);
    DOWNLOAD_ARRAY(vax, velang_x_, n); DOWNLOAD_ARRAY(vay, velang_y_, n); DOWNLOAD_ARRAY(vaz, velang_z_, n);
}
#undef DOWNLOAD_ARRAY

void GpuSolver::download_body_pose(int idx,
                                   float& px, float& py, float& pz,
                                   float& qx, float& qy, float& qz, float& qw)
{
    auto d1 = [&](float& dst, CudaArray<float>& src) {
        cudaMemcpy(&dst, src.gpu_data() + idx, sizeof(float), cudaMemcpyDeviceToHost);
    };
    d1(px, pos_x_); d1(py, pos_y_); d1(pz, pos_z_);
    d1(qx, quat_x_); d1(qy, quat_y_); d1(qz, quat_z_); d1(qw, quat_w_);
}

}  // namespace avbd
}  // namespace chysx
