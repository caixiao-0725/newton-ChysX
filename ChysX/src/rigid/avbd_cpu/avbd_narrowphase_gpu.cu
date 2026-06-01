// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU narrowphase: one thread per broadphase pair performs OBB-OBB SAT,
// writes contacts into a flat buffer via atomics.

#include "avbd_narrowphase_gpu.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>

namespace chysx {
namespace avbd {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("avbd::NarrowphaseGPU: ") + w +
                                 ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Device-side math (self-contained, no host headers)
// ---------------------------------------------------------------------------

struct F3 {
    float x, y, z;
    __device__ float  operator[](int i) const { return (&x)[i]; }
    __device__ float& operator[](int i)       { return (&x)[i]; }
};

struct Q4 { float x, y, z, w; };

__device__ F3 f3(float a, float b, float c) { return {a, b, c}; }
__device__ F3 operator+(F3 a, F3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ F3 operator-(F3 a, F3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ F3 operator*(F3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ F3 operator-(F3 a) { return {-a.x, -a.y, -a.z}; }
__device__ float dot(F3 a, F3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float lengthSq(F3 v) { return dot(v, v); }
__device__ F3 cross(F3 a, F3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ F3 normalize(F3 v) { float r = rsqrtf(dot(v, v)); return v * r; }
__device__ float absDot(F3 a, F3 b) { return fabsf(dot(a, b)); }
__device__ float clampf(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }

__device__ F3 qrot(Q4 q, F3 v) {
    F3 u = {q.x, q.y, q.z};
    F3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

__device__ Q4 qconj(Q4 q) { return {-q.x, -q.y, -q.z, q.w}; }

// Compute rotation matrix rows from quaternion (each row = rotated basis vector)
__device__ void quatToAxes(Q4 q, F3 ax[3]) {
    ax[0] = qrot(q, f3(1, 0, 0));
    ax[1] = qrot(q, f3(0, 1, 0));
    ax[2] = qrot(q, f3(0, 0, 1));
}

// Orthonormal basis from normal (row 0 = normal)
__device__ void orthonormalBasis(F3 n, float basis[9]) {
    F3 t1;
    if (fabsf(n.x) > fabsf(n.z))
        t1 = normalize(f3(-n.y, n.x, 0));
    else
        t1 = normalize(f3(0, -n.z, n.y));
    F3 t2 = cross(n, t1);
    basis[0] = n.x;  basis[1] = n.y;  basis[2] = n.z;
    basis[3] = t1.x; basis[4] = t1.y; basis[5] = t1.z;
    basis[6] = t2.x; basis[7] = t2.y; basis[8] = t2.z;
}

// ---------------------------------------------------------------------------
// SAT constants & types
// ---------------------------------------------------------------------------

constexpr int MAX_CONTACTS = 8;
constexpr int MAX_POLY_VERTS = 16;
constexpr float SAT_EPS = 1.0e-6f;
constexpr float PLANE_EPS = 1.0e-5f;
constexpr float MERGE_DIST_SQ = 1.0e-6f;

enum AxisType { AXIS_FACE_A = 0, AXIS_FACE_B = 1, AXIS_EDGE = 2 };

struct SatResult {
    AxisType type;
    int indexA, indexB;
    float separation;
    F3 normalAB;
    bool valid;
};

// ---------------------------------------------------------------------------
// SAT device functions
// ---------------------------------------------------------------------------

__device__ bool testAxis(F3 halfA, const F3 axA[3], F3 halfB, const F3 axB[3],
                         F3 delta, F3 axis, AxisType type, int idxA, int idxB,
                         SatResult& best) {
    float lenSq_ = lengthSq(axis);
    if (lenSq_ < SAT_EPS) return true;
    float invLen = rsqrtf(lenSq_);
    F3 n = axis * invLen;
    if (dot(n, delta) < 0.0f) n = -n;
    float distance = fabsf(dot(delta, n));
    float rA = halfA.x * absDot(n, axA[0]) +
               halfA.y * absDot(n, axA[1]) +
               halfA.z * absDot(n, axA[2]);
    float rB = halfB.x * absDot(n, axB[0]) +
               halfB.y * absDot(n, axB[1]) +
               halfB.z * absDot(n, axB[2]);
    float sep = distance - (rA + rB);
    if (sep > 0.0f) return false;
    if (!best.valid || sep > best.separation) {
        best.valid = true;
        best.type = type;
        best.indexA = idxA;
        best.indexB = idxB;
        best.separation = sep;
        best.normalAB = n;
    }
    return true;
}

__device__ F3 supportPoint(F3 center, F3 half, const F3 ax[3], F3 dir) {
    float sx = dot(dir, ax[0]) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot(dir, ax[1]) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot(dir, ax[2]) >= 0.0f ? 1.0f : -1.0f;
    return center + ax[0] * (half.x * sx) + ax[1] * (half.y * sy) + ax[2] * (half.z * sz);
}

__device__ void getFaceAxes(const F3 ax[3], F3 halfExt, int ai,
                            F3& u, F3& v, float& eu, float& ev) {
    if (ai == 0) { u = ax[1]; v = ax[2]; eu = halfExt.y; ev = halfExt.z; }
    else if (ai == 1) { u = ax[0]; v = ax[2]; eu = halfExt.x; ev = halfExt.z; }
    else { u = ax[0]; v = ax[1]; eu = halfExt.x; ev = halfExt.y; }
}

struct FaceFrame {
    int ai;
    F3 normal, center, u, v;
    float eu, ev;
};

__device__ void buildFaceFrame(F3 boxCenter, F3 half, const F3 ax[3],
                               int ai, F3 outward, FaceFrame& f) {
    float s = dot(outward, ax[ai]) >= 0.0f ? 1.0f : -1.0f;
    f.ai = ai;
    f.normal = ax[ai] * s;
    f.center = boxCenter + f.normal * half[ai];
    getFaceAxes(ax, half, ai, f.u, f.v, f.eu, f.ev);
}

__device__ int chooseIncidentAxis(const F3 ax[3], F3 refN) {
    int best = 0;
    float bestD = -1e30f;
    for (int i = 0; i < 3; i++) {
        float d = absDot(ax[i], refN);
        if (d > bestD) { bestD = d; best = i; }
    }
    return best;
}

__device__ void buildIncidentFace(F3 center, F3 half, const F3 ax[3],
                                  int ai, F3 refN, F3 out[4]) {
    float s = dot(ax[ai], refN) > 0.0f ? -1.0f : 1.0f;
    F3 fn = ax[ai] * s;
    F3 fc = center + fn * half[ai];
    F3 u, v; float eu, ev;
    getFaceAxes(ax, half, ai, u, v, eu, ev);
    out[0] = fc + u * eu + v * ev;
    out[1] = fc - u * eu + v * ev;
    out[2] = fc - u * eu - v * ev;
    out[3] = fc + u * eu - v * ev;
}

__device__ int clipPoly(const F3* in, int cnt, F3 pn, float po, F3* out) {
    if (cnt <= 0) return 0;
    int oc = 0;
    F3 a = in[cnt - 1];
    float da = dot(pn, a) - po;
    for (int i = 0; i < cnt; i++) {
        F3 b = in[i];
        float db = dot(pn, b) - po;
        bool aIn = da <= PLANE_EPS;
        bool bIn = db <= PLANE_EPS;
        if (aIn != bIn) {
            float t = 0.0f;
            float denom = da - db;
            if (fabsf(denom) > SAT_EPS)
                t = clampf(da / denom, 0.0f, 1.0f);
            if (oc < MAX_POLY_VERTS)
                out[oc++] = a + (b - a) * t;
        }
        if (bIn && oc < MAX_POLY_VERTS)
            out[oc++] = b;
        a = b;
        da = db;
    }
    return oc;
}

__device__ bool tryAddContact(GpuContact* contacts, int& cnt,
                              F3* midpoints,
                              F3 xA, F3 xB, int featKey,
                              Q4 qA, F3 posA, Q4 qB, F3 posB) {
    F3 mid = (xA + xB) * 0.5f;
    for (int i = 0; i < cnt; i++) {
        F3 d = mid - midpoints[i];
        if (lengthSq(d) < MERGE_DIST_SQ) return false;
    }
    if (cnt >= MAX_CONTACTS) return false;
    Q4 invA = qconj(qA);
    Q4 invB = qconj(qB);
    GpuContact& c = contacts[cnt];
    c.feature_key = featKey;
    F3 rA = qrot(invA, xA - posA);
    F3 rB = qrot(invB, xB - posB);
    c.rA_x = rA.x; c.rA_y = rA.y; c.rA_z = rA.z;
    c.rB_x = rB.x; c.rB_y = rB.y; c.rB_z = rB.z;
    // Default warm-start values (overwritten by warmstart_kernel if matched)
    c.lambda_x = 0.0f; c.lambda_y = 0.0f; c.lambda_z = 0.0f;
    c.penalty_x = 1.0f; c.penalty_y = 1.0f; c.penalty_z = 1.0f;
    c.C0_x = 0.0f; c.C0_y = 0.0f; c.C0_z = 0.0f;
    c.stick = 0;
    midpoints[cnt] = mid;
    cnt++;
    return true;
}

__device__ void supportEdge(F3 center, F3 half, const F3 ax[3],
                            int ai, F3 dir, F3& ea, F3& eb) {
    int a1 = (ai + 1) % 3, a2 = (ai + 2) % 3;
    float s1 = dot(dir, ax[a1]) >= 0.0f ? 1.0f : -1.0f;
    float s2 = dot(dir, ax[a2]) >= 0.0f ? 1.0f : -1.0f;
    F3 ec = center + ax[a1] * (half[a1] * s1) + ax[a2] * (half[a2] * s2);
    ea = ec - ax[ai] * half[ai];
    eb = ec + ax[ai] * half[ai];
}

__device__ void closestOnSegments(F3 p0, F3 p1, F3 q0, F3 q1, F3& c0, F3& c1) {
    F3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s = 0.0f, t = 0.0f;
    if (a <= SAT_EPS && e <= SAT_EPS) { c0 = p0; c1 = q0; return; }
    if (a <= SAT_EPS) {
        t = clampf(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= SAT_EPS) {
            s = clampf(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (fabsf(denom) > SAT_EPS)
                s = clampf((b * f - c * e) / denom, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = clampf(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = clampf((b - c) / a, 0.0f, 1.0f); }
        }
    }
    c0 = p0 + d1 * s;
    c1 = q0 + d2 * t;
}

__device__ int buildFaceManifold(F3 cenA, F3 halfA, const F3 axA[3], Q4 qA,
                                 F3 cenB, F3 halfB, const F3 axB[3], Q4 qB,
                                 bool refIsA, int refAxis, F3 normalAB,
                                 GpuContact* contacts) {
    F3 refCen   = refIsA ? cenA : cenB;
    F3 refHalf  = refIsA ? halfA : halfB;
    const F3* refAx  = refIsA ? axA : axB;
    F3 incCen   = refIsA ? cenB : cenA;
    F3 incHalf  = refIsA ? halfB : halfA;
    const F3* incAx  = refIsA ? axB : axA;
    F3 refOut = refIsA ? normalAB : -normalAB;

    FaceFrame rf;
    buildFaceFrame(refCen, refHalf, refAx, refAxis, refOut, rf);
    int incAxis = chooseIncidentAxis(incAx, rf.normal);

    F3 clip0[MAX_POLY_VERTS], clip1[MAX_POLY_VERTS];
    buildIncidentFace(incCen, incHalf, incAx, incAxis, rf.normal, clip0);
    int cnt = 4;

    cnt = clipPoly(clip0, cnt, rf.u,  dot(rf.u, rf.center) + rf.eu, clip1);
    if (!cnt) return 0;
    cnt = clipPoly(clip1, cnt, -rf.u, dot(-rf.u, rf.center) + rf.eu, clip0);
    if (!cnt) return 0;
    cnt = clipPoly(clip0, cnt, rf.v,  dot(rf.v, rf.center) + rf.ev, clip1);
    if (!cnt) return 0;
    cnt = clipPoly(clip1, cnt, -rf.v, dot(-rf.v, rf.center) + rf.ev, clip0);
    if (!cnt) return 0;

    int cc = 0;
    F3 mids[MAX_CONTACTS];
    int prefix = (refIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
    prefix |= (refAxis & 0xFF) << 16;
    prefix |= (incAxis & 0xFF) << 8;

    for (int i = 0; i < cnt && cc < MAX_CONTACTS; i++) {
        F3 pInc = clip0[i];
        float dist = dot(pInc - rf.center, rf.normal);
        if (dist > PLANE_EPS) continue;
        F3 pRef = pInc - rf.normal * dist;
        F3 xA = refIsA ? pRef : pInc;
        F3 xB = refIsA ? pInc : pRef;
        tryAddContact(contacts, cc, mids, xA, xB, prefix | (i & 0xFF),
                      qA, cenA, qB, cenB);
    }
    if (!cc) {
        F3 xA = supportPoint(cenA, halfA, axA, normalAB);
        F3 xB = supportPoint(cenB, halfB, axB, -normalAB);
        tryAddContact(contacts, cc, mids, xA, xB, prefix, qA, cenA, qB, cenB);
    }
    return cc;
}

__device__ int buildEdgeContact(F3 cenA, F3 halfA, const F3 axA[3], Q4 qA,
                                F3 cenB, F3 halfB, const F3 axB[3], Q4 qB,
                                int axisA, int axisB, F3 normalAB,
                                GpuContact* contacts) {
    F3 a0, a1, b0, b1;
    supportEdge(cenA, halfA, axA, axisA, normalAB, a0, a1);
    supportEdge(cenB, halfB, axB, axisB, -normalAB, b0, b1);
    F3 xA, xB;
    closestOnSegments(a0, a1, b0, b1, xA, xB);
    int cc = 0;
    F3 mids[MAX_CONTACTS];
    int fk = (AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    tryAddContact(contacts, cc, mids, xA, xB, fk, qA, cenA, qB, cenB);
    if (!cc) {
        xA = supportPoint(cenA, halfA, axA, normalAB);
        xB = supportPoint(cenB, halfB, axB, -normalAB);
        tryAddContact(contacts, cc, mids, xA, xB, fk, qA, cenA, qB, cenB);
    }
    return cc;
}

// ---------------------------------------------------------------------------
// Main narrowphase kernel: one thread per broadphase pair
// ---------------------------------------------------------------------------

__global__ void sat_narrowphase_kernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy, const float* __restrict__ qz,
    const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    const float* __restrict__ fric,
    const int* __restrict__ pair_a, const int* __restrict__ pair_b,
    int n_pairs,
    GpuManifold* __restrict__ manifolds,
    GpuContact*  __restrict__ contacts,
    int* __restrict__ manifold_count,
    int* __restrict__ contact_count,
    int max_contacts,
    int* __restrict__ vtx_counts,
    VertexEntry* __restrict__ vtx_table,
    int vtx_stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    int ia = pair_a[tid], ib = pair_b[tid];

    F3 cenA = {px[ia], py[ia], pz[ia]};
    Q4 quatA = {qx[ia], qy[ia], qz[ia], qw[ia]};
    F3 halfA = {hx[ia], hy[ia], hz[ia]};

    F3 cenB = {px[ib], py[ib], pz[ib]};
    Q4 quatB = {qx[ib], qy[ib], qz[ib], qw[ib]};
    F3 halfB = {hx[ib], hy[ib], hz[ib]};

    F3 axA[3], axB[3];
    quatToAxes(quatA, axA);
    quatToAxes(quatB, axB);

    F3 delta = cenB - cenA;

    SatResult bestFace;
    bestFace.separation = -1e30f; bestFace.valid = false;
    SatResult bestEdge;
    bestEdge.separation = -1e30f; bestEdge.valid = false;

    for (int i = 0; i < 3; i++)
        if (!testAxis(halfA, axA, halfB, axB, delta, axA[i], AXIS_FACE_A, i, -1, bestFace)) return;
    for (int i = 0; i < 3; i++)
        if (!testAxis(halfA, axA, halfB, axB, delta, axB[i], AXIS_FACE_B, -1, i, bestFace)) return;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!testAxis(halfA, axA, halfB, axB, delta, cross(axA[i], axB[j]),
                          AXIS_EDGE, i, j, bestEdge)) return;

    if (!bestFace.valid) return;

    SatResult best = bestFace;
    if (bestEdge.valid) {
        if (0.95f * bestEdge.separation > bestFace.separation + 0.01f)
            best = bestEdge;
    }

    // SAT found collision — generate contacts into thread-local buffer
    GpuContact localContacts[MAX_CONTACTS];
    int nc = 0;

    if (best.type == AXIS_EDGE)
        nc = buildEdgeContact(cenA, halfA, axA, quatA, cenB, halfB, axB, quatB,
                              best.indexA, best.indexB, best.normalAB, localContacts);
    else if (best.type == AXIS_FACE_A)
        nc = buildFaceManifold(cenA, halfA, axA, quatA, cenB, halfB, axB, quatB,
                               true, best.indexA, best.normalAB, localContacts);
    else
        nc = buildFaceManifold(cenA, halfA, axA, quatA, cenB, halfB, axB, quatB,
                               false, best.indexB, best.normalAB, localContacts);

    if (nc <= 0) return;

    // Atomically reserve space in the global contact array
    int coff = atomicAdd(contact_count, nc);
    if (coff + nc > max_contacts) return;

    // Atomically reserve a manifold slot
    int midx = atomicAdd(manifold_count, 1);
    if (midx >= max_contacts / MAX_CONTACTS) {
        atomicAdd(contact_count, -nc);
        return;
    }

    for (int i = 0; i < nc; i++)
        contacts[coff + i] = localContacts[i];

    GpuManifold& m = manifolds[midx];
    m.body_a = ia;
    m.body_b = ib;
    m.num_contacts = nc;
    m.contact_offset = coff;
    m.friction = sqrtf(fric[ia] * fric[ib]);

    F3 n = -best.normalAB;
    orthonormalBasis(n, m.basis);

    // Fill per-body vertex table: body ia sees ib at manifold midx, and vice versa
    int slotA = atomicAdd(&vtx_counts[ia], 1);
    if (slotA < vtx_stride) {
        vtx_table[ia * vtx_stride + slotA].other_body = ib;
        vtx_table[ia * vtx_stride + slotA].manifold_idx = midx;
    }
    int slotB = atomicAdd(&vtx_counts[ib], 1);
    if (slotB < vtx_stride) {
        vtx_table[ib * vtx_stride + slotB].other_body = ia;
        vtx_table[ib * vtx_stride + slotB].manifold_idx = midx;
    }
}

// ---------------------------------------------------------------------------
// Ground plane narrowphase: one thread per dynamic body, tests 8 box vertices
// against the ground plane (z = ground_z, normal = +Z).
// Ground body index = ground_body_idx (typically n_bodies).
// Appends manifolds/contacts after existing body-body results.
// ---------------------------------------------------------------------------

__device__ const float kCornerSign[8][3] = {
    {-1,-1,-1},{+1,-1,-1},{+1,+1,-1},{-1,+1,-1},
    {-1,-1,+1},{+1,-1,+1},{+1,+1,+1},{-1,+1,+1}
};

__global__ void ground_plane_narrowphase_kernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy, const float* __restrict__ qz,
    const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    const float* __restrict__ fric,
    const float* __restrict__ mass,
    int n_bodies,
    float ground_z, float ground_friction,
    int ground_body_idx,
    GpuManifold* __restrict__ manifolds,
    GpuContact*  __restrict__ contacts,
    int* __restrict__ manifold_count,
    int* __restrict__ contact_count,
    int max_contacts,
    int* __restrict__ vtx_counts,
    VertexEntry* __restrict__ vtx_table,
    int vtx_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    if (mass[i] <= 0.0f) return;

    F3 cen = {px[i], py[i], pz[i]};
    Q4 quat = {qx[i], qy[i], qz[i], qw[i]};
    F3 half = {hx[i], hy[i], hz[i]};

    GpuContact localC[MAX_CONTACTS];
    int nc = 0;

    F3 groundNormal = f3(0, 0, 1);
    F3 groundPoint  = f3(0, 0, ground_z);
    Q4 invQ = qconj(quat);

    constexpr float kGroundMargin = 0.02f;

    for (int c = 0; c < 8 && nc < MAX_CONTACTS; c++) {
        F3 local = f3(kCornerSign[c][0] * half.x,
                      kCornerSign[c][1] * half.y,
                      kCornerSign[c][2] * half.z);
        F3 world = cen + qrot(quat, local);
        float dist = world.z - ground_z;
        if (dist > kGroundMargin) continue;

        F3 contactOnBody = world;
        F3 contactOnGround = f3(world.x, world.y, ground_z);

        F3 rA = qrot(invQ, contactOnBody - cen);
        F3 rB = contactOnGround - groundPoint;

        GpuContact& ct = localC[nc];
        ct.feature_key = (3 << 24) | c;
        ct.rA_x = rA.x; ct.rA_y = rA.y; ct.rA_z = rA.z;
        ct.rB_x = rB.x; ct.rB_y = rB.y; ct.rB_z = rB.z;
        ct.lambda_x = 0; ct.lambda_y = 0; ct.lambda_z = 0;
        ct.penalty_x = 1; ct.penalty_y = 1; ct.penalty_z = 1;
        ct.C0_x = 0; ct.C0_y = 0; ct.C0_z = 0;
        ct.stick = 0;
        nc++;
    }

    if (nc <= 0) return;

    int coff = atomicAdd(contact_count, nc);
    if (coff + nc > max_contacts) return;

    int midx = atomicAdd(manifold_count, 1);
    if (midx >= max_contacts / MAX_CONTACTS) {
        atomicAdd(contact_count, -nc);
        return;
    }

    for (int j = 0; j < nc; j++)
        contacts[coff + j] = localC[j];

    GpuManifold& m = manifolds[midx];
    m.body_a = i;
    m.body_b = ground_body_idx;
    m.num_contacts = nc;
    m.contact_offset = coff;
    m.friction = sqrtf(fric[i] * ground_friction);

    orthonormalBasis(f3(0, 0, 1), m.basis);

    int slotA = atomicAdd(&vtx_counts[i], 1);
    if (slotA < vtx_stride) {
        vtx_table[i * vtx_stride + slotA].other_body = ground_body_idx;
        vtx_table[i * vtx_stride + slotA].manifold_idx = midx;
    }
    int slotB = atomicAdd(&vtx_counts[ground_body_idx], 1);
    if (slotB < vtx_stride) {
        vtx_table[ground_body_idx * vtx_stride + slotB].other_body = i;
        vtx_table[ground_body_idx * vtx_stride + slotB].manifold_idx = midx;
    }
}

// ---------------------------------------------------------------------------
// Sphere narrowphase: one thread per box body tests box-sphere overlap.
// Thread 0 also handles sphere-ground if enabled.
// Feature key prefix: (4 << 24) for box-sphere, (5 << 24) for sphere-ground.
// ---------------------------------------------------------------------------

__global__ void sphere_narrowphase_kernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy, const float* __restrict__ qz,
    const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    const float* __restrict__ fric,
    const float* __restrict__ mass,
    int n_box_bodies,
    int sphere_body_idx,
    float sphere_radius,
    float sphere_friction,
    bool has_ground,
    float ground_z,
    float ground_friction,
    int ground_body_idx,
    GpuManifold* __restrict__ manifolds,
    GpuContact*  __restrict__ contacts,
    int* __restrict__ manifold_count,
    int* __restrict__ contact_count,
    int max_contacts,
    int* __restrict__ vtx_counts,
    VertexEntry* __restrict__ vtx_table,
    int vtx_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 handles sphere-ground contact
    if (i == 0 && has_ground) {
        F3 sc = {px[sphere_body_idx], py[sphere_body_idx], pz[sphere_body_idx]};
        Q4 sq = {qx[sphere_body_idx], qy[sphere_body_idx], qz[sphere_body_idx], qw[sphere_body_idx]};
        constexpr float kMargin = 0.02f;
        float dist = sc.z - sphere_radius - ground_z;
        if (dist < kMargin) {
            F3 normal = f3(0, 0, 1);
            F3 contactOnSphere = f3(sc.x, sc.y, sc.z - sphere_radius);
            F3 contactOnGround = f3(sc.x, sc.y, ground_z);

            Q4 invSQ = qconj(sq);
            F3 rA = qrot(invSQ, contactOnSphere - sc);
            F3 rB = contactOnGround - f3(0, 0, ground_z);

            GpuContact ct;
            ct.feature_key = (5 << 24);
            ct.rA_x = rA.x; ct.rA_y = rA.y; ct.rA_z = rA.z;
            ct.rB_x = rB.x; ct.rB_y = rB.y; ct.rB_z = rB.z;
            ct.lambda_x = 0; ct.lambda_y = 0; ct.lambda_z = 0;
            ct.penalty_x = 1; ct.penalty_y = 1; ct.penalty_z = 1;
            ct.C0_x = 0; ct.C0_y = 0; ct.C0_z = 0;
            ct.stick = 0;

            int coff = atomicAdd(contact_count, 1);
            if (coff < max_contacts) {
                int midx = atomicAdd(manifold_count, 1);
                if (midx < max_contacts / MAX_CONTACTS) {
                    contacts[coff] = ct;
                    GpuManifold& m = manifolds[midx];
                    m.body_a = sphere_body_idx;
                    m.body_b = ground_body_idx;
                    m.num_contacts = 1;
                    m.contact_offset = coff;
                    m.friction = sqrtf(sphere_friction * ground_friction);
                    orthonormalBasis(normal, m.basis);

                    int slotA = atomicAdd(&vtx_counts[sphere_body_idx], 1);
                    if (slotA < vtx_stride) {
                        vtx_table[sphere_body_idx * vtx_stride + slotA].other_body = ground_body_idx;
                        vtx_table[sphere_body_idx * vtx_stride + slotA].manifold_idx = midx;
                    }
                    int slotB = atomicAdd(&vtx_counts[ground_body_idx], 1);
                    if (slotB < vtx_stride) {
                        vtx_table[ground_body_idx * vtx_stride + slotB].other_body = sphere_body_idx;
                        vtx_table[ground_body_idx * vtx_stride + slotB].manifold_idx = midx;
                    }
                } else {
                    atomicAdd(contact_count, -1);
                }
            }
        }
    }

    // Each thread handles one box body vs the sphere
    if (i >= n_box_bodies) return;
    if (mass[i] <= 0.0f) return;

    F3 sc = {px[sphere_body_idx], py[sphere_body_idx], pz[sphere_body_idx]};
    Q4 sq = {qx[sphere_body_idx], qy[sphere_body_idx], qz[sphere_body_idx], qw[sphere_body_idx]};

    F3 boxCen = {px[i], py[i], pz[i]};
    Q4 boxQ   = {qx[i], qy[i], qz[i], qw[i]};
    F3 half   = {hx[i], hy[i], hz[i]};

    Q4 invBoxQ = qconj(boxQ);
    F3 sLocal = qrot(invBoxQ, sc - boxCen);

    F3 closest;
    closest.x = clampf(sLocal.x, -half.x, half.x);
    closest.y = clampf(sLocal.y, -half.y, half.y);
    closest.z = clampf(sLocal.z, -half.z, half.z);

    F3 delta = sLocal - closest;
    float distSq = lengthSq(delta);

    constexpr float kMargin = 0.02f;
    float threshold = sphere_radius + kMargin;

    F3 normalWorld;
    float penetration;
    F3 contactOnBox_world;
    F3 contactOnSphere_world;

    if (distSq < 1e-8f) {
        // Sphere center inside box: find shallowest penetration axis
        float minPen = 1e30f;
        int bestAxis = 0;
        float bestSign = 1.0f;
        for (int a = 0; a < 3; a++) {
            float dplus  = half[a] - sLocal[a];
            float dminus = half[a] + sLocal[a];
            if (dplus < minPen) { minPen = dplus;  bestAxis = a; bestSign =  1.0f; }
            if (dminus < minPen) { minPen = dminus; bestAxis = a; bestSign = -1.0f; }
        }
        F3 normalLocal = f3(0, 0, 0);
        normalLocal[bestAxis] = bestSign;
        normalWorld = qrot(boxQ, normalLocal);
        penetration = minPen + sphere_radius;

        F3 facePoint = sLocal;
        facePoint[bestAxis] = bestSign * half[bestAxis];
        contactOnBox_world = boxCen + qrot(boxQ, facePoint);
        contactOnSphere_world = sc - normalWorld * sphere_radius;
    } else {
        float dist = sqrtf(distSq);
        if (dist >= threshold) return;

        F3 normalLocal = delta * (1.0f / dist);
        normalWorld = qrot(boxQ, normalLocal);
        penetration = sphere_radius - dist;

        contactOnBox_world = boxCen + qrot(boxQ, closest);
        contactOnSphere_world = sc - normalWorld * sphere_radius;
    }

    if (penetration < -kMargin) return;

    Q4 invBoxQ2 = qconj(boxQ);
    Q4 invSQ = qconj(sq);
    F3 rA = qrot(invBoxQ2, contactOnBox_world - boxCen);
    F3 rB = qrot(invSQ, contactOnSphere_world - sc);

    GpuContact ct;
    ct.feature_key = (4 << 24) | (i & 0xFFFFFF);
    ct.rA_x = rA.x; ct.rA_y = rA.y; ct.rA_z = rA.z;
    ct.rB_x = rB.x; ct.rB_y = rB.y; ct.rB_z = rB.z;
    ct.lambda_x = 0; ct.lambda_y = 0; ct.lambda_z = 0;
    ct.penalty_x = 1; ct.penalty_y = 1; ct.penalty_z = 1;
    ct.C0_x = 0; ct.C0_y = 0; ct.C0_z = 0;
    ct.stick = 0;

    int coff = atomicAdd(contact_count, 1);
    if (coff >= max_contacts) return;

    int midx = atomicAdd(manifold_count, 1);
    if (midx >= max_contacts / MAX_CONTACTS) {
        atomicAdd(contact_count, -1);
        return;
    }

    contacts[coff] = ct;

    GpuManifold& m = manifolds[midx];
    m.body_a = i;
    m.body_b = sphere_body_idx;
    m.num_contacts = 1;
    m.contact_offset = coff;
    m.friction = sqrtf(fric[i] * sphere_friction);

    // Normal points from box (A) toward sphere (B)
    orthonormalBasis(-normalWorld, m.basis);

    int slotA = atomicAdd(&vtx_counts[i], 1);
    if (slotA < vtx_stride) {
        vtx_table[i * vtx_stride + slotA].other_body = sphere_body_idx;
        vtx_table[i * vtx_stride + slotA].manifold_idx = midx;
    }
    int slotB = atomicAdd(&vtx_counts[sphere_body_idx], 1);
    if (slotB < vtx_stride) {
        vtx_table[sphere_body_idx * vtx_stride + slotB].other_body = i;
        vtx_table[sphere_body_idx * vtx_stride + slotB].manifold_idx = midx;
    }
}

// ---------------------------------------------------------------------------
// Warm-start kernel: match current contacts against previous frame
// ---------------------------------------------------------------------------

__global__ void warmstart_kernel(
    GpuManifold* __restrict__ cur_manifolds,
    GpuContact*  __restrict__ cur_contacts,
    int n_cur_manifolds,
    const GpuManifold* __restrict__ prev_manifolds,
    const GpuContact*  __restrict__ prev_contacts,
    const int* __restrict__ prev_vtx_counts,
    const VertexEntry* __restrict__ prev_vtx_table,
    int prev_vtx_stride,
    int prev_n_bodies,
    float /*alpha_unused*/,
    float /*gamma_unused*/)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cur_manifolds) return;

    const GpuManifold& cm = cur_manifolds[tid];
    int bodyA = cm.body_a;
    int bodyB = cm.body_b;

    // Look up in prev frame's vertex table: does bodyA have bodyB as neighbor?
    int prev_midx = -1;
    if (bodyA < prev_n_bodies) {
        int cnt = prev_vtx_counts[bodyA];
        if (cnt > prev_vtx_stride) cnt = prev_vtx_stride;
        for (int s = 0; s < cnt; s++) {
            const VertexEntry& e = prev_vtx_table[bodyA * prev_vtx_stride + s];
            if (e.other_body == bodyB) {
                prev_midx = e.manifold_idx;
                break;
            }
        }
    }

    if (prev_midx < 0) return;

    const GpuManifold& pm = prev_manifolds[prev_midx];

    // For each current contact, search for matching feature_key in prev manifold
    for (int ci = 0; ci < cm.num_contacts; ci++) {
        GpuContact& cc = cur_contacts[cm.contact_offset + ci];

        for (int pj = 0; pj < pm.num_contacts; pj++) {
            const GpuContact& pc = prev_contacts[pm.contact_offset + pj];
            if (cc.feature_key != pc.feature_key) continue;

            // Found matching contact — raw transfer of post-solver state.
            // CPU initialize() will apply alpha*gamma scaling.
            bool prev_stick = (pc.stick != 0);
            if (prev_stick) {
                cc.rA_x = pc.rA_x; cc.rA_y = pc.rA_y; cc.rA_z = pc.rA_z;
                cc.rB_x = pc.rB_x; cc.rB_y = pc.rB_y; cc.rB_z = pc.rB_z;
            }
            cc.lambda_x = pc.lambda_x;
            cc.lambda_y = pc.lambda_y;
            cc.lambda_z = pc.lambda_z;
            cc.penalty_x = pc.penalty_x;
            cc.penalty_y = pc.penalty_y;
            cc.penalty_z = pc.penalty_z;
            cc.C0_x = pc.C0_x;
            cc.C0_y = pc.C0_y;
            cc.C0_z = pc.C0_z;
            cc.stick = pc.stick;
            break;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-side interface
// ---------------------------------------------------------------------------

void NarrowphaseGPU::build(int max_pairs) {
    max_pairs_ = max_pairs;
    int max_contacts = max_pairs * MAX_CONTACTS;
    manifolds_.resize(max_pairs);
    contacts_.resize(max_contacts);
    manifold_count_.resize(1);
    contact_count_.resize(1);
}

int NarrowphaseGPU::query(
    const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
    const float* quat_x_dev, const float* quat_y_dev,
    const float* quat_z_dev, const float* quat_w_dev,
    const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
    const float* friction_dev,
    const int* pair_a_dev, const int* pair_b_dev,
    int n_pairs, int n_bodies,
    GpuManifold* manifolds_out, GpuContact* contacts_out,
    int& total_contacts_out)
{
    if (n_pairs <= 0) {
        total_contacts_out = 0;
        return 0;
    }

    if (n_pairs > max_pairs_)
        build(n_pairs * 2);

    // Resize vertex table if body count changed
    if (n_bodies > max_bodies_) {
        max_bodies_ = n_bodies;
        vtx_counts_.resize(max_bodies_);
        vtx_table_.resize(max_bodies_ * VERTEX_TABLE_MAX_NEIGHBORS);
    }

    int max_contacts = max_pairs_ * MAX_CONTACTS;

    // Zero the counters and vertex counts
    check(cudaMemset(manifold_count_.gpu_data(), 0, sizeof(int)), "zero manifold_count");
    check(cudaMemset(contact_count_.gpu_data(), 0, sizeof(int)), "zero contact_count");
    check(cudaMemset(vtx_counts_.gpu_data(), 0, n_bodies * sizeof(int)), "zero vtx_counts");

    sat_narrowphase_kernel<<<grid(n_pairs), kBlock>>>(
        pos_x_dev, pos_y_dev, pos_z_dev,
        quat_x_dev, quat_y_dev, quat_z_dev, quat_w_dev,
        half_x_dev, half_y_dev, half_z_dev,
        friction_dev,
        pair_a_dev, pair_b_dev,
        n_pairs,
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        manifold_count_.gpu_data(),
        contact_count_.gpu_data(),
        max_contacts,
        vtx_counts_.gpu_data(),
        vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS);
    check(cudaGetLastError(), "sat_narrowphase_kernel launch");

    // Read back counters
    int n_manifolds = 0, n_contacts = 0;
    check(cudaMemcpy(&n_manifolds, manifold_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "manifold_count D2H");
    check(cudaMemcpy(&n_contacts, contact_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "contact_count D2H");

    n_manifolds = min(n_manifolds, max_pairs_);
    n_contacts = min(n_contacts, max_contacts);

    if (n_manifolds > 0) {
        check(cudaMemcpy(manifolds_out, manifolds_.gpu_data(),
                         n_manifolds * sizeof(GpuManifold), cudaMemcpyDeviceToHost),
              "manifolds D2H");
    }
    if (n_contacts > 0) {
        check(cudaMemcpy(contacts_out, contacts_.gpu_data(),
                         n_contacts * sizeof(GpuContact), cudaMemcpyDeviceToHost),
              "contacts D2H");
    }

    // Download vertex table
    check(cudaMemcpy(vtx_counts_.cpu_data(), vtx_counts_.gpu_data(),
                     n_bodies * sizeof(int), cudaMemcpyDeviceToHost),
          "vtx_counts D2H");
    check(cudaMemcpy(vtx_table_.cpu_data(), vtx_table_.gpu_data(),
                     n_bodies * VERTEX_TABLE_MAX_NEIGHBORS * sizeof(VertexEntry),
                     cudaMemcpyDeviceToHost),
          "vtx_table D2H");

    total_contacts_out = n_contacts;
    return n_manifolds;
}

void NarrowphaseGPU::warmstart(int n_manifolds, int n_contacts, int n_bodies,
                                GpuContact* contacts_out)
{
    if (n_manifolds <= 0 || prev_n_manifolds_ <= 0 || n_contacts <= 0)
        return;

    // Run warm-start kernel: match current contacts against prev frame.
    // The kernel transfers lambda/penalty/C0/stick directly (no scaling here;
    // these are raw values from the previous frame's post-solver state).
    warmstart_kernel<<<grid(n_manifolds), kBlock>>>(
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        n_manifolds,
        prev_manifolds_.gpu_data(),
        prev_contacts_.gpu_data(),
        prev_vtx_counts_.gpu_data(),
        prev_vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS,
        prev_n_bodies_,
        1.0f, 1.0f);  // no scaling — raw transfer
    check(cudaGetLastError(), "warmstart_kernel launch");

    // Re-download the warm-started contacts
    if (contacts_out) {
        check(cudaMemcpy(contacts_out, contacts_.gpu_data(),
                         n_contacts * sizeof(GpuContact), cudaMemcpyDeviceToHost),
              "ws contacts D2H");
    }
}

void NarrowphaseGPU::query_gpu(
    const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
    const float* quat_x_dev, const float* quat_y_dev,
    const float* quat_z_dev, const float* quat_w_dev,
    const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
    const float* friction_dev,
    const int* pair_a_dev, const int* pair_b_dev,
    int n_pairs, int n_bodies,
    int& n_manifolds_out, int& n_contacts_out)
{
    n_manifolds_out = 0;
    n_contacts_out = 0;

    // Size buffers for both broadphase pairs AND potential ground-plane contacts
    // (up to n_bodies additional manifolds from append_ground_plane_gpu).
    int needed = n_pairs + n_bodies;
    if (needed > max_pairs_)
        build(needed * 2);

    if (max_pairs_ == 0)
        build(256);

    if (n_bodies > max_bodies_) {
        max_bodies_ = n_bodies;
        vtx_counts_.resize(max_bodies_);
        vtx_table_.resize(max_bodies_ * VERTEX_TABLE_MAX_NEIGHBORS);
    }

    int max_contacts = max_pairs_ * MAX_CONTACTS;

    check(cudaMemset(manifold_count_.gpu_data(), 0, sizeof(int)), "zero manifold_count");
    check(cudaMemset(contact_count_.gpu_data(), 0, sizeof(int)), "zero contact_count");
    check(cudaMemset(vtx_counts_.gpu_data(), 0, n_bodies * sizeof(int)), "zero vtx_counts");

    if (n_pairs <= 0) return;

    sat_narrowphase_kernel<<<grid(n_pairs), kBlock>>>(
        pos_x_dev, pos_y_dev, pos_z_dev,
        quat_x_dev, quat_y_dev, quat_z_dev, quat_w_dev,
        half_x_dev, half_y_dev, half_z_dev,
        friction_dev,
        pair_a_dev, pair_b_dev,
        n_pairs,
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        manifold_count_.gpu_data(),
        contact_count_.gpu_data(),
        max_contacts,
        vtx_counts_.gpu_data(),
        vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS);
    check(cudaGetLastError(), "sat_narrowphase_kernel launch");

    int n_manifolds = 0, n_contacts = 0;
    check(cudaMemcpy(&n_manifolds, manifold_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "manifold_count D2H");
    check(cudaMemcpy(&n_contacts, contact_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "contact_count D2H");

    n_manifolds_out = min(n_manifolds, max_pairs_);
    n_contacts_out = min(n_contacts, max_contacts);
}

void NarrowphaseGPU::append_ground_plane_gpu(
    const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
    const float* quat_x_dev, const float* quat_y_dev,
    const float* quat_z_dev, const float* quat_w_dev,
    const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
    const float* friction_dev, const float* mass_dev,
    int n_bodies, float ground_z, float ground_friction,
    int ground_body_idx,
    int& n_manifolds_inout, int& n_contacts_inout)
{
    if (n_bodies <= 0) return;

    int n_bodies_with_ground = ground_body_idx + 1;
    if (n_bodies_with_ground > max_bodies_) {
        max_bodies_ = n_bodies_with_ground;
        vtx_counts_.resize(max_bodies_);
        vtx_table_.resize(max_bodies_ * VERTEX_TABLE_MAX_NEIGHBORS);
    }
    // Zero only the ground body's vtx_count slot (body slots 0..n_bodies-1 already zeroed)
    check(cudaMemset(vtx_counts_.gpu_data() + ground_body_idx, 0, sizeof(int)),
          "zero ground vtx_count");

    int max_contacts = max_pairs_ * MAX_CONTACTS;

    ground_plane_narrowphase_kernel<<<grid(n_bodies), kBlock>>>(
        pos_x_dev, pos_y_dev, pos_z_dev,
        quat_x_dev, quat_y_dev, quat_z_dev, quat_w_dev,
        half_x_dev, half_y_dev, half_z_dev,
        friction_dev, mass_dev,
        n_bodies,
        ground_z, ground_friction,
        ground_body_idx,
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        manifold_count_.gpu_data(),
        contact_count_.gpu_data(),
        max_contacts,
        vtx_counts_.gpu_data(),
        vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS);
    check(cudaGetLastError(), "ground_plane_narrowphase_kernel launch");

    int n_manifolds = 0, n_contacts = 0;
    check(cudaMemcpy(&n_manifolds, manifold_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "manifold_count D2H (ground)");
    check(cudaMemcpy(&n_contacts, contact_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "contact_count D2H (ground)");

    n_manifolds_inout = min(n_manifolds, max_pairs_);
    n_contacts_inout = min(n_contacts, max_contacts);
}

void NarrowphaseGPU::append_sphere_gpu(
    const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
    const float* quat_x_dev, const float* quat_y_dev,
    const float* quat_z_dev, const float* quat_w_dev,
    const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
    const float* friction_dev, const float* mass_dev,
    int n_box_bodies, int sphere_body_idx,
    float sphere_radius, float sphere_friction,
    bool has_ground, float ground_z, float ground_friction,
    int ground_body_idx,
    int& n_manifolds_inout, int& n_contacts_inout)
{
    if (n_box_bodies <= 0) return;

    int max_contacts = max_pairs_ * MAX_CONTACTS;

    sphere_narrowphase_kernel<<<grid(n_box_bodies), kBlock>>>(
        pos_x_dev, pos_y_dev, pos_z_dev,
        quat_x_dev, quat_y_dev, quat_z_dev, quat_w_dev,
        half_x_dev, half_y_dev, half_z_dev,
        friction_dev, mass_dev,
        n_box_bodies,
        sphere_body_idx,
        sphere_radius,
        sphere_friction,
        has_ground,
        ground_z,
        ground_friction,
        ground_body_idx,
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        manifold_count_.gpu_data(),
        contact_count_.gpu_data(),
        max_contacts,
        vtx_counts_.gpu_data(),
        vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS);
    check(cudaGetLastError(), "sphere_narrowphase_kernel launch");

    int n_manifolds = 0, n_contacts = 0;
    check(cudaMemcpy(&n_manifolds, manifold_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "manifold_count D2H (sphere)");
    check(cudaMemcpy(&n_contacts, contact_count_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "contact_count D2H (sphere)");

    n_manifolds_inout = min(n_manifolds, max_pairs_);
    n_contacts_inout = min(n_contacts, max_contacts);
}

void NarrowphaseGPU::warmstart_gpu(int n_manifolds, int n_contacts, int n_bodies)
{
    if (n_manifolds <= 0 || prev_n_manifolds_ <= 0 || n_contacts <= 0)
        return;

    warmstart_kernel<<<grid(n_manifolds), kBlock>>>(
        manifolds_.gpu_data(),
        contacts_.gpu_data(),
        n_manifolds,
        prev_manifolds_.gpu_data(),
        prev_contacts_.gpu_data(),
        prev_vtx_counts_.gpu_data(),
        prev_vtx_table_.gpu_data(),
        VERTEX_TABLE_MAX_NEIGHBORS,
        prev_n_bodies_,
        1.0f, 1.0f);
    check(cudaGetLastError(), "warmstart_kernel launch");
}

void NarrowphaseGPU::upload_contacts(const GpuContact* contacts, int n_contacts) {
    if (n_contacts <= 0 || !contacts) return;
    check(cudaMemcpy(contacts_.gpu_data(), contacts,
                     n_contacts * sizeof(GpuContact), cudaMemcpyHostToDevice),
          "upload_contacts H2D");
}

void NarrowphaseGPU::snapshot_for_next_frame(int n_manifolds, int n_contacts, int n_bodies)
{
    size_t need_m = (size_t)(max_pairs_ > n_manifolds ? max_pairs_ : n_manifolds);
    size_t need_c = (size_t)(max_pairs_ * MAX_CONTACTS);
    if (need_c < (size_t)n_contacts) need_c = (size_t)n_contacts;

    if (prev_manifolds_.gpu_size() < need_m)
        prev_manifolds_.allocate_device(need_m);
    if (prev_contacts_.gpu_size() < need_c)
        prev_contacts_.allocate_device(need_c);
    if (prev_vtx_counts_.gpu_size() < (size_t)n_bodies)
        prev_vtx_counts_.allocate_device(n_bodies);
    size_t vtx_total = (size_t)n_bodies * VERTEX_TABLE_MAX_NEIGHBORS;
    if (prev_vtx_table_.gpu_size() < vtx_total)
        prev_vtx_table_.allocate_device(vtx_total);

    if (n_manifolds > 0)
        check(cudaMemcpy(prev_manifolds_.gpu_data(), manifolds_.gpu_data(),
                         n_manifolds * sizeof(GpuManifold), cudaMemcpyDeviceToDevice),
              "snap manifolds D2D");
    if (n_contacts > 0)
        check(cudaMemcpy(prev_contacts_.gpu_data(), contacts_.gpu_data(),
                         n_contacts * sizeof(GpuContact), cudaMemcpyDeviceToDevice),
              "snap contacts D2D");
    if (n_bodies > 0) {
        check(cudaMemcpy(prev_vtx_counts_.gpu_data(), vtx_counts_.gpu_data(),
                         n_bodies * sizeof(int), cudaMemcpyDeviceToDevice),
              "snap vtx_counts D2D");
        check(cudaMemcpy(prev_vtx_table_.gpu_data(), vtx_table_.gpu_data(),
                         vtx_total * sizeof(VertexEntry), cudaMemcpyDeviceToDevice),
              "snap vtx_table D2D");
    }

    prev_n_manifolds_ = n_manifolds;
    prev_n_bodies_ = n_bodies;
}

}  // namespace avbd
}  // namespace chysx
