// SPDX-License-Identifier: Apache-2.0
//
// Narrow-phase contact detection between rigid shape pairs.
// Supports: sphere-sphere, sphere-box, sphere-capsule, sphere-plane,
//           box-plane, capsule-plane, capsule-capsule, box-box (SAT).

#include "narrow_phase.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace chysx {
namespace rigid {

namespace {

using math::Vec3f;
using math::Quatf;
using math::Mat3f;
using math::quat_rotate;
using math::quat_multiply;
using math::quat_to_matrix;
using math::transform_point;

constexpr int kBlock = 256;

// Collision margin added to C_n in AVBD constraint. At geometric contact
// (surfaces touching), C_n = COLLISION_MARGIN > 0, giving the solver a
// repulsive target before deep penetration occurs. This matches the reference
// AVBD implementation strategy.
constexpr float kCollisionMargin = 0.01f;

// World-space pose of a shape.
struct ShapePose {
    Vec3f center;
    Quatf rotation;
};

__device__ ShapePose get_shape_pose(
    int shape_idx,
    const int*   shape_body,
    const Vec3f* body_pos,
    const Quatf* body_quat,
    const Vec3f* shape_pos_local,
    const Quatf* shape_quat_local)
{
    ShapePose sp;
    int b = shape_body[shape_idx];
    if (b >= 0) {
        sp.center = transform_point(body_pos[b], body_quat[b],
                                    shape_pos_local[shape_idx]);
        sp.rotation = quat_multiply(body_quat[b], shape_quat_local[shape_idx]);
    } else {
        sp.center = shape_pos_local[shape_idx];
        sp.rotation = shape_quat_local[shape_idx];
    }
    return sp;
}

struct ContactResult {
    Vec3f p0_local;   // contact point in body-A local frame
    Vec3f p1_local;   // contact point in body-B local frame
    Vec3f normal;     // world-space, A→B
    float depth;      // positive = penetrating
    bool  valid;
};

// Transform world point to body-local frame.
__device__ Vec3f to_body_local(const Vec3f& p_world,
                               int body,
                               const Vec3f* body_pos,
                               const Quatf* body_quat) {
    if (body < 0) return p_world;
    return math::quat_rotate_inv(body_quat[body], p_world - body_pos[body]);
}

// --- Sphere-Sphere ----------------------------------------------------------

__device__ ContactResult contact_sphere_sphere(
    const ShapePose& a, float ra,
    const ShapePose& b, float rb)
{
    ContactResult cr{};
    Vec3f d = b.center - a.center;
    float dist2 = math::dot(d, d);
    float rsum = ra + rb;
    if (dist2 >= rsum * rsum || dist2 < 1e-14f) return cr;
    float dist = sqrtf(dist2);
    cr.normal = d * (1.f / dist);
    cr.depth = rsum - dist;
    // World-space contact points on surface
    Vec3f pa_world = a.center + cr.normal * ra;
    Vec3f pb_world = b.center - cr.normal * rb;
    cr.p0_local = pa_world;
    cr.p1_local = pb_world;
    cr.valid = true;
    return cr;
}

// --- Sphere-Plane -----------------------------------------------------------

__device__ ContactResult contact_sphere_plane(
    const ShapePose& sphere, float radius,
    const ShapePose& plane_sp)
{
    ContactResult cr{};
    Vec3f plane_n = quat_rotate(plane_sp.rotation, Vec3f(0.f, 1.f, 0.f));
    float d = math::dot(sphere.center - plane_sp.center, plane_n);
    if (d >= radius) return cr;
    // Convention: body A (sphere) is pushed in -n for separation.
    // The sphere should be pushed AWAY from the plane, so -n should
    // point away from the plane = along plane_n.  Thus n = -plane_n.
    cr.normal = -plane_n;
    cr.depth = radius - d;
    cr.p0_local = sphere.center - plane_n * radius;
    cr.p1_local = sphere.center - plane_n * d;
    cr.valid = true;
    return cr;
}

// --- Sphere-Box -------------------------------------------------------------

__device__ ContactResult contact_sphere_box(
    const ShapePose& sphere, float radius,
    const ShapePose& box_sp, const Vec3f& half_ext)
{
    ContactResult cr{};
    Vec3f local = math::quat_rotate_inv(box_sp.rotation, sphere.center - box_sp.center);
    Vec3f closest;
    closest.x = fminf(fmaxf(local.x, -half_ext.x), half_ext.x);
    closest.y = fminf(fmaxf(local.y, -half_ext.y), half_ext.y);
    closest.z = fminf(fmaxf(local.z, -half_ext.z), half_ext.z);
    Vec3f diff = local - closest;
    float dist2 = math::dot(diff, diff);
    if (dist2 >= radius * radius && dist2 > 1e-14f) return cr;

    float dist = sqrtf(fmaxf(dist2, 1e-14f));
    Vec3f n_local = diff * (1.f / dist);
    // n points from box toward sphere; negate to get A→B convention
    // (body A=sphere pushed in -n = toward box is wrong, so we negate)
    cr.normal = -quat_rotate(box_sp.rotation, n_local);
    cr.depth = radius - dist;
    Vec3f n_world = quat_rotate(box_sp.rotation, n_local);
    cr.p0_local = sphere.center - n_world * radius;
    cr.p1_local = box_sp.center + quat_rotate(box_sp.rotation, closest);
    cr.valid = true;
    return cr;
}

// --- Capsule helper: closest point on segment -------------------------------

__device__ Vec3f closest_point_segment(const Vec3f& a, const Vec3f& b,
                                       const Vec3f& p) {
    Vec3f ab = b - a;
    float t = math::dot(p - a, ab) / fmaxf(math::dot(ab, ab), 1e-14f);
    t = fminf(fmaxf(t, 0.f), 1.f);
    return a + ab * t;
}

// --- Sphere-Capsule ---------------------------------------------------------

__device__ ContactResult contact_sphere_capsule(
    const ShapePose& sphere, float sr,
    const ShapePose& cap, float cr_r, float cr_hh)
{
    ContactResult res{};
    Vec3f axis = quat_rotate(cap.rotation, Vec3f(0.f, 1.f, 0.f));
    Vec3f cap_a = cap.center - axis * cr_hh;
    Vec3f cap_b = cap.center + axis * cr_hh;
    Vec3f cp = closest_point_segment(cap_a, cap_b, sphere.center);
    Vec3f d = sphere.center - cp;
    float dist2 = math::dot(d, d);
    float rsum = sr + cr_r;
    if (dist2 >= rsum * rsum || dist2 < 1e-14f) return res;
    float dist = sqrtf(dist2);
    Vec3f sep = d * (1.f / dist);  // points from capsule toward sphere
    // Negate: n should point from A (sphere) toward B (capsule)
    res.normal = -sep;
    res.depth = rsum - dist;
    res.p0_local = sphere.center - sep * sr;
    res.p1_local = cp + sep * cr_r;
    res.valid = true;
    return res;
}

// --- Capsule-Capsule --------------------------------------------------------

__device__ void closest_points_segments(
    const Vec3f& a0, const Vec3f& a1,
    const Vec3f& b0, const Vec3f& b1,
    Vec3f& ca, Vec3f& cb)
{
    Vec3f d1 = a1 - a0;
    Vec3f d2 = b1 - b0;
    Vec3f r  = a0 - b0;
    float a = math::dot(d1, d1);
    float e = math::dot(d2, d2);
    float f = math::dot(d2, r);
    float s, t;
    if (a < 1e-14f && e < 1e-14f) { s = t = 0.f; }
    else if (a < 1e-14f) { s = 0.f; t = fminf(fmaxf(f / e, 0.f), 1.f); }
    else {
        float c = math::dot(d1, r);
        if (e < 1e-14f) { t = 0.f; s = fminf(fmaxf(-c / a, 0.f), 1.f); }
        else {
            float b_ = math::dot(d1, d2);
            float denom = a * e - b_ * b_;
            s = (denom > 1e-14f) ? fminf(fmaxf((b_ * f - c * e) / denom, 0.f), 1.f) : 0.f;
            t = (b_ * s + f) / e;
            if (t < 0.f) { t = 0.f; s = fminf(fmaxf(-c / a, 0.f), 1.f); }
            else if (t > 1.f) { t = 1.f; s = fminf(fmaxf((b_ - c) / a, 0.f), 1.f); }
        }
    }
    ca = a0 + d1 * s;
    cb = b0 + d2 * t;
}

__device__ ContactResult contact_capsule_capsule(
    const ShapePose& a, float ra, float ha,
    const ShapePose& b, float rb, float hb)
{
    ContactResult res{};
    Vec3f aAxis = quat_rotate(a.rotation, Vec3f(0.f, 1.f, 0.f));
    Vec3f bAxis = quat_rotate(b.rotation, Vec3f(0.f, 1.f, 0.f));
    Vec3f a0 = a.center - aAxis * ha, a1 = a.center + aAxis * ha;
    Vec3f b0 = b.center - bAxis * hb, b1 = b.center + bAxis * hb;
    Vec3f ca, cb;
    closest_points_segments(a0, a1, b0, b1, ca, cb);
    Vec3f d = cb - ca;
    float dist2 = math::dot(d, d);
    float rsum = ra + rb;
    if (dist2 >= rsum * rsum || dist2 < 1e-14f) return res;
    float dist = sqrtf(dist2);
    res.normal = d * (1.f / dist);
    res.depth = rsum - dist;
    res.p0_local = ca + res.normal * ra;
    res.p1_local = cb - res.normal * rb;
    res.valid = true;
    return res;
}

// --- Capsule-Plane ----------------------------------------------------------

__device__ ContactResult contact_capsule_plane(
    const ShapePose& cap, float cr_r, float cr_hh,
    const ShapePose& plane_sp)
{
    ContactResult res{};
    Vec3f plane_n = quat_rotate(plane_sp.rotation, Vec3f(0.f, 1.f, 0.f));
    Vec3f axis = quat_rotate(cap.rotation, Vec3f(0.f, 1.f, 0.f));
    Vec3f cap_a = cap.center - axis * cr_hh;
    Vec3f cap_b = cap.center + axis * cr_hh;
    float da = math::dot(cap_a - plane_sp.center, plane_n);
    float db = math::dot(cap_b - plane_sp.center, plane_n);
    Vec3f pt = (da < db) ? cap_a : cap_b;
    float d = fminf(da, db);
    if (d >= cr_r) return res;
    // n points from A (capsule) toward B (plane) = -plane_n
    res.normal = -plane_n;
    res.depth = cr_r - d;
    res.p0_local = pt - plane_n * cr_r;
    res.p1_local = pt - plane_n * d;
    res.valid = true;
    return res;
}

// --- Box-Plane --------------------------------------------------------------

__device__ ContactResult contact_box_plane(
    const ShapePose& box_sp, const Vec3f& half_ext,
    const ShapePose& plane_sp)
{
    ContactResult res{};
    Vec3f plane_n = quat_rotate(plane_sp.rotation, Vec3f(0.f, 1.f, 0.f));
    Mat3f R = quat_to_matrix(box_sp.rotation);
    Vec3f sign;
    sign.x = (math::dot(Vec3f(R(0,0), R(1,0), R(2,0)), plane_n) < 0.f) ? 1.f : -1.f;
    sign.y = (math::dot(Vec3f(R(0,1), R(1,1), R(2,1)), plane_n) < 0.f) ? 1.f : -1.f;
    sign.z = (math::dot(Vec3f(R(0,2), R(1,2), R(2,2)), plane_n) < 0.f) ? 1.f : -1.f;
    Vec3f local_pt(sign.x * half_ext.x, sign.y * half_ext.y, sign.z * half_ext.z);
    Vec3f world_pt = box_sp.center + quat_rotate(box_sp.rotation, local_pt);
    float d = math::dot(world_pt - plane_sp.center, plane_n);
    if (d >= 0.f) return res;
    // n points from A (box) toward B (plane) = -plane_n
    res.normal = -plane_n;
    res.depth = -d;
    res.p0_local = world_pt;
    res.p1_local = world_pt - plane_n * d;
    res.valid = true;
    return res;
}

// --- Box-Box SAT (full implementation, up to 8 contacts) --------------------
// Ported from Newton's collide_box_box (collision_primitive.py).

constexpr float kBoxMinVal = 1e-15f;

__device__ Mat3f compute_rotmore(int face_idx) {
    Mat3f m{};
    switch (face_idx) {
        case 0: m(0,2)=-1.f; m(1,1)=+1.f; m(2,0)=+1.f; break;
        case 1: m(0,0)=+1.f; m(1,2)=-1.f; m(2,1)=+1.f; break;
        case 2: m(0,0)=+1.f; m(1,1)=+1.f; m(2,2)=+1.f; break;
        case 3: m(0,2)=+1.f; m(1,1)=+1.f; m(2,0)=-1.f; break;
        case 4: m(0,0)=+1.f; m(1,2)=+1.f; m(2,1)=-1.f; break;
        case 5: m(0,0)=-1.f; m(1,1)=+1.f; m(2,2)=-1.f; break;
    }
    return m;
}

__device__ float vec3_get(const Vec3f& v, int i) {
    return (i == 0) ? v.x : (i == 1) ? v.y : v.z;
}

__device__ void vec3_set(Vec3f& v, int i, float val) {
    if (i == 0) v.x = val; else if (i == 1) v.y = val; else v.z = val;
}

__device__ Vec3f mat3_col(const Mat3f& m, int c) {
    return Vec3f(m(0,c), m(1,c), m(2,c));
}

struct BoxBoxResult {
    Vec3f pos[8];
    Vec3f normal;
    float depth[8];
    int count;
};

__device__ BoxBoxResult collide_box_box_sat(
    Vec3f box1_pos, Mat3f box1_rot, Vec3f box1_size,
    Vec3f box2_pos, Mat3f box2_rot, Vec3f box2_size,
    float margin)
{
    BoxBoxResult result{};
    result.count = 0;

    Mat3f box1_rot_T = math::transpose(box1_rot);
    Mat3f box2_rot_T = math::transpose(box2_rot);

    Vec3f pos21 = box1_rot_T * (box2_pos - box1_pos);
    Vec3f pos12 = box2_rot_T * (box1_pos - box2_pos);

    Mat3f rot21 = box1_rot_T * box2_rot;
    Mat3f rot12 = math::transpose(rot21);

    Mat3f rot21abs;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rot21abs(i,j) = fabsf(rot21(i,j));
    Mat3f rot12abs = math::transpose(rot21abs);

    Vec3f plen2 = rot21abs * box2_size;
    Vec3f plen1 = rot12abs * box1_size;

    float s_sum_3 = 3.f * (box1_size.x + box1_size.y + box1_size.z +
                            box2_size.x + box2_size.y + box2_size.z);
    float separation = margin + s_sum_3;
    int axis_code = -1;

    // Test face normals of both boxes (6 axes)
    for (int i = 0; i < 3; i++) {
        float c1 = -fabsf(vec3_get(pos21, i)) + vec3_get(box1_size, i) + vec3_get(plen2, i);
        float c2 = -fabsf(vec3_get(pos12, i)) + vec3_get(box2_size, i) + vec3_get(plen1, i);

        if (c1 < -margin || c2 < -margin) return result;

        if (c1 < separation) {
            separation = c1;
            axis_code = i + 3 * (int)(vec3_get(pos21, i) < 0.f);
        }
        if (c2 < separation) {
            separation = c2;
            axis_code = i + 3 * (int)(vec3_get(pos12, i) < 0.f) + 6;
        }
    }

    Vec3f clnorm(0.f);
    bool inv_flag = false;
    int cle1 = 0, cle2 = 0;

    // Test edge-edge cross products (9 axes)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Vec3f cross_axis;
            if (i == 0)      cross_axis = Vec3f(0.f, -rot12(j,2), rot12(j,1));
            else if (i == 1) cross_axis = Vec3f(rot12(j,2), 0.f, -rot12(j,0));
            else             cross_axis = Vec3f(-rot12(j,1), rot12(j,0), 0.f);

            float cross_length = sqrtf(math::dot(cross_axis, cross_axis));
            if (cross_length < kBoxMinVal) continue;
            float inv_cl = 1.f / cross_length;
            cross_axis = cross_axis * inv_cl;

            float box_dist = math::dot(pos21, cross_axis);
            float c3 = 0.f;
            for (int k = 0; k < 3; k++) {
                if (k != i) c3 += vec3_get(box1_size, k) * fabsf(vec3_get(cross_axis, k));
                if (k != j) {
                    int other = 3 - k - j;
                    c3 += vec3_get(box2_size, k) * rot21abs(i, other) * inv_cl;
                }
            }
            c3 -= fabsf(box_dist);

            if (c3 < -margin) return result;

            if (c3 < separation * (1.f - 1e-12f)) {
                separation = c3;
                cle1 = 0; cle2 = 0;
                for (int k = 0; k < 3; k++) {
                    if (k != i && ((vec3_get(cross_axis, k) > 0.f) != (box_dist < 0.f)))
                        cle1 += (1 << k);
                    if (k != j) {
                        int other = 3 - k - j;
                        bool sign_cond = (rot21(i, other) > 0.f) != (box_dist < 0.f);
                        if (((k - j + 3) % 3 == 1)) sign_cond = !sign_cond;
                        if (sign_cond) cle2 += (1 << k);
                    }
                }
                axis_code = 12 + i * 3 + j;
                clnorm = cross_axis;
                inv_flag = (box_dist < 0.f);
            }
        }
    }

    if (axis_code == -1) return result;

    // Contact point generation
    Vec3f points[8];
    float depth_arr[8];
    int n = 0;
    constexpr int max_con_pair = 8;

    Mat3f rw;
    Vec3f normal(0.f);
    float hz_final = 0.f;  // store hz for final transform
    Vec3f pw_final(0.f);

    if (axis_code < 12) {
        // === Face-vertex collision ===
        int face_idx = axis_code % 6;
        int box_idx = axis_code / 6;
        Mat3f rotmore = compute_rotmore(face_idx);

        Mat3f r = rotmore * (box_idx ? rot12 : rot21);
        Vec3f p = rotmore * (box_idx ? pos12 : pos21);
        Vec3f ss_v = rotmore * (box_idx ? box2_size : box1_size);
        Vec3f ss(fabsf(ss_v.x), fabsf(ss_v.y), fabsf(ss_v.z));
        Vec3f s = box_idx ? box1_size : box2_size;
        Mat3f rt = math::transpose(r);

        float lx = ss.x, ly = ss.y, hz = ss.z;
        p.z -= hz;

        int clcorner = 0;
        for (int i = 0; i < 3; i++)
            if (r(2,i) < 0.f) clcorner += (1 << i);

        Vec3f lp = p;
        for (int i = 0; i < 3; i++) {
            Vec3f rti = mat3_col(rt, i);
            lp = lp + rti * (vec3_get(s, i) * ((clcorner & (1<<i)) ? 1.f : -1.f));
        }

        int dirs = 0;
        Vec3f cn1(0.f), cn2(0.f);
        for (int i = 0; i < 3; i++) {
            if (fabsf(r(2,i)) < 0.5f) {
                Vec3f rti = mat3_col(rt, i);
                Vec3f cn = rti * (vec3_get(s, i) * ((clcorner & (1<<i)) ? -2.f : 2.f));
                if (dirs == 0) cn1 = cn; else cn2 = cn;
                dirs++;
            }
        }

        int k = dirs * dirs;

        // Edge-face intersection points
        for (int ii = 0; ii < k && n < max_con_pair; ii++) {
            for (int q = 0; q < 2 && n < max_con_pair; q++) {
                Vec3f lav = lp;
                if (ii >= 2) lav = lav + ((ii == 2) ? cn1 : cn2);
                Vec3f lbv = (ii == 0 || ii == 3) ? cn1 : cn2;
                float lbq = vec3_get(lbv, q);
                if (fabsf(lbq) <= kBoxMinVal) continue;
                float br = 1.f / lbq;
                float ssq = vec3_get(ss, q);
                float ssq_other = vec3_get(ss, 1 - q);
                for (int js = -1; js <= 1; js += 2) {
                    if (n >= max_con_pair) break;
                    float l = ssq * (float)js;
                    float c1 = (l - vec3_get(lav, q)) * br;
                    if (c1 < 0.f || c1 > 1.f) continue;
                    float c2 = vec3_get(lav, 1-q) + vec3_get(lbv, 1-q) * c1;
                    if (fabsf(c2) > ssq_other) continue;
                    points[n] = lav + lbv * c1;
                    n++;
                }
            }
        }

        // Face corners inside non-face box face
        if (dirs == 2) {
            float ax_v = cn1.x, bx_v = cn2.x;
            float ay_v = cn1.y, by_v = cn2.y;
            float det = ax_v * by_v - bx_v * ay_v;
            float C_inv = (fabsf(det) > kBoxMinVal) ? (1.f / det) : 0.f;
            for (int ii = 0; ii < 4 && n < max_con_pair; ii++) {
                float llx = (ii / 2) ? lx : -lx;
                float lly = (ii % 2) ? ly : -ly;
                float x = llx - lp.x;
                float y = lly - lp.y;
                float u = (x * by_v - y * bx_v) * C_inv;
                float v = (y * ax_v - x * ay_v) * C_inv;
                if (u > 0.f && v > 0.f && u < 1.f && v < 1.f) {
                    points[n] = Vec3f(llx, lly, lp.z + u * cn1.z + v * cn2.z);
                    n++;
                }
            }
        }

        // Vertices of non-face box inside face
        int ndirs = (1 << dirs);
        for (int ii = 0; ii < ndirs && n < max_con_pair; ii++) {
            Vec3f tmpv = lp;
            if (ii & 1) tmpv = tmpv + cn1;
            if (ii & 2) tmpv = tmpv + cn2;
            if (tmpv.x > -lx && tmpv.x < lx && tmpv.y > -ly && tmpv.y < ly) {
                points[n] = tmpv;
                n++;
            }
        }

        // Filter by depth
        int m = n; n = 0;
        for (int ii = 0; ii < m; ii++) {
            if (points[ii].z > margin) continue;
            points[ii].z *= 0.5f;
            depth_arr[n] = points[ii].z * 2.f;
            if (ii != n) points[n] = points[ii];
            n++;
        }

        rw = (box_idx ? box2_rot : box1_rot) * math::transpose(rotmore);
        normal = rw * Vec3f(0.f, 0.f, box_idx ? -1.f : 1.f);
        hz_final = hz;
        pw_final = box_idx ? box2_pos : box1_pos;

    } else {
        // === Edge-edge collision ===
        int edge1 = (axis_code - 12) / 3;
        int edge2 = (axis_code - 12) % 3;

        int ax1 = 1 - (edge2 & 1);
        int ax2 = 2 - (edge2 & 2);
        int pax1 = 1 - (edge1 & 1);
        int pax2 = 2 - (edge1 & 2);

        if (rot21abs(edge1, ax1) < rot21abs(edge1, ax2)) { int t = ax1; ax1 = ax2; ax2 = t; }
        if (rot12abs(edge2, pax1) < rot12abs(edge2, pax2)) { int t = pax1; pax1 = pax2; pax2 = t; }

        int rotmore_idx = (cle1 & (1 << pax2)) ? pax2 : pax2 + 3;
        Mat3f rotmore = compute_rotmore(rotmore_idx);

        Vec3f p_e = rotmore * pos21;
        Vec3f rnorm = rotmore * clnorm;
        Mat3f r_e = rotmore * rot21;
        Mat3f rt_e = math::transpose(r_e);
        Vec3f s_abs = rotmore * box1_size;
        Vec3f s_e(fabsf(s_abs.x), fabsf(s_abs.y), fabsf(s_abs.z));
        float lx = s_e.x, ly = s_e.y, hz = s_e.z;
        p_e.z -= hz;

        // 4 edge endpoints of box2 in this frame
        Vec3f pt0 = p_e;
        for (int kk = 0; kk < 3; kk++) {
            if (kk == edge2) continue;
            Vec3f col = mat3_col(rt_e, kk);
            float b2k = vec3_get(box2_size, kk);
            float sign = (cle2 & (1 << kk)) ? 1.f : -1.f;
            pt0 = pt0 + col * (b2k * sign);
        }
        Vec3f edge_dir = mat3_col(rt_e, edge2) * vec3_get(box2_size, edge2);
        points[0] = pt0 + edge_dir;
        points[1] = pt0 - edge_dir;

        Vec3f pt2 = p_e;
        for (int kk = 0; kk < 3; kk++) {
            if (kk == edge2) continue;
            Vec3f col = mat3_col(rt_e, kk);
            float b2k = vec3_get(box2_size, kk);
            float sign;
            if (kk == ax1) sign = (cle2 & (1 << kk)) ? -1.f : 1.f;
            else sign = (cle2 & (1 << kk)) ? 1.f : -1.f;
            pt2 = pt2 + col * (b2k * sign);
        }
        points[2] = pt2 + edge_dir;
        points[3] = pt2 - edge_dir;

        n = 4;
        Vec3f axi_lp = points[0];
        Vec3f axi_cn1 = points[1] - points[0];
        Vec3f axi_cn2 = points[2] - points[0];

        if (fabsf(rnorm.z) < kBoxMinVal) return result;
        float innorm = (inv_flag ? -1.f : 1.f) / rnorm.z;

        Vec3f pu[4];
        for (int ii = 0; ii < 4; ii++) {
            pu[ii] = points[ii];
            float c_scl = points[ii].z * (inv_flag ? -1.f : 1.f) * innorm;
            points[ii] = points[ii] - rnorm * c_scl;
        }

        Vec3f pts_lp = points[0];
        Vec3f pts_cn1 = points[1] - points[0];
        Vec3f pts_cn2 = points[2] - points[0];

        n = 0;
        // Clip edges against face
        for (int ii = 0; ii < 4 && n < max_con_pair; ii++) {
            for (int q = 0; q < 2 && n < max_con_pair; q++) {
                float la = vec3_get(pts_lp, q);
                if (ii == 2) la += vec3_get(pts_cn1, q);
                else if (ii == 3) la += vec3_get(pts_cn2, q);

                float lb = vec3_get((ii==0||ii==3) ? pts_cn1 : pts_cn2, q);
                Vec3f lua_v = axi_lp;
                if (ii == 2) lua_v = lua_v + axi_cn1;
                else if (ii == 3) lua_v = lua_v + axi_cn2;
                Vec3f lub_v = (ii==0||ii==3) ? axi_cn1 : axi_cn2;

                if (fabsf(lb) <= kBoxMinVal) continue;
                float br = 1.f / lb;
                float sq = vec3_get(s_e, q);
                float s_other = vec3_get(s_e, 1-q);

                float lc = vec3_get(pts_lp, 1-q);
                if (ii == 2) lc += vec3_get(pts_cn1, 1-q);
                else if (ii == 3) lc += vec3_get(pts_cn2, 1-q);
                float ld = vec3_get((ii==0||ii==3) ? pts_cn1 : pts_cn2, 1-q);

                for (int js = -1; js <= 1; js += 2) {
                    if (n >= max_con_pair) break;
                    float l = sq * (float)js;
                    float c1 = (l - la) * br;
                    if (c1 < 0.f || c1 > 1.f) continue;
                    float c2 = lc + ld * c1;
                    if (fabsf(c2) > s_other) continue;
                    if ((lua_v.z + lub_v.z * c1) * innorm > margin) continue;

                    Vec3f pt = (lua_v + lub_v * c1) * 0.5f;
                    vec3_set(pt, q, vec3_get(pt, q) + 0.5f * l);
                    vec3_set(pt, 1-q, vec3_get(pt, 1-q) + 0.5f * c2);
                    depth_arr[n] = pt.z * innorm * 2.f;
                    points[n] = pt;
                    n++;
                }
            }
        }

        // Face corners
        int nl = n;
        float det = pts_cn1.x * pts_cn2.y - pts_cn2.x * pts_cn1.y;
        float C_inv = (fabsf(det) > kBoxMinVal) ? (1.f / det) : 0.f;
        for (int ii = 0; ii < 4 && n < max_con_pair; ii++) {
            float llx = (ii / 2) ? lx : -lx;
            float lly = (ii % 2) ? ly : -ly;
            float x = llx - pts_lp.x;
            float y = lly - pts_lp.y;
            float u = (x * pts_cn2.y - y * pts_cn2.x) * C_inv;
            float v = (y * pts_cn1.x - x * pts_cn1.y) * C_inv;
            if (nl == 0) {
                if ((u < 0.f || u > 1.f) && (v < 0.f || v > 1.f)) continue;
            } else {
                if (u < 0.f || v < 0.f || u > 1.f || v > 1.f) continue;
            }
            if (u < 0.f) u = 0.f; if (u > 1.f) u = 1.f;
            if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
            Vec3f vtmp = pu[0] * (1.f - u - v) + pu[1] * u + pu[2] * v;
            Vec3f ptmp(llx, lly, 0.f);
            Vec3f diff = ptmp - vtmp;
            float tc1 = math::dot(diff, diff);
            if (vtmp.z > 0.f && tc1 > margin * margin) continue;
            ptmp = (ptmp + vtmp) * 0.5f;
            depth_arr[n] = sqrtf(tc1) * (vtmp.z < 0.f ? -1.f : 1.f);
            points[n] = ptmp;
            n++;
        }

        // Box2 vertices inside face
        for (int ii = 0; ii < 4 && n < max_con_pair; ii++) {
            float px = pu[ii].x, py = pu[ii].y;
            if (nl == 0 && n > nl) {
                if ((px < -lx || px > lx) && (py < -ly || py > ly)) continue;
            } else if (px < -lx || px > lx || py < -ly || py > ly) continue;

            float c1_v = 0.f;
            for (int jj = 0; jj < 2; jj++) {
                float puij = vec3_get(pu[ii], jj);
                float sj = vec3_get(s_e, jj);
                if (puij < -sj) c1_v += (puij + sj) * (puij + sj);
                else if (puij > sj) c1_v += (puij - sj) * (puij - sj);
            }
            c1_v += pu[ii].z * innorm * pu[ii].z * innorm;
            if (pu[ii].z > 0.f && c1_v > margin * margin) continue;

            Vec3f tmp_p(pu[ii].x, pu[ii].y, 0.f);
            for (int jj = 0; jj < 2; jj++) {
                float puij = vec3_get(pu[ii], jj);
                float sj = vec3_get(s_e, jj);
                if (puij < -sj) vec3_set(tmp_p, jj, -sj * 0.5f);
                else if (puij > sj) vec3_set(tmp_p, jj, sj * 0.5f);
            }
            tmp_p = tmp_p + pu[ii];
            points[n] = tmp_p * 0.5f;
            depth_arr[n] = sqrtf(c1_v) * (pu[ii].z < 0.f ? -1.f : 1.f);
            n++;
        }

        rw = box1_rot * math::transpose(rotmore);
        normal = (inv_flag ? -1.f : 1.f) * (rw * rnorm);
        hz_final = hz;
        pw_final = box1_pos;
    }

    // Final transform: add hz back, convert to world
    result.count = n;
    for (int i = 0; i < n; i++) {
        points[i].z += hz_final;
        result.pos[i] = rw * points[i] + pw_final;
        result.depth[i] = depth_arr[i];
    }
    result.normal = normal;
    return result;
}

// --- Main narrowphase kernel ------------------------------------------------

__global__ void narrow_phase_kernel(
    const math::Vec2i* __restrict__ pairs,
    int pair_count,
    const int*         __restrict__ shape_body,
    const int*         __restrict__ shape_geo_type,
    const Vec3f*       __restrict__ shape_geo_scale,
    const Vec3f*       __restrict__ shape_pos_local,
    const Quatf*       __restrict__ shape_quat_local,
    const float*       __restrict__ shape_ke,
    const float*       __restrict__ shape_kd,
    const float*       __restrict__ shape_mu,
    const float*       __restrict__ shape_gap,
    const Vec3f*       __restrict__ body_pos,
    const Quatf*       __restrict__ body_quat,
    const math::Vec2i* __restrict__ excluded_body_pairs,
    int                excluded_pair_count,
    int*         contact_shape0,
    int*         contact_shape1,
    Vec3f*       contact_point0,
    Vec3f*       contact_point1,
    Vec3f*       contact_normal,
    float*       contact_margin0,
    float*       contact_margin1,
    float*       contact_material_ke,
    float*       contact_material_kd,
    float*       contact_material_mu,
    int*         contact_count,
    int          contact_max)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    math::Vec2i pair = pairs[tid];
    int si = pair.x, sj = pair.y;

    // Filter: skip if bodies are in the excluded pairs list (joint-connected)
    int bi_raw = shape_body[si];
    int bj_raw = shape_body[sj];
    if (bi_raw >= 0 && bj_raw >= 0 && bi_raw != bj_raw && excluded_pair_count > 0) {
        int lo = (bi_raw < bj_raw) ? bi_raw : bj_raw;
        int hi = (bi_raw < bj_raw) ? bj_raw : bi_raw;
        // Binary search in sorted excluded_body_pairs
        int left = 0, right = excluded_pair_count - 1;
        bool found = false;
        while (left <= right) {
            int mid = (left + right) / 2;
            math::Vec2i ep = excluded_body_pairs[mid];
            if (ep.x == lo && ep.y == hi) { found = true; break; }
            if (ep.x < lo || (ep.x == lo && ep.y < hi)) left = mid + 1;
            else right = mid - 1;
        }
        if (found) return;
    }
    // Also skip same-body contacts
    if (bi_raw >= 0 && bi_raw == bj_raw) return;

    int gi = shape_geo_type[si];
    int gj = shape_geo_type[sj];
    Vec3f sci = shape_geo_scale[si];
    Vec3f scj = shape_geo_scale[sj];

    ShapePose spi = get_shape_pose(si, shape_body, body_pos, body_quat,
                                   shape_pos_local, shape_quat_local);
    ShapePose spj = get_shape_pose(sj, shape_body, body_pos, body_quat,
                                   shape_pos_local, shape_quat_local);

    // Sort so that gi <= gj for simpler dispatch.  The contact normal
    // convention is: n points such that body A is pushed in -n direction
    // and body B in +n direction for separation.
    bool swapped = false;
    if (gi > gj) {
        int ti = si; si = sj; sj = ti;
        int tg = gi; gi = gj; gj = tg;
        Vec3f ts = sci; sci = scj; scj = ts;
        ShapePose tp = spi; spi = spj; spj = tp;
        swapped = true;
    }

    ContactResult cr{};

    if (gi == GEO_SPHERE && gj == GEO_SPHERE) {
        cr = contact_sphere_sphere(spi, sci.x, spj, scj.x);
    } else if (gi == GEO_SPHERE && gj == GEO_BOX) {
        cr = contact_sphere_box(spi, sci.x, spj, scj);
    } else if (gi == GEO_SPHERE && gj == GEO_CAPSULE) {
        cr = contact_sphere_capsule(spi, sci.x, spj, scj.x, scj.y);
    } else if (gi == GEO_SPHERE && gj == GEO_PLANE) {
        cr = contact_sphere_plane(spi, sci.x, spj);
    } else if (gi == GEO_BOX && gj == GEO_PLANE) {
        cr = contact_box_plane(spi, sci, spj);
    } else if (gi == GEO_CAPSULE && gj == GEO_CAPSULE) {
        cr = contact_capsule_capsule(spi, sci.x, sci.y, spj, scj.x, scj.y);
    } else if (gi == GEO_CAPSULE && gj == GEO_PLANE) {
        cr = contact_capsule_plane(spi, sci.x, sci.y, spj);
    }

    // --- Multi-point contacts for box-plane and box-box ---
    bool is_box_plane = (gi == GEO_BOX && gj == GEO_PLANE);
    bool is_box_box = (gi == GEO_BOX && gj == GEO_BOX);

    if (is_box_plane || is_box_box) {
        float gap_sum = shape_gap[si] + shape_gap[sj];
        float ke_avg = 0.5f * (shape_ke[si] + shape_ke[sj]);
        float kd_avg = 0.5f * (shape_kd[si] + shape_kd[sj]);
        float mu_avg = 0.5f * (shape_mu[si] + shape_mu[sj]);

        int si_out = swapped ? sj : si;
        int sj_out = swapped ? si : sj;
        int bi_out = shape_body[si_out];
        int bj_out = shape_body[sj_out];
        float g0 = 0.f;
        float g1 = 0.f;

        if (is_box_plane) {
            Vec3f plane_n = quat_rotate(spj.rotation, Vec3f(0.f, 1.f, 0.f));
            Mat3f Rbox = quat_to_matrix(spi.rotation);
            for (int ix = -1; ix <= 1; ix += 2)
            for (int iy = -1; iy <= 1; iy += 2)
            for (int iz = -1; iz <= 1; iz += 2) {
                Vec3f lv(ix * sci.x, iy * sci.y, iz * sci.z);
                Vec3f wv = spi.center + Rbox * lv;
                float d = math::dot(wv - spj.center, plane_n);
                if (d >= gap_sum) continue;
                Vec3f n_out = swapped ? plane_n : -plane_n;
                Vec3f p0w = wv;
                Vec3f p1w = wv - plane_n * d;
                if (swapped) { Vec3f t = p0w; p0w = p1w; p1w = t; }
                Vec3f p0l = to_body_local(p0w, bi_out, body_pos, body_quat);
                Vec3f p1l = to_body_local(p1w, bj_out, body_pos, body_quat);
                int slot = atomicAdd(contact_count, 1);
                if (slot >= contact_max) return;
                contact_shape0[slot] = si_out;
                contact_shape1[slot] = sj_out;
                contact_point0[slot] = p0l;
                contact_point1[slot] = p1l;
                contact_normal[slot] = n_out;
                contact_margin0[slot] = g0;
                contact_margin1[slot] = g1;
                contact_material_ke[slot] = ke_avg;
                contact_material_kd[slot] = kd_avg;
                contact_material_mu[slot] = mu_avg;
            }
        } else {
            // Box-box: full SAT with clipped contact generation.
            // Use a large margin for SAT so deep penetrations are never
            // filtered out (broadphase already limits which pairs reach here).
            Mat3f Ra = quat_to_matrix(spi.rotation);
            Mat3f Rb = quat_to_matrix(spj.rotation);
            float sat_margin = fmaxf(gap_sum, 1.0f);
            BoxBoxResult bbr = collide_box_box_sat(
                spi.center, Ra, sci,
                spj.center, Rb, scj,
                sat_margin);

            for (int ic = 0; ic < bbr.count; ic++) {
                Vec3f n_out = bbr.normal;
                Vec3f contact_world = bbr.pos[ic];
                float d = bbr.depth[ic];
                // SAT depth: negative=penetrating, positive=separated.
                // p0 on body_a surface, p1 on body_b surface:
                Vec3f p0w = contact_world - n_out * (d * 0.5f);
                Vec3f p1w = contact_world + n_out * (d * 0.5f);

                if (swapped) { n_out = -n_out; Vec3f t = p0w; p0w = p1w; p1w = t; }
                Vec3f p0l = to_body_local(p0w, bi_out, body_pos, body_quat);
                Vec3f p1l = to_body_local(p1w, bj_out, body_pos, body_quat);
                int slot = atomicAdd(contact_count, 1);
                if (slot >= contact_max) return;
                contact_shape0[slot] = si_out;
                contact_shape1[slot] = sj_out;
                contact_point0[slot] = p0l;
                contact_point1[slot] = p1l;
                contact_normal[slot] = n_out;
                contact_margin0[slot] = g0;
                contact_margin1[slot] = g1;
                contact_material_ke[slot] = ke_avg;
                contact_material_kd[slot] = kd_avg;
                contact_material_mu[slot] = mu_avg;
            }
        }
        return;
    }

    if (!cr.valid) return;

    // Check gap sum
    float gap_sum = shape_gap[si] + shape_gap[sj];
    if (cr.depth < -gap_sum) return;

    // Unswap if needed
    if (swapped) {
        int t = si; si = sj; sj = t;
        cr.normal = -cr.normal;
        Vec3f tp = cr.p0_local; cr.p0_local = cr.p1_local; cr.p1_local = tp;
    }

    // Convert contact points to body-local frames
    int bi = shape_body[si];
    int bj = shape_body[sj];
    Vec3f p0_local = to_body_local(cr.p0_local, bi, body_pos, body_quat);
    Vec3f p1_local = to_body_local(cr.p1_local, bj, body_pos, body_quat);

    int slot = atomicAdd(contact_count, 1);
    if (slot >= contact_max) return;

    contact_shape0[slot] = si;
    contact_shape1[slot] = sj;
    contact_point0[slot] = p0_local;
    contact_point1[slot] = p1_local;
    contact_normal[slot] = cr.normal;
    contact_margin0[slot] = shape_gap[si];
    contact_margin1[slot] = shape_gap[sj];
    contact_material_ke[slot] = 0.5f * (shape_ke[si] + shape_ke[sj]);
    contact_material_kd[slot] = 0.5f * (shape_kd[si] + shape_kd[sj]);
    contact_material_mu[slot] = 0.5f * (shape_mu[si] + shape_mu[sj]);
}

__global__ void clear_int_kernel(int* p) { *p = 0; }

}  // namespace

void narrow_phase_detect(
    const math::Vec2i* pair_list,
    int                pair_count,
    const int*         shape_body,
    const int*         shape_geo_type,
    const math::Vec3f* shape_geo_scale,
    const math::Vec3f* shape_pos_local,
    const math::Quatf* shape_quat_local,
    const float*       shape_ke,
    const float*       shape_kd,
    const float*       shape_mu,
    const float*       shape_gap,
    const math::Vec3f* body_pos,
    const math::Quatf* body_quat,
    const math::Vec2i* excluded_body_pairs,
    int                excluded_pair_count,
    int*         contact_shape0,
    int*         contact_shape1,
    math::Vec3f* contact_point0,
    math::Vec3f* contact_point1,
    math::Vec3f* contact_normal,
    float*       contact_margin0,
    float*       contact_margin1,
    float*       contact_material_ke,
    float*       contact_material_kd,
    float*       contact_material_mu,
    int*         contact_count,
    int          contact_max,
    std::uintptr_t cuda_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cuda_stream);

    clear_int_kernel<<<1, 1, 0, s>>>(contact_count);

    if (pair_count <= 0) return;

    int blocks = (pair_count + kBlock - 1) / kBlock;
    narrow_phase_kernel<<<blocks, kBlock, 0, s>>>(
        pair_list, pair_count,
        shape_body, shape_geo_type, shape_geo_scale,
        shape_pos_local, shape_quat_local,
        shape_ke, shape_kd, shape_mu, shape_gap,
        body_pos, body_quat,
        excluded_body_pairs, excluded_pair_count,
        contact_shape0, contact_shape1,
        contact_point0, contact_point1, contact_normal,
        contact_margin0, contact_margin1,
        contact_material_ke, contact_material_kd, contact_material_mu,
        contact_count, contact_max);
}

}  // namespace rigid
}  // namespace chysx
