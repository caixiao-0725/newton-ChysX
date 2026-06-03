// SPDX-License-Identifier: Apache-2.0
//
// Spatial algebra primitives for rigid-body dynamics (Featherstone).
//
// Convention (matches Newton / Warp):
//   SpatialVector = (v, omega)  —  linear first, angular second
//   SpatialMatrix = 6x6 row-major
//   Transform7    = (px,py,pz, qx,qy,qz,qw)  — Warp layout
//
// All functions are device-inlineable and usable from host code.

#pragma once

#include "../../math/common.cuh"
#include "../../math/matrix.cuh"
#include "../../math/quat.cuh"
#include "../../math/vec.cuh"

namespace chysx {
namespace rigid {

using math::Vec3f;
using math::Vec4f;
using math::Mat3f;
using math::Quatf;
using math::quat_identity;
using math::quat_multiply;
using math::quat_rotate;
using math::quat_rotate_inv;
using math::quat_from_axis_angle;
using math::quat_normalize;
using math::quat_conjugate;
using math::quat_inverse;
using math::skew;
using math::cross;
using math::dot;
using math::length;

// ============================================================================
// Transform7: rigid transform as (position, quaternion)
// Layout: [px, py, pz, qx, qy, qz, qw]   (matches Warp wp.transform)
// ============================================================================

struct Transform7 {
    Vec3f p;
    Quatf q;

    CHYSX_HD Transform7() : p(), q(quat_identity()) {}
    CHYSX_HD Transform7(Vec3f p_, Quatf q_) : p(p_), q(q_) {}
};

CHYSX_HDI Transform7 tf_identity() {
    return Transform7(Vec3f(), quat_identity());
}

CHYSX_HDI Transform7 tf_inverse(const Transform7& t) {
    Quatf q_inv = quat_conjugate(t.q);
    Vec3f p_inv = -quat_rotate(q_inv, t.p);
    return Transform7(p_inv, q_inv);
}

CHYSX_HDI Transform7 tf_multiply(const Transform7& a, const Transform7& b) {
    Quatf q = quat_multiply(a.q, b.q);
    Vec3f p = a.p + quat_rotate(a.q, b.p);
    return Transform7(p, q);
}

CHYSX_HDI Vec3f tf_get_translation(const Transform7& t) {
    return t.p;
}

CHYSX_HDI Quatf tf_get_rotation(const Transform7& t) {
    return t.q;
}

CHYSX_HDI Vec3f tf_point(const Transform7& t, const Vec3f& p) {
    return t.p + quat_rotate(t.q, p);
}

CHYSX_HDI Vec3f tf_vector(const Transform7& t, const Vec3f& v) {
    return quat_rotate(t.q, v);
}

// Load a Transform7 from a float[7] array at offset
CHYSX_DI Transform7 load_transform7(const float* ptr, int idx) {
    const float* base = ptr + idx * 7;
    Vec3f p(base[0], base[1], base[2]);
    Quatf q(base[3], base[4], base[5], base[6]);
    return Transform7(p, q);
}

// ============================================================================
// SpatialVector: 6D vector (v_linear, omega_angular)
// Convention: top 3 = linear (v), bottom 3 = angular (omega)
// ============================================================================

struct SpatialVector {
    float data[6];

    CHYSX_HD SpatialVector() : data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f} {}

    CHYSX_HD SpatialVector(float v0, float v1, float v2,
                           float w0, float w1, float w2)
        : data{v0, v1, v2, w0, w1, w2} {}

    CHYSX_HD SpatialVector(Vec3f v, Vec3f w)
        : data{v.x, v.y, v.z, w.x, w.y, w.z} {}

    CHYSX_HD float& operator[](int i) { return data[i]; }
    CHYSX_HD const float& operator[](int i) const { return data[i]; }

    CHYSX_HD Vec3f top() const { return Vec3f(data[0], data[1], data[2]); }
    CHYSX_HD Vec3f bottom() const { return Vec3f(data[3], data[4], data[5]); }
    CHYSX_HD Vec3f linear() const { return top(); }
    CHYSX_HD Vec3f angular() const { return bottom(); }
};

CHYSX_HDI SpatialVector operator+(const SpatialVector& a, const SpatialVector& b) {
    return SpatialVector(
        a[0]+b[0], a[1]+b[1], a[2]+b[2],
        a[3]+b[3], a[4]+b[4], a[5]+b[5]);
}

CHYSX_HDI SpatialVector operator-(const SpatialVector& a, const SpatialVector& b) {
    return SpatialVector(
        a[0]-b[0], a[1]-b[1], a[2]-b[2],
        a[3]-b[3], a[4]-b[4], a[5]-b[5]);
}

CHYSX_HDI SpatialVector operator-(const SpatialVector& a) {
    return SpatialVector(-a[0], -a[1], -a[2], -a[3], -a[4], -a[5]);
}

CHYSX_HDI SpatialVector operator*(const SpatialVector& a, float s) {
    return SpatialVector(
        a[0]*s, a[1]*s, a[2]*s,
        a[3]*s, a[4]*s, a[5]*s);
}

CHYSX_HDI SpatialVector operator*(float s, const SpatialVector& a) {
    return a * s;
}

CHYSX_HDI SpatialVector& operator+=(SpatialVector& a, const SpatialVector& b) {
    for (int i = 0; i < 6; ++i) a[i] += b[i];
    return a;
}

CHYSX_HDI float spatial_dot(const SpatialVector& a, const SpatialVector& b) {
    float s = 0.f;
    for (int i = 0; i < 6; ++i) s += a[i] * b[i];
    return s;
}

CHYSX_HDI float spatial_length(const SpatialVector& v) {
    return sqrtf(spatial_dot(v, v));
}

// ============================================================================
// SpatialMatrix: 6x6 row-major
// ============================================================================

struct SpatialMatrix {
    float data[36];

    CHYSX_HD SpatialMatrix() : data{} {}

    CHYSX_HD SpatialMatrix(
        float m00, float m01, float m02, float m03, float m04, float m05,
        float m10, float m11, float m12, float m13, float m14, float m15,
        float m20, float m21, float m22, float m23, float m24, float m25,
        float m30, float m31, float m32, float m33, float m34, float m35,
        float m40, float m41, float m42, float m43, float m44, float m45,
        float m50, float m51, float m52, float m53, float m54, float m55)
        : data{m00,m01,m02,m03,m04,m05,
               m10,m11,m12,m13,m14,m15,
               m20,m21,m22,m23,m24,m25,
               m30,m31,m32,m33,m34,m35,
               m40,m41,m42,m43,m44,m45,
               m50,m51,m52,m53,m54,m55} {}

    CHYSX_HD float& operator()(int r, int c) { return data[r * 6 + c]; }
    CHYSX_HD const float& operator()(int r, int c) const { return data[r * 6 + c]; }
};

CHYSX_HDI SpatialMatrix operator+(const SpatialMatrix& a, const SpatialMatrix& b) {
    SpatialMatrix r;
    for (int i = 0; i < 36; ++i) r.data[i] = a.data[i] + b.data[i];
    return r;
}

CHYSX_HDI SpatialMatrix operator*(const SpatialMatrix& a, float s) {
    SpatialMatrix r;
    for (int i = 0; i < 36; ++i) r.data[i] = a.data[i] * s;
    return r;
}

// Matrix-vector multiply: M * v
CHYSX_HDI SpatialVector operator*(const SpatialMatrix& M, const SpatialVector& v) {
    SpatialVector r;
    for (int i = 0; i < 6; ++i) {
        float s = 0.f;
        for (int j = 0; j < 6; ++j)
            s += M(i, j) * v[j];
        r[i] = s;
    }
    return r;
}

// Matrix-matrix multiply: A * B
CHYSX_HDI SpatialMatrix operator*(const SpatialMatrix& A, const SpatialMatrix& B) {
    SpatialMatrix r;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.f;
            for (int k = 0; k < 6; ++k)
                s += A(i, k) * B(k, j);
            r(i, j) = s;
        }
    }
    return r;
}

CHYSX_HDI SpatialMatrix spatial_transpose(const SpatialMatrix& M) {
    SpatialMatrix r;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            r(i, j) = M(j, i);
    return r;
}

// ============================================================================
// Spatial algebra operations
// ============================================================================

// spatial_cross: motion × motion
// a = (v_a, w_a), b = (v_b, w_b)
// result.w = w_a × w_b
// result.v = w_a × v_b + v_a × w_b
CHYSX_HDI SpatialVector spatial_cross(const SpatialVector& a, const SpatialVector& b) {
    Vec3f w_a = a.angular();
    Vec3f v_a = a.linear();
    Vec3f w_b = b.angular();
    Vec3f v_b = b.linear();

    Vec3f w = cross(w_a, w_b);
    Vec3f v = cross(w_a, v_b) + cross(v_a, w_b);
    return SpatialVector(v, w);
}

// spatial_cross_dual: motion ×* force
// a = (v_a, w_a), b = (f_b, tau_b)
// result.tau = w_a × tau_b + v_a × f_b
// result.f   = w_a × f_b
CHYSX_HDI SpatialVector spatial_cross_dual(const SpatialVector& a, const SpatialVector& b) {
    Vec3f w_a = a.angular();
    Vec3f v_a = a.linear();
    Vec3f w_b = b.angular();
    Vec3f v_b = b.linear();

    Vec3f w = cross(w_a, w_b) + cross(v_a, v_b);
    Vec3f v = cross(w_a, v_b);
    return SpatialVector(v, w);
}

// ============================================================================
// Transform a spatial twist between coordinate frames
// Newton convention: spatial_vector = (v, omega)
//
// For rigid transform t = (R, p) from source to destination:
//   omega' = R * omega
//   v'     = R * v + p × omega'
// ============================================================================

CHYSX_HDI SpatialVector transform_twist(const Transform7& t, const SpatialVector& x) {
    Vec3f v = x.linear();
    Vec3f w = x.angular();

    Vec3f w_new = quat_rotate(t.q, w);
    Vec3f v_new = quat_rotate(t.q, v) + cross(t.p, w_new);
    return SpatialVector(v_new, w_new);
}

// velocity_at_point: v_p = v + omega × r
CHYSX_HDI Vec3f velocity_at_point(const SpatialVector& qd, const Vec3f& r) {
    return qd.linear() + cross(qd.angular(), r);
}

// ============================================================================
// Transform spatial inertia tensor to a new coordinate frame
//   I_new = adj_t^{-T} * I * adj_t^{-1}
// ============================================================================

CHYSX_HDI SpatialMatrix transform_spatial_inertia(const Transform7& t, const SpatialMatrix& I) {
    Transform7 t_inv = tf_inverse(t);
    Quatf q = tf_get_rotation(t_inv);
    Vec3f p = tf_get_translation(t_inv);

    Vec3f r1 = quat_rotate(q, Vec3f(1.f, 0.f, 0.f));
    Vec3f r2 = quat_rotate(q, Vec3f(0.f, 1.f, 0.f));
    Vec3f r3 = quat_rotate(q, Vec3f(0.f, 0.f, 1.f));

    // R = [r1 r2 r3] as column vectors → Mat3 row-major
    Mat3f R(r1.x, r2.x, r3.x,
            r1.y, r2.y, r3.y,
            r1.z, r2.z, r3.z);

    Mat3f S = skew(p) * R;

    // Build the 6x6 spatial transformation matrix T:
    // T = [ R   S ]
    //     [ 0   R ]
    SpatialMatrix T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T(i, j)     = R(i, j);
            T(i, j + 3) = S(i, j);
            T(i + 3, j) = 0.f;
            T(i + 3, j + 3) = R(i, j);
        }
    }

    // I_new = T^T * I * T
    return spatial_transpose(T) * I * T;
}

// ============================================================================
// Build spatial inertia from mass and 3x3 inertia tensor
// Layout: I_m = [ m*eye3   0    ]
//               [   0    Inertia ]
// ============================================================================

CHYSX_HDI SpatialMatrix build_spatial_inertia(const Mat3f& inertia, float mass) {
    SpatialMatrix I;
    for (int i = 0; i < 3; ++i) {
        I(i, i) = mass;
        for (int j = 0; j < 3; ++j)
            I(i + 3, j + 3) = inertia(i, j);
    }
    return I;
}

// ============================================================================
// Conversions between COM-twist and origin-twist
// ============================================================================

// origin twist → COM twist
CHYSX_HDI SpatialVector origin_twist_to_com_twist(
    const SpatialVector& qd, const Transform7& X_wb, const Vec3f& body_com
) {
    Vec3f omega = qd.angular();
    Vec3f r_com_world = tf_vector(X_wb, body_com);
    Vec3f v_com = velocity_at_point(qd, r_com_world);
    return SpatialVector(v_com, omega);
}

// COM twist → origin twist
CHYSX_HDI SpatialVector com_twist_to_origin_twist(
    const SpatialVector& qd, const Transform7& X_wb, const Vec3f& body_com
) {
    Vec3f omega = qd.angular();
    Vec3f v_origin = qd.linear() - cross(omega, tf_vector(X_wb, body_com));
    return SpatialVector(v_origin, omega);
}

// COM twist → point velocity
CHYSX_HDI Vec3f com_twist_to_point_velocity(
    const SpatialVector& qd, const Transform7& X_wb,
    const Vec3f& body_com, const Vec3f& point
) {
    return velocity_at_point(qd, point - tf_point(X_wb, body_com));
}

// ============================================================================
// Quaternion from 3x3 rotation matrix (column vectors form)
// ============================================================================

CHYSX_HDI Quatf quat_from_matrix(const Mat3f& m) {
    float tr = m(0,0) + m(1,1) + m(2,2);
    if (tr > 0.f) {
        float s = sqrtf(tr + 1.f) * 2.f;
        return Quatf(
            (m(2,1) - m(1,2)) / s,
            (m(0,2) - m(2,0)) / s,
            (m(1,0) - m(0,1)) / s,
            0.25f * s);
    }
    if (m(0,0) > m(1,1) && m(0,0) > m(2,2)) {
        float s = sqrtf(1.f + m(0,0) - m(1,1) - m(2,2)) * 2.f;
        return Quatf(
            0.25f * s,
            (m(0,1) + m(1,0)) / s,
            (m(0,2) + m(2,0)) / s,
            (m(2,1) - m(1,2)) / s);
    }
    if (m(1,1) > m(2,2)) {
        float s = sqrtf(1.f + m(1,1) - m(0,0) - m(2,2)) * 2.f;
        return Quatf(
            (m(0,1) + m(1,0)) / s,
            0.25f * s,
            (m(1,2) + m(2,1)) / s,
            (m(0,2) - m(2,0)) / s);
    }
    float s = sqrtf(1.f + m(2,2) - m(0,0) - m(1,1)) * 2.f;
    return Quatf(
        (m(0,2) + m(2,0)) / s,
        (m(1,2) + m(2,1)) / s,
        0.25f * s,
        (m(1,0) - m(0,1)) / s);
}

// Build rotation matrix from 3 column vectors
CHYSX_HDI Mat3f mat3_from_cols(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2) {
    return Mat3f(
        c0.x, c1.x, c2.x,
        c0.y, c1.y, c2.y,
        c0.z, c1.z, c2.z);
}

}  // namespace rigid
}  // namespace chysx
