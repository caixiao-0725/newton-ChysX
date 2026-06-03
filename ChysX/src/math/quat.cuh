// SPDX-License-Identifier: Apache-2.0
//
// Quaternion helpers for rigid-body dynamics, usable from host and device.
//
// Convention: q = (x, y, z, w) where w is the scalar part.  This matches
// the layout used by NVIDIA Warp and Newton so interop is straightforward.
// The identity quaternion is (0, 0, 0, 1).
//
// All functions are header-only CHYSX_HDI so they can be inlined into
// CUDA kernels and C++ host code alike.

#pragma once

#include "common.cuh"
#include "matrix.cuh"
#include "vec.cuh"

#include <cmath>

namespace chysx {
namespace math {

using Quatf = Vec4f;

// --- Construction -----------------------------------------------------------

CHYSX_HDI Quatf quat_identity() { return Quatf(0.f, 0.f, 0.f, 1.f); }

CHYSX_HDI Quatf quat_from_axis_angle(const Vec3f& axis, float angle) {
    using std::sin;
    using std::cos;
    float ha = 0.5f * angle;
    float s = sin(ha);
    return Quatf(axis.x * s, axis.y * s, axis.z * s, cos(ha));
}

// --- Arithmetic -------------------------------------------------------------

CHYSX_HDI Quatf quat_multiply(const Quatf& a, const Quatf& b) {
    return Quatf(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

CHYSX_HDI Quatf quat_conjugate(const Quatf& q) {
    return Quatf(-q.x, -q.y, -q.z, q.w);
}

CHYSX_HDI Quatf quat_inverse(const Quatf& q) {
    float inv_n2 = 1.f / dot(q, q);
    Quatf c = quat_conjugate(q);
    return c * inv_n2;
}

CHYSX_HDI Quatf quat_normalize(const Quatf& q) {
    float n = length(q);
    return (n > 1e-12f) ? q * (1.f / n) : quat_identity();
}

// --- Rotation ---------------------------------------------------------------

CHYSX_HDI Vec3f quat_rotate(const Quatf& q, const Vec3f& v) {
    Vec3f u(q.x, q.y, q.z);
    float s = q.w;
    return 2.f * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.f * s * cross(u, v);
}

CHYSX_HDI Vec3f quat_rotate_inv(const Quatf& q, const Vec3f& v) {
    return quat_rotate(quat_conjugate(q), v);
}

// --- Conversion to/from Mat3 ------------------------------------------------

CHYSX_HDI Mat3f quat_to_matrix(const Quatf& q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
    return Mat3f(
        1.f - (yy + zz), xy - wz,         xz + wy,
        xy + wz,          1.f - (xx + zz), yz - wx,
        xz - wy,          yz + wx,         1.f - (xx + yy));
}

// --- Velocity ---------------------------------------------------------------

// Angular velocity from quaternion finite difference: omega = 2/dt * Im(dq * q_prev^-1)
// where dq = q_cur * conj(q_prev).  Returns world-frame angular velocity.
CHYSX_HDI Vec3f quat_velocity(const Quatf& q_cur, const Quatf& q_prev, float dt) {
    Quatf dq = quat_multiply(q_cur, quat_conjugate(q_prev));
    if (dq.w < 0.f) {
        dq = -dq;
    }
    float inv_dt = 1.f / dt;
    return Vec3f(2.f * dq.x * inv_dt,
                 2.f * dq.y * inv_dt,
                 2.f * dq.z * inv_dt);
}

// Integrate quaternion by angular velocity: q_new = normalize(q + 0.5*dt*omega_quat*q)
// where omega_quat = (omega.x, omega.y, omega.z, 0).
CHYSX_HDI Quatf quat_integrate(const Quatf& q, const Vec3f& omega, float dt) {
    Quatf omega_q(omega.x, omega.y, omega.z, 0.f);
    Quatf dq = quat_multiply(omega_q, q) * (0.5f * dt);
    return quat_normalize(q + dq);
}

// Small-angle rotation: q_new ≈ normalize(q + 0.5 * (theta.x, theta.y, theta.z, 0) * q)
// Used by AVBD's position-level rotation update.
CHYSX_HDI Quatf quat_apply_small_rotation(const Quatf& q, const Vec3f& theta) {
    Quatf dq(theta.x, theta.y, theta.z, 0.f);
    Quatf half_dq_q = quat_multiply(dq, q) * 0.5f;
    return quat_normalize(q + half_dq_q);
}

// --- Skew-symmetric matrix (for r × f = skew(r) * f) -----------------------

CHYSX_HDI Mat3f skew(const Vec3f& v) {
    return Mat3f(
         0.f, -v.z,  v.y,
         v.z,  0.f, -v.x,
        -v.y,  v.x,  0.f);
}

// --- Transform helpers (pos + quat) ----------------------------------------

CHYSX_HDI Vec3f transform_point(const Vec3f& pos, const Quatf& quat,
                                const Vec3f& p) {
    return pos + quat_rotate(quat, p);
}

}  // namespace math
}  // namespace chysx
