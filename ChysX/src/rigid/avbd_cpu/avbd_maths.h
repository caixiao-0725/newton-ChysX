// SPDX-License-Identifier: MIT
// Math types and utilities for the AVBD rigid-body solver.
// Adapted from avbd-demo3d (Chris Giles, 2026).

#pragma once

#include <cmath>
#include <cfloat>

namespace chysx {
namespace avbd {

struct float2 {
    float x, y;
    float& operator[](int i) { return ((float*)this)[i]; }
    const float& operator[](int i) const { return ((float*)this)[i]; }
};

struct float3 {
    float x, y, z;
    float2& xy() { return *(float2*)this; }
    const float2& xy() const { return *(float2*)this; }
    float& operator[](int i) { return ((float*)this)[i]; }
    const float& operator[](int i) const { return ((float*)this)[i]; }
};

struct quat {
    float x, y, z, w;
    float3& vec() { return *(float3*)this; }
    const float3& vec() const { return *(float3*)this; }
    float& operator[](int i) { return ((float*)this)[i]; }
    const float& operator[](int i) const { return ((float*)this)[i]; }
};

struct float2x2 {
    float2 row[2];
    float2& operator[](int i) { return row[i]; }
    const float2& operator[](int i) const { return row[i]; }
    float2 col(int i) const { return float2{row[0][i], row[1][i]}; }
};

struct float3x3 {
    float3 row[3];
    float3& operator[](int i) { return row[i]; }
    const float3& operator[](int i) const { return row[i]; }
    float3 col(int i) const { return float3{row[0][i], row[1][i], row[2][i]}; }
};

// ---- float2 operators ----

inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

inline float2 operator+=(float2& a, float2 b) { a.x += b.x; a.y += b.y; return a; }
inline float2 operator-=(float2& a, float2 b) { a.x -= b.x; a.y -= b.y; return a; }
inline float2 operator-(float2 v) { return {-v.x, -v.y}; }
inline float2 operator+(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
inline float2 operator-(float2 a, float2 b) { return {a.x - b.x, a.y - b.y}; }
inline float2 operator*(float2 a, float b) { return {a.x * b, a.y * b}; }
inline float2 operator/(float2 a, float b) { return {a.x / b, a.y / b}; }
inline float2 operator*(float2x2 a, float2 b) { return {dot(a[0], b), dot(a[1], b)}; }

// ---- float3 operators ----

inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline float3& operator-=(float3& a, float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline float3 operator-(float3 v) { return {-v.x, -v.y, -v.z}; }
inline float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float3 operator*(float3 a, float b) { return {a.x * b, a.y * b, a.z * b}; }
inline float3 operator/(float3 a, float b) { return {a.x / b, a.y / b, a.z / b}; }
inline float3 operator*(float3x3 a, float3 b) { return {dot(a[0], b), dot(a[1], b), dot(a[2], b)}; }

// ---- float2x2 operators ----

inline float2x2 operator+(float2x2 a, float2x2 b) { return {a[0] + b[0], a[1] + b[1]}; }
inline float2x2 operator-(float2x2 a, float2x2 b) { return {a[0] - b[0], a[1] - b[1]}; }
inline float2x2 operator*(float2x2 a, float b) { return {a[0] * b, a[1] * b}; }
inline float2x2 operator/(float2x2 a, float b) { return {a[0] / b, a[1] / b}; }
inline float2x2 operator*(float2x2 a, float2x2 b) {
    return {
        float2{dot(a.row[0], b.col(0)), dot(a.row[0], b.col(1))},
        float2{dot(a.row[1], b.col(0)), dot(a.row[1], b.col(1))}};
}

// ---- float3x3 operators ----

inline float3x3& operator+=(float3x3& a, float3x3 b) { a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; return a; }
inline float3x3 operator+(float3x3 a, float3x3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline float3x3 operator-(float3x3 a, float3x3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline float3x3 operator*(float3x3 a, float b) { return {a[0] * b, a[1] * b, a[2] * b}; }
inline float3x3 operator/(float3x3 a, float b) { return {a[0] / b, a[1] / b, a[2] / b}; }
inline float3x3 operator-(float3x3 a) { return {-a.row[0], -a.row[1], -a.row[2]}; }
inline float3x3 operator*(float3x3 a, float3x3 b) {
    return {
        float3{dot(a.row[0], b.col(0)), dot(a.row[0], b.col(1)), dot(a.row[0], b.col(2))},
        float3{dot(a.row[1], b.col(0)), dot(a.row[1], b.col(1)), dot(a.row[1], b.col(2))},
        float3{dot(a.row[2], b.col(0)), dot(a.row[2], b.col(1)), dot(a.row[2], b.col(2))}};
}

// ---- quat operators ----

inline float lengthSq(quat q) { return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; }
inline float length(quat q) { return std::sqrt(lengthSq(q)); }
inline quat operator*(quat a, float b) { return {a.x * b, a.y * b, a.z * b, a.w * b}; }
inline quat operator/(quat a, float b) { return {a.x / b, a.y / b, a.z / b, a.w / b}; }
inline quat conjugate(quat q) { return {-q.x, -q.y, -q.z, q.w}; }
inline quat inverse(quat q) { return conjugate(q) / lengthSq(q); }
inline quat normalize(quat q) { return q / length(q); }

inline quat operator*(quat a, quat b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline quat operator+(quat a, quat b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }

// quat difference -> angular error (vec3)
inline float3 operator-(quat a, quat b) { return (a * inverse(b)).vec() * 2.0f; }

// quat + angular increment
inline quat operator+(quat a, float3 b) { return normalize(a + quat{b.x, b.y, b.z, 0} * a * 0.5f); }

// ---- scalar math ----

inline float rad(float deg) { return deg * 0.01745329251994329577f; }
inline float sign(float x) { return x < 0 ? -1.0f : x > 0 ? 1.0f : 0.0f; }

inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline float3 min(float3 a, float b) { return {min(a.x, b), min(a.y, b), min(a.z, b)}; }
inline float clamp(float x, float a, float b) { return max(a, min(b, x)); }
inline float3 clamp(float3 v, float a, float b) { return {clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)}; }

// ---- vector math ----

inline float lengthSq(float2 v) { return dot(v, v); }
inline float length(float2 v) { return std::sqrt(lengthSq(v)); }
inline float lengthSq(float3 v) { return dot(v, v); }
inline float length(float3 v) { return std::sqrt(lengthSq(v)); }
inline float3 normalize(float3 v) { return v / length(v); }

inline float cross(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
inline float3 cross(float3 a, float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float3x3 skew(float3 r) {
    return float3x3{0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0};
}

inline float2x2 outer(float2 a, float2 b) { return {b * a.x, b * a.y}; }
inline float3x3 outer(float3 a, float3 b) { return {b * a.x, b * a.y, b * a.z}; }

inline float2 abs(float2 v) { return {std::fabs(v.x), std::fabs(v.y)}; }
inline float3 abs(float3 v) { return {std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)}; }
inline float2x2 abs(float2x2 a) { return {abs(a[0]), abs(a[1])}; }

inline float2x2 transpose(float2x2 a) { return {float2{a[0][0], a[1][0]}, float2{a[0][1], a[1][1]}}; }
inline float3x3 transpose(float3x3 a) {
    return {float3{a[0][0], a[1][0], a[2][0]}, float3{a[0][1], a[1][1], a[2][1]}, float3{a[0][2], a[1][2], a[2][2]}};
}

inline float3x3 diagonal(float m00, float m11, float m22) {
    return float3x3{m00, 0, 0, 0, m11, 0, 0, 0, m22};
}

inline float3 rotate(quat angle, float3 v) {
    float3 u = {angle.x, angle.y, angle.z};
    float3 t = cross(u, v) * 2.0f;
    return v + t * angle.w + cross(u, t);
}

inline float3 transform(float3 qLin, quat qAng, float3 v) {
    return rotate(qAng, v) + qLin;
}

inline float3x3 rotation(quat q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    return float3x3{
        1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy),
        2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx),
        2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)};
}

inline float3x3 orthonormal(float3 normal) {
    float3 t1 = std::fabs(normal.x) > std::fabs(normal.z) ?
        float3{-normal.y, normal.x, 0} : float3{0, -normal.z, normal.y};
    t1 = normalize(t1);
    float3 t2 = cross(normal, t1);
    return float3x3{normal, t1, t2};
}

inline float3x3 diagonalize(float3x3 m) {
    return diagonal(length(m.col(0)), length(m.col(1)), length(m.col(2)));
}

// 6x6 LDL^T solver for coupled linear-angular system.
inline void solve(float3x3 aLin, float3x3 aAng, float3x3 aCross,
                  float3 bLin, float3 bAng, float3& xLin, float3& xAng) {
    float A11 = aLin[0][0];
    float A21 = aLin[1][0], A22 = aLin[1][1];
    float A31 = aLin[2][0], A32 = aLin[2][1], A33 = aLin[2][2];
    float A41 = aCross[0][0], A42 = aCross[0][1], A43 = aCross[0][2], A44 = aAng[0][0];
    float A51 = aCross[1][0], A52 = aCross[1][1], A53 = aCross[1][2], A54 = aAng[1][0], A55 = aAng[1][1];
    float A61 = aCross[2][0], A62 = aCross[2][1], A63 = aCross[2][2], A64 = aAng[2][0], A65 = aAng[2][1], A66 = aAng[2][2];

    float L21 = A21 / A11;
    float L31 = A31 / A11;
    float L41 = A41 / A11;
    float L51 = A51 / A11;
    float L61 = A61 / A11;
    float D1 = A11;

    float D2 = A22 - L21 * L21 * D1;
    float L32 = (A32 - L21 * L31 * D1) / D2;
    float L42 = (A42 - L21 * L41 * D1) / D2;
    float L52 = (A52 - L21 * L51 * D1) / D2;
    float L62 = (A62 - L21 * L61 * D1) / D2;

    float D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
    float L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    float L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    float L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    float D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
    float L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    float L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    float D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    float L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    float D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

    float y1 = bLin[0];
    float y2 = bLin[1] - L21 * y1;
    float y3 = bLin[2] - L31 * y1 - L32 * y2;
    float y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;
    float z4 = y4 / D4;
    float z5 = y5 / D5;
    float z6 = y6 / D6;

    xAng[2] = z6;
    xAng[1] = z5 - L65 * xAng[2];
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];
}

}  // namespace avbd
}  // namespace chysx
