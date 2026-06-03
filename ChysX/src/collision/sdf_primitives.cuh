// SPDX-License-Identifier: Apache-2.0
//
// Analytic signed-distance fields and outward gradients for Newton-style
// primitive shapes.  Ported 1:1 from newton/_src/geometry/kernels.py and
// newton/_src/utils/heightfield.py.

#pragma once

#include "../math/common.cuh"
#include "../math/vec.cuh"

namespace chysx {
namespace collision {

using math::Vec2f;
using math::Vec3f;

// Axis convention: X = 0, Y = 1, Z = 2.

struct HeightfieldData {
    int data_offset;
    int nrow;
    int ncol;
    float hx;
    float hy;
    float min_z;
    float max_z;
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

CHYSX_DI Vec3f _sdf_point_to_z_up(Vec3f point, int up_axis) {
    if (up_axis == 0) {
        return Vec3f(point.y, point.z, point.x);
    }
    if (up_axis == 1) {
        return Vec3f(point.x, point.z, point.y);
    }
    return point;
}

CHYSX_DI Vec3f _sdf_vector_from_z_up(Vec3f v, int up_axis) {
    if (up_axis == 0) {
        return Vec3f(v.z, v.x, v.y);
    }
    if (up_axis == 1) {
        return Vec3f(v.x, v.z, v.y);
    }
    return v;
}

CHYSX_DI float _sdf_capped_cone_z(float bottom_radius, float top_radius,
                                  float half_height, Vec3f point_z_up) {
    const Vec2f q(math::length(Vec2f(point_z_up.x, point_z_up.y)), point_z_up.z);
    const Vec2f k1(top_radius, half_height);
    const Vec2f k2(top_radius - bottom_radius, 2.0f * half_height);

    Vec2f ca;
    if (q.y < 0.0f) {
        ca = Vec2f(q.x - math::min(q.x, bottom_radius),
                   math::abs(q.y) - half_height);
    } else {
        ca = Vec2f(q.x - math::min(q.x, top_radius),
                   math::abs(q.y) - half_height);
    }

    const float denom = math::dot(k2, k2);
    float t = 0.0f;
    if (denom > 0.0f) {
        t = math::clamp(math::dot(k1 - q, k2) / denom, 0.0f, 1.0f);
    }
    const Vec2f cb = q - k1 + k2 * t;

    float sd_sign = 1.0f;
    if (cb.x < 0.0f && ca.y < 0.0f) {
        sd_sign = -1.0f;
    }

    return sd_sign * sqrtf(math::min(math::dot(ca, ca), math::dot(cb, cb)));
}

CHYSX_DI void _heightfield_surface_query(const HeightfieldData& hfd,
                                         const float* elevation_data,
                                         Vec3f pos,
                                         float& d_plane,
                                         Vec3f& normal,
                                         float& lateral_dist_sq) {
    if (hfd.nrow <= 1 || hfd.ncol <= 1) {
        d_plane = 1.0e10f;
        normal = Vec3f(0.0f, 0.0f, 1.0f);
        lateral_dist_sq = 0.0f;
        return;
    }

    const float dx = 2.0f * hfd.hx / static_cast<float>(hfd.ncol - 1);
    const float dy = 2.0f * hfd.hy / static_cast<float>(hfd.nrow - 1);
    const float z_range = hfd.max_z - hfd.min_z;

    const float cx = math::clamp(pos.x, -hfd.hx, hfd.hx);
    const float cy = math::clamp(pos.y, -hfd.hy, hfd.hy);
    const float out_x = pos.x - cx;
    const float out_y = pos.y - cy;
    lateral_dist_sq = out_x * out_x + out_y * out_y;

    float col_f = (cx + hfd.hx) / dx;
    float row_f = (cy + hfd.hy) / dy;
    col_f = math::clamp(col_f, 0.0f, static_cast<float>(hfd.ncol - 1));
    row_f = math::clamp(row_f, 0.0f, static_cast<float>(hfd.nrow - 1));

    int col = static_cast<int>(col_f);
    int row = static_cast<int>(row_f);
    if (col > hfd.ncol - 2) {
        col = hfd.ncol - 2;
    }
    if (row > hfd.nrow - 2) {
        row = hfd.nrow - 2;
    }
    const float fx = col_f - static_cast<float>(col);
    const float fy = row_f - static_cast<float>(row);

    const int base = hfd.data_offset;
    const float h00 = hfd.min_z + elevation_data[base + row * hfd.ncol + col] * z_range;
    const float h10 = hfd.min_z + elevation_data[base + row * hfd.ncol + col + 1] * z_range;
    const float h01 = hfd.min_z + elevation_data[base + (row + 1) * hfd.ncol + col] * z_range;
    const float h11 = hfd.min_z + elevation_data[base + (row + 1) * hfd.ncol + col + 1] * z_range;

    const float x0 = -hfd.hx + static_cast<float>(col) * dx;
    const float y0 = -hfd.hy + static_cast<float>(row) * dy;

    Vec3f v0(x0, y0, h00);
    Vec3f e1;
    Vec3f e2;
    if (fx >= fy) {
        e1 = Vec3f(dx, 0.0f, h10 - h00);
        e2 = Vec3f(dx, dy, h11 - h00);
    } else {
        e1 = Vec3f(dx, dy, h11 - h00);
        e2 = Vec3f(0.0f, dy, h01 - h00);
    }

    normal = math::normalize(math::cross(e1, e2));
    d_plane = math::dot(pos - v0, normal);
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_sphere(Vec3f point, float radius) {
    return math::length(point) - radius;
}

CHYSX_DI Vec3f sdf_sphere_grad(Vec3f point, float /*radius*/) {
    constexpr float eps = 1.0e-8f;
    const float p_len = math::length(point);
    if (p_len > eps) {
        return point / p_len;
    }
    return Vec3f(0.0f, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_box(Vec3f point, float hx, float hy, float hz) {
    const float qx = math::abs(point.x) - hx;
    const float qy = math::abs(point.y) - hy;
    const float qz = math::abs(point.z) - hz;

    const Vec3f e(math::max(qx, 0.0f), math::max(qy, 0.0f), math::max(qz, 0.0f));

    return math::length(e) + math::min(math::max(qx, math::max(qy, qz)), 0.0f);
}

CHYSX_DI Vec3f sdf_box_grad(Vec3f point, float hx, float hy, float hz) {
    const float qx = math::abs(point.x) - hx;
    const float qy = math::abs(point.y) - hy;
    const float qz = math::abs(point.z) - hz;

    if (qx > 0.0f || qy > 0.0f || qz > 0.0f) {
        const float x = math::clamp(point.x, -hx, hx);
        const float y = math::clamp(point.y, -hy, hy);
        const float z = math::clamp(point.z, -hz, hz);
        return math::normalize(point - Vec3f(x, y, z));
    }

    const float sx = static_cast<float>(math::sign(point.x));
    const float sy = static_cast<float>(math::sign(point.y));
    const float sz = static_cast<float>(math::sign(point.z));

    if ((qx > qy && qx > qz) || (qy == 0.0f && qz == 0.0f)) {
        return Vec3f(sx, 0.0f, 0.0f);
    }
    if ((qy > qx && qy > qz) || (qx == 0.0f && qz == 0.0f)) {
        return Vec3f(0.0f, sy, 0.0f);
    }
    return Vec3f(0.0f, 0.0f, sz);
}

// ---------------------------------------------------------------------------
// Capsule
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_capsule(Vec3f point, float radius, float half_height,
                           int up_axis) {
    const Vec3f p = _sdf_point_to_z_up(point, up_axis);
    if (p.z > half_height) {
        return math::length(Vec3f(p.x, p.y, p.z - half_height)) - radius;
    }
    if (p.z < -half_height) {
        return math::length(Vec3f(p.x, p.y, p.z + half_height)) - radius;
    }
    return math::length(Vec3f(p.x, p.y, 0.0f)) - radius;
}

CHYSX_DI Vec3f sdf_capsule_grad(Vec3f point, float /*radius*/, float half_height,
                                int up_axis) {
    constexpr float eps = 1.0e-8f;
    const Vec3f p = _sdf_point_to_z_up(point, up_axis);
    Vec3f grad_z_up;
    if (p.z > half_height) {
        const Vec3f v(p.x, p.y, p.z - half_height);
        const float v_len = math::length(v);
        grad_z_up = Vec3f(0.0f, 0.0f, 1.0f);
        if (v_len > eps) {
            grad_z_up = v / v_len;
        }
    } else if (p.z < -half_height) {
        const Vec3f v(p.x, p.y, p.z + half_height);
        const float v_len = math::length(v);
        grad_z_up = Vec3f(0.0f, 0.0f, -1.0f);
        if (v_len > eps) {
            grad_z_up = v / v_len;
        }
    } else {
        const Vec3f v(p.x, p.y, 0.0f);
        const float v_len = math::length(v);
        grad_z_up = Vec3f(0.0f, 0.0f, 1.0f);
        if (v_len > eps) {
            grad_z_up = v / v_len;
        }
    }
    return _sdf_vector_from_z_up(grad_z_up, up_axis);
}

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_cylinder(Vec3f point, float radius, float half_height,
                            int up_axis, float top_radius = -1.0f) {
    const Vec3f p = _sdf_point_to_z_up(point, up_axis);
    if (top_radius < 0.0f || math::abs(top_radius - radius) <= 1.0e-6f) {
        const float dx = math::length(Vec3f(p.x, p.y, 0.0f)) - radius;
        const float dy = math::abs(p.z) - half_height;
        return math::min(math::max(dx, dy), 0.0f)
             + math::length(Vec2f(math::max(dx, 0.0f), math::max(dy, 0.0f)));
    }
    return _sdf_capped_cone_z(radius, top_radius, half_height, p);
}

CHYSX_DI Vec3f sdf_cylinder_grad(Vec3f point, float radius, float half_height,
                                 int up_axis, float top_radius = -1.0f) {
    constexpr float eps = 1.0e-8f;
    const Vec3f p = _sdf_point_to_z_up(point, up_axis);
    if (top_radius >= 0.0f && math::abs(top_radius - radius) > 1.0e-6f) {
        constexpr float fd_eps = 1.0e-4f;
        const float dx = _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p + Vec3f(fd_eps, 0.0f, 0.0f))
                       - _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p - Vec3f(fd_eps, 0.0f, 0.0f));
        const float dy = _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p + Vec3f(0.0f, fd_eps, 0.0f))
                       - _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p - Vec3f(0.0f, fd_eps, 0.0f));
        const float dz = _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p + Vec3f(0.0f, 0.0f, fd_eps))
                       - _sdf_capped_cone_z(radius, top_radius, half_height,
                                            p - Vec3f(0.0f, 0.0f, fd_eps));
        Vec3f grad_z_up(dx, dy, dz);
        const float grad_len = math::length(grad_z_up);
        if (grad_len > eps) {
            grad_z_up = grad_z_up / grad_len;
        } else {
            grad_z_up = Vec3f(0.0f, 0.0f, 1.0f);
        }
        return _sdf_vector_from_z_up(grad_z_up, up_axis);
    }

    const Vec3f v(p.x, p.y, 0.0f);
    const float v_len = math::length(v);
    Vec3f radial(0.0f, 0.0f, 1.0f);
    if (v_len > eps) {
        radial = v / v_len;
    }
    const Vec3f axial(0.0f, 0.0f, static_cast<float>(math::sign(p.z)));
    const float dx = v_len - radius;
    const float dy = math::abs(p.z) - half_height;
    Vec3f grad_z_up;
    if (dx > 0.0f && dy > 0.0f) {
        const Vec3f g = radial * dx + axial * dy;
        const float g_len = math::length(g);
        if (g_len > eps) {
            grad_z_up = g / g_len;
        } else {
            grad_z_up = radial;
        }
    } else if (dx > dy) {
        grad_z_up = radial;
    } else {
        grad_z_up = axial;
    }
    return _sdf_vector_from_z_up(grad_z_up, up_axis);
}

// ---------------------------------------------------------------------------
// Cone
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_cone(Vec3f point, float radius, float half_height,
                        int up_axis) {
    return _sdf_capped_cone_z(radius, 0.0f, half_height,
                              _sdf_point_to_z_up(point, up_axis));
}

CHYSX_DI Vec3f sdf_cone_grad(Vec3f point, float radius, float half_height,
                             int up_axis) {
    const Vec3f p = _sdf_point_to_z_up(point, up_axis);
    if (half_height <= 0.0f) {
        return _sdf_vector_from_z_up(
            Vec3f(0.0f, 0.0f, static_cast<float>(math::sign(p.z))), up_axis);
    }

    const float r = math::length(Vec3f(p.x, p.y, 0.0f));
    const float dx = r - radius * (half_height - p.z) / (2.0f * half_height);
    const float dy = math::abs(p.z) - half_height;
    Vec3f grad_z_up;
    if (dx > dy) {
        if (r > 0.0f) {
            const Vec3f radial_dir(p.x / r, p.y / r, 0.0f);
            grad_z_up = math::normalize(
                radial_dir + Vec3f(0.0f, 0.0f, radius / (2.0f * half_height)));
        } else {
            grad_z_up = Vec3f(0.0f, 0.0f, 1.0f);
        }
    } else {
        grad_z_up = Vec3f(0.0f, 0.0f, static_cast<float>(math::sign(p.z)));
    }
    return _sdf_vector_from_z_up(grad_z_up, up_axis);
}

// ---------------------------------------------------------------------------
// Ellipsoid
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_ellipsoid(Vec3f point, Vec3f radii) {
    constexpr float eps = 1.0e-8f;
    const Vec3f r(math::max(fabsf(radii.x), eps),
                  math::max(fabsf(radii.y), eps),
                  math::max(fabsf(radii.z), eps));
    const Vec3f inv_r(1.0f / r.x, 1.0f / r.y, 1.0f / r.z);
    const Vec3f inv_r2(inv_r.x * inv_r.x, inv_r.y * inv_r.y, inv_r.z * inv_r.z);
    const Vec3f q0 = point * inv_r;
    const Vec3f q1 = point * inv_r2;
    const float k0 = math::length(q0);
    const float k1 = math::length(q1);
    if (k1 > eps) {
        return k0 * (k0 - 1.0f) / k1;
    }
    return -math::min(r.x, math::min(r.y, r.z));
}

CHYSX_DI Vec3f sdf_ellipsoid_grad(Vec3f point, Vec3f radii) {
    constexpr float eps = 1.0e-8f;
    const Vec3f r(math::max(fabsf(radii.x), eps),
                  math::max(fabsf(radii.y), eps),
                  math::max(fabsf(radii.z), eps));
    const Vec3f inv_r(1.0f / r.x, 1.0f / r.y, 1.0f / r.z);
    const Vec3f inv_r2(inv_r.x * inv_r.x, inv_r.y * inv_r.y, inv_r.z * inv_r.z);
    const Vec3f q0 = point * inv_r;
    const Vec3f q1 = point * inv_r2;
    const float k0 = math::length(q0);
    const float k1 = math::length(q1);
    if (k1 < eps) {
        return Vec3f(0.0f, 0.0f, 1.0f);
    }
    Vec3f grad = q1 * (k0 / k1);
    const float grad_len = math::length(grad);
    if (grad_len > eps) {
        return grad / grad_len;
    }
    return Vec3f(0.0f, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

CHYSX_DI float sdf_plane(Vec3f point, float width, float length) {
    if (width > 0.0f && length > 0.0f) {
        const float d = math::max(math::abs(point.x) - width,
                                  math::abs(point.y) - length);
        return math::max(d, math::abs(point.z));
    }
    return point.z;
}

// ---------------------------------------------------------------------------
// Heightfield
// ---------------------------------------------------------------------------

CHYSX_DI float sample_sdf_grad_heightfield(const HeightfieldData& hfd,
                                            const float* elevation_data,
                                            Vec3f pos,
                                            Vec3f& grad) {
    float d_plane;
    Vec3f normal;
    float lateral_dist_sq;
    _heightfield_surface_query(hfd, elevation_data, pos, d_plane, normal,
                               lateral_dist_sq);

    if (lateral_dist_sq > 0.0f) {
        const float dist = sqrtf(lateral_dist_sq + d_plane * d_plane);
        const float cx = math::clamp(pos.x, -hfd.hx, hfd.hx);
        const float cy = math::clamp(pos.y, -hfd.hy, hfd.hy);
        const Vec3f lateral(pos.x - cx, pos.y - cy, 0.0f);
        const Vec3f raw_grad = lateral + d_plane * normal;
        if (math::length_sqr(raw_grad) > 1.0e-20f) {
            grad = math::normalize(raw_grad);
        } else {
            grad = Vec3f(0.0f, 0.0f, 1.0f);
        }
        return dist;
    }

    grad = normal;
    return d_plane;
}

}  // namespace collision
}  // namespace chysx
