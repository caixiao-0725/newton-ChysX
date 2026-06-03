// SPDX-License-Identifier: Apache-2.0
//
// CollisionPipeline — brute-force body-particle collision detection.
// Ports Newton's ``create_soft_contacts`` kernel to C++/CUDA.

#include "collision_pipeline.h"
#include "sdf_primitives.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

// Transform7 — same layout as Newton's wp.transform:
//   (px, py, pz, qx, qy, qz, qw)
struct Transform7 {
    float px, py, pz;
    float qx, qy, qz, qw;
};

__device__ __forceinline__ Transform7 load_tf(const float* ptr, int idx) {
    const float* p = ptr + idx * 7;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6]};
}

// q * p * conj(q) + t  (Hamilton product, matches fixed tf_point)
__device__ __forceinline__ math::Vec3f tf_point(
    const Transform7& t, const math::Vec3f& p) {
    float ax = t.qw*p.x + t.qy*p.z - t.qz*p.y;
    float ay = t.qw*p.y + t.qz*p.x - t.qx*p.z;
    float az = t.qw*p.z + t.qx*p.y - t.qy*p.x;
    float aw = -(t.qx*p.x + t.qy*p.y + t.qz*p.z);
    return math::Vec3f(
        t.px + ax*t.qw - aw*t.qx - ay*t.qz + az*t.qy,
        t.py + ay*t.qw - aw*t.qy - az*t.qx + ax*t.qz,
        t.pz + az*t.qw - aw*t.qz - ax*t.qy + ay*t.qx);
}

// q * v * conj(q)  (rotation only, no translation)
__device__ __forceinline__ math::Vec3f tf_vector(
    const Transform7& t, const math::Vec3f& v) {
    float ax = t.qw*v.x + t.qy*v.z - t.qz*v.y;
    float ay = t.qw*v.y + t.qz*v.x - t.qx*v.z;
    float az = t.qw*v.z + t.qx*v.y - t.qy*v.x;
    float aw = -(t.qx*v.x + t.qy*v.y + t.qz*v.z);
    return math::Vec3f(
        ax*t.qw - aw*t.qx - ay*t.qz + az*t.qy,
        ay*t.qw - aw*t.qy - az*t.qx + ax*t.qz,
        az*t.qw - aw*t.qz - ax*t.qy + ay*t.qx);
}

// Inverse of a Transform7: inv_q = conj(q), inv_p = -(conj(q) * p * q)
__device__ __forceinline__ Transform7 tf_inverse(const Transform7& t) {
    Transform7 inv;
    inv.qx = -t.qx; inv.qy = -t.qy; inv.qz = -t.qz; inv.qw = t.qw;
    inv.px = 0.0f; inv.py = 0.0f; inv.pz = 0.0f;
    math::Vec3f neg_p = tf_vector(inv, math::Vec3f(t.px, t.py, t.pz));
    inv.px = -neg_p.x; inv.py = -neg_p.y; inv.pz = -neg_p.z;
    return inv;
}

// Multiply two Transform7: X_ws = X_wb * X_bs
__device__ __forceinline__ Transform7 tf_multiply(
    const Transform7& a, const Transform7& b) {
    Transform7 out;
    // q_out = q_a * q_b (Hamilton product)
    out.qw = a.qw*b.qw - a.qx*b.qx - a.qy*b.qy - a.qz*b.qz;
    out.qx = a.qw*b.qx + a.qx*b.qw + a.qy*b.qz - a.qz*b.qy;
    out.qy = a.qw*b.qy - a.qx*b.qz + a.qy*b.qw + a.qz*b.qx;
    out.qz = a.qw*b.qz + a.qx*b.qy - a.qy*b.qx + a.qz*b.qw;
    // p_out = q_a * p_b * conj(q_a) + p_a
    math::Vec3f rotated = tf_vector(a, math::Vec3f(b.px, b.py, b.pz));
    out.px = a.px + rotated.x;
    out.py = a.py + rotated.y;
    out.pz = a.pz + rotated.z;
    return out;
}

__device__ __forceinline__ Transform7 make_identity_tf() {
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}

__device__ __forceinline__ Transform7 load_shape_tf(const float* ptr, int idx) {
    const float* p = ptr + idx * 7;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6]};
}

// -----------------------------------------------------------------------
// Main collision detection kernel
// -----------------------------------------------------------------------
__global__ void create_soft_contacts_kernel(
    const math::Vec3f* __restrict__ particle_q,
    const float* __restrict__ particle_radius,
    const int* __restrict__ particle_flags,
    int n_particles,
    const float* __restrict__ body_q,  // 7 floats per body
    int n_bodies,
    // shape SoA
    const int* __restrict__ shape_body,
    const int* __restrict__ shape_type,
    const math::Vec3f* __restrict__ shape_scale,
    const float* __restrict__ shape_transform,  // 7 floats per shape
    const uint64_t* __restrict__ shape_mesh_id,
    const int* __restrict__ shape_flags,
    int n_shapes,
    float margin,
    // outputs
    int* __restrict__ contact_count,
    int max_contacts,
    int* __restrict__ contact_particle,
    int* __restrict__ contact_shape,
    math::Vec3f* __restrict__ contact_body_pos,
    math::Vec3f* __restrict__ contact_body_vel,
    math::Vec3f* __restrict__ contact_normal,
    int n_launch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;

    const int particle_index = tid / n_shapes;
    const int shape_index    = tid % n_shapes;

    if (particle_index >= n_particles) return;

    if ((particle_flags[particle_index] & PARTICLE_ACTIVE) == 0) return;
    if ((shape_flags[shape_index] & SHAPE_COLLIDE_PARTICLES) == 0) return;

    const int rigid_index = shape_body[shape_index];

    const math::Vec3f px = particle_q[particle_index];
    const float radius = particle_radius[particle_index];

    Transform7 X_wb = make_identity_tf();
    if (rigid_index >= 0 && rigid_index < n_bodies) {
        X_wb = load_tf(body_q, rigid_index);
    }

    Transform7 X_bs = load_shape_tf(shape_transform, shape_index);

    Transform7 X_ws = tf_multiply(X_wb, X_bs);
    Transform7 X_sw = tf_inverse(X_ws);

    // Transform particle to shape local space
    math::Vec3f x_local = tf_point(X_sw, px);

    const int geo_t = shape_type[shape_index];
    const math::Vec3f geo_scale = shape_scale[shape_index];

    float d = 1.0e6f;
    math::Vec3f n(0.0f, 0.0f, 0.0f);
    math::Vec3f v(0.0f, 0.0f, 0.0f);

    // up_axis = Z = 2 (Newton convention for capsule/cylinder/cone)
    constexpr int UP_AXIS_Z = 2;

    if (geo_t == GEO_SPHERE) {
        d = sdf_sphere(x_local, geo_scale.x);
        n = sdf_sphere_grad(x_local, geo_scale.x);
    }
    else if (geo_t == GEO_BOX) {
        d = sdf_box(x_local, geo_scale.x, geo_scale.y, geo_scale.z);
        n = sdf_box_grad(x_local, geo_scale.x, geo_scale.y, geo_scale.z);
    }
    else if (geo_t == GEO_CAPSULE) {
        d = sdf_capsule(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
        n = sdf_capsule_grad(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
    }
    else if (geo_t == GEO_CYLINDER) {
        d = sdf_cylinder(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
        n = sdf_cylinder_grad(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
    }
    else if (geo_t == GEO_CONE) {
        d = sdf_cone(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
        n = sdf_cone_grad(x_local, geo_scale.x, geo_scale.y, UP_AXIS_Z);
    }
    else if (geo_t == GEO_ELLIPSOID) {
        d = sdf_ellipsoid(x_local, geo_scale);
        n = sdf_ellipsoid_grad(x_local, geo_scale);
    }
    else if (geo_t == GEO_PLANE) {
        d = sdf_plane(x_local, geo_scale.x * 0.5f, geo_scale.y * 0.5f);
        n = math::Vec3f(0.0f, 0.0f, 1.0f);
    }
    // MESH and CONVEX_MESH: skip for now (requires Warp mesh API)
    // HFIELD: skip for now (requires elevation data)

    if (d < margin + radius) {
        int index = atomicAdd(contact_count, 1);
        if (index < max_contacts) {
            // contact point in body local space: X_bs * (x_local - n * d)
            math::Vec3f surface_local = x_local - n * d;
            math::Vec3f body_pos_val = tf_point(X_bs, surface_local);
            math::Vec3f body_vel_val = tf_vector(X_bs, v);
            math::Vec3f world_normal = tf_vector(X_ws, n);

            contact_shape[index]    = shape_index;
            contact_body_pos[index] = body_pos_val;
            contact_body_vel[index] = body_vel_val;
            contact_particle[index] = particle_index;
            contact_normal[index]   = world_normal;
        }
    }
}

// -----------------------------------------------------------------------
// Material mixing kernel
// -----------------------------------------------------------------------
__global__ void init_contact_materials_kernel(
    const int* __restrict__ contact_count,
    int max_contacts,
    const int* __restrict__ contact_shape,
    float soft_ke, float soft_kd, float soft_mu,
    const float* __restrict__ shape_ke,
    const float* __restrict__ shape_kd,
    const float* __restrict__ shape_mu,
    float* __restrict__ out_ke,
    float* __restrict__ out_kd,
    float* __restrict__ out_mu)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(max_contacts, contact_count[0]);
    if (tid >= n) return;

    const int si = contact_shape[tid];
    if (si < 0) return;

    out_ke[tid] = 0.5f * (soft_ke + shape_ke[si]);
    out_kd[tid] = 0.5f * (soft_kd + shape_kd[si]);
    out_mu[tid] = sqrtf(soft_mu * shape_mu[si]);
}

}  // anonymous namespace

// ====================================================================
// CollisionPipeline methods
// ====================================================================

void CollisionPipeline::add_shape(
    int body, int geo_type, math::Vec3f scale,
    const float* local_tf_7, int flags, uint64_t mesh_id,
    float mat_ke, float mat_kd, float mat_mu)
{
    shapes_host_.body.push_back(body);
    shapes_host_.geo_type.push_back(geo_type);
    shapes_host_.geo_scale.push_back(scale);
    for (int i = 0; i < 7; ++i) {
        shapes_host_.local_transform.push_back(local_tf_7[i]);
    }
    shapes_host_.flags.push_back(flags);
    shapes_host_.mesh_id.push_back(mesh_id);
    shapes_host_.material_ke.push_back(mat_ke);
    shapes_host_.material_kd.push_back(mat_kd);
    shapes_host_.material_mu.push_back(mat_mu);
    shapes_host_.count++;
}

void CollisionPipeline::finalize(int max_soft_contacts) {
    const int n = shapes_host_.count;
    if (n == 0) return;

    shapes_gpu_.count = n;

    auto upload = [&](auto& dst, auto& src) {
        using T = typename std::remove_reference_t<decltype(dst)>::value_type;
        dst.resize(src.size());
        std::memcpy(dst.cpu_data(), src.data(), src.size() * sizeof(T));
        dst.copy_to_device();
    };

    upload(shapes_gpu_.body,            shapes_host_.body);
    upload(shapes_gpu_.geo_type,        shapes_host_.geo_type);
    upload(shapes_gpu_.geo_scale,       shapes_host_.geo_scale);
    upload(shapes_gpu_.local_transform, shapes_host_.local_transform);
    upload(shapes_gpu_.flags,           shapes_host_.flags);
    upload(shapes_gpu_.mesh_id,         shapes_host_.mesh_id);
    upload(shapes_gpu_.material_ke,     shapes_host_.material_ke);
    upload(shapes_gpu_.material_kd,     shapes_host_.material_kd);
    upload(shapes_gpu_.material_mu,     shapes_host_.material_mu);

    // Allocate contact buffer
    if (max_soft_contacts <= 0) {
        max_soft_contacts = 64 * 1024;  // reasonable default
    }
    contacts_.max_contacts = max_soft_contacts;
    contacts_.count.resize(1);
    contacts_.particle.allocate_device(max_soft_contacts);
    contacts_.shape.allocate_device(max_soft_contacts);
    contacts_.body_pos.allocate_device(max_soft_contacts);
    contacts_.body_vel.allocate_device(max_soft_contacts);
    contacts_.normal.allocate_device(max_soft_contacts);
    contacts_.ke.allocate_device(max_soft_contacts);
    contacts_.kd.allocate_device(max_soft_contacts);
    contacts_.mu.allocate_device(max_soft_contacts);
}

void CollisionPipeline::collide(
    const math::Vec3f* particle_q,
    const float*  particle_radius,
    const int*    particle_flags,
    int n_particles,
    const float*  body_q,
    int n_bodies,
    float margin,
    std::uintptr_t cuda_stream)
{
    if (shapes_gpu_.count == 0 || n_particles == 0) return;

    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    // Reset contact counter
    check_cuda(cudaMemsetAsync(contacts_.count.gpu_data(), 0,
               sizeof(int), stream), "zero contact count");

    const int n_shapes = shapes_gpu_.count;
    const long long n_launch_ll = (long long)n_particles * n_shapes;
    if (n_launch_ll > INT_MAX || n_launch_ll <= 0) {
        return;
    }
    const int n_launch = (int)n_launch_ll;
    constexpr int block = 256;
    const int grid = (n_launch + block - 1) / block;

    create_soft_contacts_kernel<<<grid, block, 0, stream>>>(
        particle_q, particle_radius, particle_flags,
        n_particles,
        body_q, n_bodies,
        shapes_gpu_.body.gpu_data(),
        shapes_gpu_.geo_type.gpu_data(),
        shapes_gpu_.geo_scale.gpu_data(),
        shapes_gpu_.local_transform.gpu_data(),
        shapes_gpu_.mesh_id.gpu_data(),
        shapes_gpu_.flags.gpu_data(),
        n_shapes, margin,
        contacts_.count.gpu_data(),
        contacts_.max_contacts,
        contacts_.particle.gpu_data(),
        contacts_.shape.gpu_data(),
        contacts_.body_pos.gpu_data(),
        contacts_.body_vel.gpu_data(),
        contacts_.normal.gpu_data(),
        n_launch);
    check_cuda(cudaGetLastError(), "create_soft_contacts_kernel");
}

void CollisionPipeline::init_contact_materials(
    float soft_contact_ke, float soft_contact_kd, float soft_contact_mu,
    std::uintptr_t cuda_stream)
{
    if (contacts_.max_contacts == 0) return;

    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int n_launch = contacts_.max_contacts;
    constexpr int block = 256;
    const int grid = (n_launch + block - 1) / block;

    init_contact_materials_kernel<<<grid, block, 0, stream>>>(
        contacts_.count.gpu_data(),
        contacts_.max_contacts,
        contacts_.shape.gpu_data(),
        soft_contact_ke, soft_contact_kd, soft_contact_mu,
        shapes_gpu_.material_ke.gpu_data(),
        shapes_gpu_.material_kd.gpu_data(),
        shapes_gpu_.material_mu.gpu_data(),
        contacts_.ke.gpu_data(),
        contacts_.kd.gpu_data(),
        contacts_.mu.gpu_data());
    check_cuda(cudaGetLastError(), "init_contact_materials_kernel");
}

}  // namespace collision
}  // namespace chysx
