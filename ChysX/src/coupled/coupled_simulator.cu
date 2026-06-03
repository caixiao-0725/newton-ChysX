// SPDX-License-Identifier: Apache-2.0
//
// CoupledSimulator CUDA implementation.
//
// Extends VBD particle solver with body-particle contact forces.
// Contact geometry comes from Newton's CollisionPipeline; this code
// computes the penalty force law and accumulates into VBD's per-color
// particle_forces / particle_hessians buffers.

#include "coupled_simulator.h"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace chysx {
namespace coupled {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::coupled: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

// -----------------------------------------------------------------------
// Transform helpers (Newton layout: pos[3] + quat[4] = 7 floats)
// quat order in Warp wp.transform: (px,py,pz,  qx,qy,qz,qw)
// -----------------------------------------------------------------------
struct Transform7 {
    float px, py, pz;
    float qx, qy, qz, qw;
};

__device__ __forceinline__ math::Vec3f tf_point(const Transform7& t, const math::Vec3f& p) {
    float ax = t.qw*p.x + t.qy*p.z - t.qz*p.y;
    float ay = t.qw*p.y + t.qz*p.x - t.qx*p.z;
    float az = t.qw*p.z + t.qx*p.y - t.qy*p.x;
    float aw = -(t.qx*p.x + t.qy*p.y + t.qz*p.z);
    return math::Vec3f(
        t.px + ax*t.qw - aw*t.qx - ay*t.qz + az*t.qy,
        t.py + ay*t.qw - aw*t.qy - az*t.qx + ax*t.qz,
        t.pz + az*t.qw - aw*t.qz - ax*t.qy + ay*t.qx);
}

__device__ __forceinline__ math::Vec3f tf_vector(const Transform7& t, const math::Vec3f& v) {
    float ax = t.qw*v.x + t.qy*v.z - t.qz*v.y;
    float ay = t.qw*v.y + t.qz*v.x - t.qx*v.z;
    float az = t.qw*v.z + t.qx*v.y - t.qy*v.x;
    float aw = -(t.qx*v.x + t.qy*v.y + t.qz*v.z);
    return math::Vec3f(
        ax*t.qw - aw*t.qx - ay*t.qz + az*t.qy,
        ay*t.qw - aw*t.qy - az*t.qx + ax*t.qz,
        az*t.qw - aw*t.qz - ax*t.qy + ay*t.qx);
}

__device__ __forceinline__ Transform7 load_tf(const float* ptr, int idx) {
    const float* p = ptr + idx * 7;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6]};
}

__device__ __forceinline__ math::Vec3f load_vec3(const float* ptr, int idx) {
    const float* p = ptr + idx * 3;
    return {p[0], p[1], p[2]};
}

// -----------------------------------------------------------------------
// IPC-style isotropic friction
// -----------------------------------------------------------------------
__device__ void compute_friction(
    float mu, float f_n, const math::Vec3f& n, const math::Vec3f& slip,
    float eps_u,
    math::Vec3f& out_force, math::Mat3f& out_hessian) {

    const float dot_nu = n.x*slip.x + n.y*slip.y + n.z*slip.z;
    const math::Vec3f u_t = slip - n * dot_nu;
    const float u_norm = sqrtf(u_t.x*u_t.x + u_t.y*u_t.y + u_t.z*u_t.z);

    if (u_norm > 0.0f) {
        float scale;
        if (u_norm > eps_u) {
            scale = mu * f_n / u_norm;
        } else {
            scale = mu * f_n * (-u_norm / eps_u + 2.0f) / eps_u;
        }
        out_force = u_t * (-scale);
        out_hessian = math::Mat3f(
            scale*(1.0f - n.x*n.x), scale*(-n.x*n.y),      scale*(-n.x*n.z),
            scale*(-n.y*n.x),       scale*(1.0f - n.y*n.y), scale*(-n.y*n.z),
            scale*(-n.z*n.x),       scale*(-n.z*n.y),       scale*(1.0f - n.z*n.z));
    } else {
        out_force = math::Vec3f(0.0f, 0.0f, 0.0f);
        out_hessian = math::Mat3f();
    }
}

// -----------------------------------------------------------------------
// Body-particle contact force kernel (per-color)
// -----------------------------------------------------------------------
__global__ void accumulate_body_particle_contacts_kernel(
    float dt,
    int current_color,
    const math::Vec3f* __restrict__ pos_prev,
    const math::Vec3f* __restrict__ pos,
    const int* __restrict__ particle_colors,
    float friction_epsilon,
    const float* __restrict__ particle_radius,
    const int* __restrict__ contact_particle,
    const int* __restrict__ contact_count,
    int contact_max,
    const float* __restrict__ contact_ke,
    const float* __restrict__ contact_kd,
    const float* __restrict__ contact_mu,
    const int* __restrict__ shape_body,
    const float* __restrict__ body_q,
    const float* __restrict__ body_q_prev,
    const int* __restrict__ contact_shape,
    const float* __restrict__ contact_body_pos,
    const float* __restrict__ contact_body_vel,
    const float* __restrict__ contact_normal,
    math::Vec3f* __restrict__ particle_forces,
    math::Mat3f* __restrict__ particle_hessians,
    int n_launch) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;

    const int actual_count = min(contact_max, contact_count[0]);
    if (tid >= actual_count) return;

    const int pi = contact_particle[tid];
    if (particle_colors[pi] != current_color) return;

    const float ke = contact_ke[tid];
    const float kd = contact_kd[tid];
    const float mu = contact_mu[tid];

    const int shape_idx = contact_shape[tid];
    const int body_idx = shape_body[shape_idx];

    math::Vec3f bx;
    if (body_idx >= 0) {
        Transform7 tf = load_tf(body_q, body_idx);
        bx = tf_point(tf, load_vec3(contact_body_pos, tid));
    } else {
        bx = load_vec3(contact_body_pos, tid);
    }

    const math::Vec3f n = load_vec3(contact_normal, tid);
    const math::Vec3f p_pos = pos[pi];
    const float radius = particle_radius[pi];

    const math::Vec3f diff = p_pos - bx;
    const float gap = n.x*diff.x + n.y*diff.y + n.z*diff.z;
    const float penetration = -(gap - radius);

    if (!(penetration > 0.0f)) return;

    const math::Vec3f dx = p_pos - pos_prev[pi];
    math::Vec3f bv(0.0f, 0.0f, 0.0f);

    if (body_q_prev != nullptr && body_idx >= 0) {
        Transform7 tf_prev = load_tf(body_q_prev, body_idx);
        const math::Vec3f bx_prev = tf_point(tf_prev, load_vec3(contact_body_pos, tid));
        Transform7 tf_cur = load_tf(body_q, body_idx);
        const math::Vec3f local_vel = tf_vector(tf_cur, load_vec3(contact_body_vel, tid));
        bv = (bx - bx_prev) * (1.0f / dt) + local_vel;
    }
    const math::Vec3f rel_trans = dx - bv * dt;

    const float f_n = penetration * ke;
    math::Vec3f force = n * f_n;
    math::Mat3f hessian(
        ke*n.x*n.x, ke*n.x*n.y, ke*n.x*n.z,
        ke*n.y*n.x, ke*n.y*n.y, ke*n.y*n.z,
        ke*n.z*n.x, ke*n.z*n.y, ke*n.z*n.z);

    const float n_dot_rel = n.x*rel_trans.x + n.y*rel_trans.y + n.z*rel_trans.z;
    if (n_dot_rel < 0.0f) {
        const float damp_coeff = kd * ke / dt;
        math::Mat3f damp_h(
            damp_coeff*n.x*n.x, damp_coeff*n.x*n.y, damp_coeff*n.x*n.z,
            damp_coeff*n.y*n.x, damp_coeff*n.y*n.y, damp_coeff*n.y*n.z,
            damp_coeff*n.z*n.x, damp_coeff*n.z*n.y, damp_coeff*n.z*n.z);
        hessian = hessian + damp_h;
        force = force - damp_h * rel_trans;
    }

    math::Vec3f f_fric;
    math::Mat3f h_fric;
    compute_friction(mu, f_n, n, rel_trans, friction_epsilon * dt, f_fric, h_fric);
    force = force + f_fric;
    hessian = hessian + h_fric;

    atomicAdd(&particle_forces[pi].x, force.x);
    atomicAdd(&particle_forces[pi].y, force.y);
    atomicAdd(&particle_forces[pi].z, force.z);

    float* h_ptr = reinterpret_cast<float*>(&particle_hessians[pi]);
    const float* h_src = reinterpret_cast<const float*>(&hessian);
    for (int k = 0; k < 9; ++k)
        atomicAdd(h_ptr + k, h_src[k]);
}

// -----------------------------------------------------------------------
// Contact callback data
// -----------------------------------------------------------------------
struct ContactCallbackData {
    float dt;
    float friction_epsilon;
    int n_particles;
    const BodyParticleContacts* contacts;
    const ExternalBodies* bodies;
    const int* vbd_particle_colors;  // from VBDSolver, matches color_groups
};

// Per-color callback: zero accumulators, accumulate contacts
void contact_callback_fn(
    int color,
    const math::Vec3f* q_prev,
    const math::Vec3f* pos,
    const float* /*mass*/,
    math::Vec3f* particle_forces,
    math::Mat3f* particle_hessians,
    cudaStream_t stream,
    void* user_data) {

    auto* d = static_cast<ContactCallbackData*>(user_data);
    const int n = d->n_particles;

    check_cuda(cudaMemsetAsync(particle_forces, 0,
               n * sizeof(math::Vec3f), stream), "zero pf");
    check_cuda(cudaMemsetAsync(particle_hessians, 0,
               n * sizeof(math::Mat3f), stream), "zero ph");

    const auto& c = *d->contacts;
    const auto& b = *d->bodies;

    const int n_launch = c.contact_max;
    constexpr int block = 256;
    const int grid = (n_launch + block - 1) / block;

    const int* particle_colors = d->vbd_particle_colors;

    accumulate_body_particle_contacts_kernel<<<grid, block, 0, stream>>>(
        d->dt, color,
        q_prev, pos,
        particle_colors,
        d->friction_epsilon,
        b.particle_radius,
        c.contact_particle,
        c.contact_count,
        c.contact_max,
        c.contact_ke,
        c.contact_kd,
        c.contact_mu,
        b.shape_body,
        b.body_q,
        b.body_q_prev,
        c.contact_shape,
        c.contact_body_pos,
        c.contact_body_vel,
        c.contact_normal,
        particle_forces,
        particle_hessians,
        n_launch);
    check_cuda(cudaGetLastError(), "contact_callback kernel");
}

}  // namespace

// ====================================================================
// CoupledSimulator methods
// ====================================================================

void CoupledSimulator::build_coloring(
    const math::Vec4i* host_tets, int n_tets, int n_particles) {
    vbd_.build_coloring(host_tets, n_tets, n_particles);
    // Pre-allocate contact force buffers so step() never allocates
    // during CUDA graph capture.
    if (n_particles > 0) {
        particle_forces_.allocate_device(static_cast<std::size_t>(n_particles));
        particle_hessians_.allocate_device(static_cast<std::size_t>(n_particles));
    }
}

void CoupledSimulator::build_adjacency(
    const math::Vec4i* host_tets, int n_tets, int n_particles) {
    vbd_.build_adjacency(host_tets, n_tets, n_particles);
}

void CoupledSimulator::set_coloring(const int* host_colors, int n_particles) {
    vbd_.set_coloring(host_colors, n_particles);
    if (n_particles > 0) {
        particle_forces_.allocate_device(static_cast<std::size_t>(n_particles));
        particle_hessians_.allocate_device(static_cast<std::size_t>(n_particles));
    }
}

void CoupledSimulator::step(
    DeviceSpan<math::Vec3f> pos,
    DeviceSpan<math::Vec3f> vel,
    DeviceSpan<float> inv_mass,
    DeviceSpan<math::Vec4i> tet_indices,
    DeviceSpan<math::Mat3f> tet_poses,
    DeviceSpan<math::Vec3f> tet_materials,
    math::Vec3f gravity,
    float dt,
    int iterations,
    const BodyParticleContacts& contacts,
    const ExternalBodies& bodies,
    float friction_epsilon,
    std::uintptr_t cuda_stream) {

    const int n = static_cast<int>(pos.size());
    if (n <= 0) return;

    const bool has_contacts = contacts.contact_particle != nullptr
                           && contacts.contact_count != nullptr
                           && contacts.contact_max > 0;
    if (has_contacts) {
        ContactCallbackData cb_data;
        cb_data.dt = dt;
        cb_data.friction_epsilon = friction_epsilon;
        cb_data.n_particles = n;
        cb_data.contacts = &contacts;
        cb_data.bodies = &bodies;
        cb_data.vbd_particle_colors = (bodies.particle_colors != nullptr)
            ? bodies.particle_colors
            : vbd_.particle_colors_ptr();

        vbd_.step_with_contacts(
            pos, vel, inv_mass,
            tet_indices, tet_poses, tet_materials,
            gravity, dt, iterations,
            contact_callback_fn,
            &cb_data,
            particle_forces_.gpu_data(),
            particle_hessians_.gpu_data(),
            cuda_stream);
    } else {
        vbd_.step(pos, vel, inv_mass,
                  tet_indices, tet_poses, tet_materials,
                  gravity, dt, iterations, cuda_stream);
    }
}

// ====================================================================
// Collision pipeline integration
// ====================================================================

void CoupledSimulator::add_collision_shape(
    int body, int geo_type,
    float sx, float sy, float sz,
    const float* local_tf_7, int flags,
    uint64_t mesh_id,
    float mat_ke, float mat_kd, float mat_mu) {
    collision_.add_shape(body, geo_type, math::Vec3f(sx, sy, sz),
                         local_tf_7, flags, mesh_id,
                         mat_ke, mat_kd, mat_mu);
}

void CoupledSimulator::finalize_collision(int max_soft_contacts) {
    collision_.finalize(max_soft_contacts);
}

void CoupledSimulator::step_with_collision(
    DeviceSpan<math::Vec3f> pos,
    DeviceSpan<math::Vec3f> vel,
    DeviceSpan<float> inv_mass,
    DeviceSpan<math::Vec4i> tet_indices,
    DeviceSpan<math::Mat3f> tet_poses,
    DeviceSpan<math::Vec3f> tet_materials,
    math::Vec3f gravity,
    float dt,
    int iterations,
    const float* body_q,
    const float* body_q_prev,
    int n_bodies,
    const float* particle_radius,
    const int*   particle_flags,
    float margin,
    float friction_epsilon,
    float soft_contact_ke,
    float soft_contact_kd,
    float soft_contact_mu,
    std::uintptr_t cuda_stream)
{
    const int n = static_cast<int>(pos.size());
    if (n <= 0) return;

    // 1. Run collision detection
    collision_.collide(
        pos.data(), particle_radius, particle_flags,
        n, body_q, n_bodies, margin, cuda_stream);

    // 2. Initialize per-contact material coefficients
    collision_.init_contact_materials(
        soft_contact_ke, soft_contact_kd, soft_contact_mu, cuda_stream);

    // 3. Build the BodyParticleContacts struct from collision results
    BodyParticleContacts contacts;
    contacts.contact_particle = collision_.contact_particle_ptr();
    contacts.contact_count    = collision_.contact_count_ptr();
    contacts.contact_max      = collision_.contact_max();
    contacts.contact_ke       = collision_.contact_ke_ptr();
    contacts.contact_kd       = collision_.contact_kd_ptr();
    contacts.contact_mu       = collision_.contact_mu_ptr();
    contacts.contact_shape    = collision_.contact_shape_ptr();
    contacts.contact_body_pos = collision_.contact_body_pos_ptr();
    contacts.contact_body_vel = collision_.contact_body_vel_ptr();
    contacts.contact_normal   = collision_.contact_normal_ptr();

    // 4. Build the ExternalBodies struct
    ExternalBodies bodies;
    bodies.body_q       = const_cast<float*>(body_q);
    bodies.body_q_prev  = const_cast<float*>(body_q_prev);
    bodies.body_qd      = nullptr;
    bodies.body_com     = nullptr;
    bodies.shape_body   = collision_.shape_body_ptr();
    bodies.particle_radius = const_cast<float*>(particle_radius);
    bodies.particle_colors = nullptr;  // use VBD's internal coloring
    bodies.n_bodies     = n_bodies;
    bodies.n_shapes     = collision_.shape_count();

    // 5. Run VBD step with contacts (reuse existing path)
    ContactCallbackData cb_data;
    cb_data.dt = dt;
    cb_data.friction_epsilon = friction_epsilon;
    cb_data.n_particles = n;
    cb_data.contacts = &contacts;
    cb_data.bodies = &bodies;
    cb_data.vbd_particle_colors = vbd_.particle_colors_ptr();

    vbd_.step_with_contacts(
        pos, vel, inv_mass,
        tet_indices, tet_poses, tet_materials,
        gravity, dt, iterations,
        contact_callback_fn,
        &cb_data,
        particle_forces_.gpu_data(),
        particle_hessians_.gpu_data(),
        cuda_stream);
}

}  // namespace coupled
}  // namespace chysx
