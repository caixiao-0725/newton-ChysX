// SPDX-License-Identifier: Apache-2.0
//
// RigidSimulator — orchestrates the full AVBD rigid-body pipeline.

#include "rigid_simulator.h"

#include "avbd_kernels.cuh"
#include "graph_coloring.h"
#include "narrow_phase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace chysx {
namespace rigid {

// ============================================================================
// Scene construction
// ============================================================================

int RigidSimulator::add_body(float mass, const math::Mat3f& inertia,
                             const math::Vec3f& com,
                             const math::Vec3f& pos, const math::Quatf& quat) {
    if (finalized_) throw std::runtime_error("Cannot add body after finalize()");
    int idx = static_cast<int>(body_staging_.size());
    body_staging_.push_back({mass, inertia, com, pos, quat});
    return idx;
}

int RigidSimulator::add_shape_sphere(int body, float radius,
                                     float ke, float kd, float mu, float gap) {
    if (finalized_) throw std::runtime_error("Cannot add shape after finalize()");
    int idx = static_cast<int>(shape_staging_.size());
    shape_staging_.push_back({body, GEO_SPHERE,
                              math::Vec3f(radius, 0.f, 0.f),
                              math::Vec3f(0.f), math::quat_identity(),
                              ke, kd, mu, gap});
    return idx;
}

int RigidSimulator::add_shape_box(int body, const math::Vec3f& half_extents,
                                  float ke, float kd, float mu, float gap) {
    if (finalized_) throw std::runtime_error("Cannot add shape after finalize()");
    int idx = static_cast<int>(shape_staging_.size());
    shape_staging_.push_back({body, GEO_BOX, half_extents,
                              math::Vec3f(0.f), math::quat_identity(),
                              ke, kd, mu, gap});
    return idx;
}

int RigidSimulator::add_shape_capsule(int body, float radius, float half_height,
                                      float ke, float kd, float mu, float gap) {
    if (finalized_) throw std::runtime_error("Cannot add shape after finalize()");
    int idx = static_cast<int>(shape_staging_.size());
    shape_staging_.push_back({body, GEO_CAPSULE,
                              math::Vec3f(radius, half_height, 0.f),
                              math::Vec3f(0.f), math::quat_identity(),
                              ke, kd, mu, gap});
    return idx;
}

void RigidSimulator::add_ground_plane(float ke, float kd, float mu) {
    if (finalized_) throw std::runtime_error("Cannot add ground after finalize()");
    int idx = static_cast<int>(shape_staging_.size());
    // Plane normal = quat_rotate(quat, Y-axis).
    // Newton uses Z-up, so rotate Y→Z: 90° around X.
    constexpr float s = 0.70710678118f;  // sin(45°)
    constexpr float c = 0.70710678118f;  // cos(45°)
    math::Quatf ground_quat(s, 0.f, 0.f, c);
    shape_staging_.push_back({-1, GEO_PLANE,
                              math::Vec3f(0.f),
                              math::Vec3f(0.f), ground_quat,
                              ke, kd, mu, 0.f});
    (void)idx;
}

int RigidSimulator::add_joint(JointTypeArg type, int parent, int child,
                              const math::Vec3f& anchor_p, const math::Quatf& frame_p,
                              const math::Vec3f& anchor_c, const math::Quatf& frame_c) {
    if (finalized_) throw std::runtime_error("Cannot add joint after finalize()");
    int idx = static_cast<int>(joint_staging_.size());
    joint_staging_.push_back({static_cast<int>(type), parent, child,
                              anchor_p, frame_p, anchor_c, frame_c});
    return idx;
}

// ============================================================================
// Finalize
// ============================================================================

void RigidSimulator::finalize() {
    if (finalized_) return;

    int nb = static_cast<int>(body_staging_.size());
    int ns = static_cast<int>(shape_staging_.size());
    int nj = static_cast<int>(joint_staging_.size());

    // --- Bodies ---
    bodies_.resize(nb);
    for (int i = 0; i < nb; ++i) {
        auto& b = body_staging_[i];
        bodies_.com[i] = b.com;
        bodies_.mass[i] = b.mass;
        bodies_.inv_mass[i] = (b.mass > 0.f) ? (1.f / b.mass) : 0.f;
        bodies_.inertia[i] = b.inertia;
        bodies_.inv_inertia[i] = (b.mass > 0.f) ? math::inverse(b.inertia) : math::Mat3f::zero();
        bodies_.pos[i] = b.pos;
        bodies_.quat[i] = b.quat;
        bodies_.vel[i] = math::Vec3f(0.f);
        bodies_.omega[i] = math::Vec3f(0.f);
        bodies_.pos_prev[i] = b.pos;
        bodies_.quat_prev[i] = b.quat;
        bodies_.inertia_pos[i] = b.pos;
        bodies_.inertia_quat[i] = b.quat;
        bodies_.forces[i] = math::Vec3f(0.f);
        bodies_.torques[i] = math::Vec3f(0.f);
        bodies_.hessian_ll[i] = math::Mat3f::zero();
        bodies_.hessian_al[i] = math::Mat3f::zero();
        bodies_.hessian_aa[i] = math::Mat3f::zero();
    }
    bodies_.com.copy_to_device();
    bodies_.mass.copy_to_device();
    bodies_.inv_mass.copy_to_device();
    bodies_.inertia.copy_to_device();
    bodies_.inv_inertia.copy_to_device();
    bodies_.pos.copy_to_device();
    bodies_.quat.copy_to_device();
    bodies_.vel.copy_to_device();
    bodies_.omega.copy_to_device();
    bodies_.pos_prev.copy_to_device();
    bodies_.quat_prev.copy_to_device();
    bodies_.inertia_pos.copy_to_device();
    bodies_.inertia_quat.copy_to_device();
    bodies_.forces.copy_to_device();
    bodies_.torques.copy_to_device();
    bodies_.hessian_ll.copy_to_device();
    bodies_.hessian_al.copy_to_device();
    bodies_.hessian_aa.copy_to_device();

    // --- Shapes ---
    shapes_.resize(ns);
    for (int i = 0; i < ns; ++i) {
        auto& s = shape_staging_[i];
        shapes_.body[i] = s.body;
        shapes_.geo_type[i] = s.geo_type;
        shapes_.geo_scale[i] = s.geo_scale;
        shapes_.pos_local[i] = s.pos_local;
        shapes_.quat_local[i] = s.quat_local;
        shapes_.ke[i] = s.ke;
        shapes_.kd[i] = s.kd;
        shapes_.mu[i] = s.mu;
        shapes_.gap[i] = s.gap;
    }
    shapes_.body.copy_to_device();
    shapes_.geo_type.copy_to_device();
    shapes_.geo_scale.copy_to_device();
    shapes_.pos_local.copy_to_device();
    shapes_.quat_local.copy_to_device();
    shapes_.ke.copy_to_device();
    shapes_.kd.copy_to_device();
    shapes_.mu.copy_to_device();
    shapes_.gap.copy_to_device();

    // --- Joints ---
    if (nj > 0) {
        joints_.resize(nj);
        for (int i = 0; i < nj; ++i) {
            auto& j = joint_staging_[i];
            joints_.type[i] = j.type;
            joints_.parent[i] = j.parent;
            joints_.child[i] = j.child;
            joints_.X_p_pos[i] = j.X_p_pos;
            joints_.X_p_quat[i] = j.X_p_quat;
            joints_.X_c_pos[i] = j.X_c_pos;
            joints_.X_c_quat[i] = j.X_c_quat;
            joints_.penalty_k_lin[i] = 1e5f;
            joints_.penalty_k_ang[i] = 1e5f;
            joints_.lambda_lin[i] = math::Vec3f(0.f);
            joints_.lambda_ang[i] = math::Vec3f(0.f);
            joints_.C0_lin[i] = math::Vec3f(0.f);
            joints_.C0_ang[i] = math::Vec3f(0.f);
            joints_.is_hard[i] = 1;
        }
        joints_.type.copy_to_device();
        joints_.parent.copy_to_device();
        joints_.child.copy_to_device();
        joints_.X_p_pos.copy_to_device();
        joints_.X_p_quat.copy_to_device();
        joints_.X_c_pos.copy_to_device();
        joints_.X_c_quat.copy_to_device();
        joints_.penalty_k_lin.copy_to_device();
        joints_.penalty_k_ang.copy_to_device();
        joints_.lambda_lin.copy_to_device();
        joints_.lambda_ang.copy_to_device();
        joints_.C0_lin.copy_to_device();
        joints_.C0_ang.copy_to_device();
        joints_.is_hard.copy_to_device();
    }

    // --- Joint adjacency ---
    // Count joints per body
    std::vector<std::vector<int>> body_joints(nb);
    for (int j = 0; j < nj; ++j) {
        int p = joint_staging_[j].parent;
        int c = joint_staging_[j].child;
        if (p >= 0) body_joints[p].push_back(j);
        if (c >= 0) body_joints[c].push_back(j);
    }
    int max_jpb = 0;
    for (int i = 0; i < nb; ++i) {
        int sz = static_cast<int>(body_joints[i].size());
        if (sz > max_jpb) max_jpb = sz;
    }
    if (max_jpb > 0) {
        joints_.resize_adjacency(nb, max_jpb);
        for (int i = 0; i < nb; ++i) {
            joints_.body_joint_count[i] = static_cast<int>(body_joints[i].size());
            for (int k = 0; k < static_cast<int>(body_joints[i].size()); ++k) {
                joints_.body_joint_indices[i * max_jpb + k] = body_joints[i][k];
            }
        }
        joints_.body_joint_count.copy_to_device();
        joints_.body_joint_indices.copy_to_device();
    }

    // --- Excluded body pairs for collision filtering ---
    // Bodies directly connected via a joint should not collide.
    {
        std::vector<math::Vec2i> excl;
        for (int j = 0; j < nj; ++j) {
            int p = joint_staging_[j].parent;
            int c = joint_staging_[j].child;
            if (p >= 0 && c >= 0 && p != c) {
                int lo = (p < c) ? p : c;
                int hi = (p < c) ? c : p;
                excl.push_back({lo, hi});
            }
        }
        // Simple insertion sort (n = joint_count, typically small enough)
        for (int i = 1; i < static_cast<int>(excl.size()); ++i) {
            math::Vec2i key = excl[i];
            int k = i - 1;
            while (k >= 0 && (excl[k].x > key.x || (excl[k].x == key.x && excl[k].y > key.y))) {
                excl[k + 1] = excl[k];
                --k;
            }
            excl[k + 1] = key;
        }
        // Deduplicate
        int write = 0;
        for (int i = 0; i < static_cast<int>(excl.size()); ++i) {
            if (i == 0 || excl[i].x != excl[i - 1].x || excl[i].y != excl[i - 1].y) {
                excl[write++] = excl[i];
            }
        }
        excl.resize(write);
        excluded_pair_count_ = static_cast<int>(excl.size());
        if (excluded_pair_count_ > 0) {
            excluded_body_pairs_.resize(excluded_pair_count_);
            for (int i = 0; i < excluded_pair_count_; ++i)
                excluded_body_pairs_[i] = excl[i];
            excluded_body_pairs_.copy_to_device();
        }
    }

    // --- Graph coloring ---
    std::vector<std::pair<int, int>> edges;
    for (int j = 0; j < nj; ++j) {
        int p = joint_staging_[j].parent;
        int c = joint_staging_[j].child;
        if (p >= 0 && c >= 0) edges.emplace_back(p, c);
    }
    auto groups = color_rigid_bodies(nb, edges);
    color_groups_.resize(groups.size());
    for (size_t c = 0; c < groups.size(); ++c) {
        int gs = static_cast<int>(groups[c].size());
        color_groups_[c].resize(gs);
        for (int i = 0; i < gs; ++i) {
            color_groups_[c][i] = groups[c][i];
        }
        color_groups_[c].copy_to_device();
    }

    // --- Contacts ---
    contacts_.resize(contact_buffer_size_, nb, per_body_contact_cap_);

    contact_count_dev_.resize(1);
    contact_count_dev_[0] = 0;
    contact_count_dev_.copy_to_device();

    // --- Broadphase ---
    broadphase_.build(ns, max_broadphase_pairs_);

    // --- Contact matching ---
    if (contact_history_) {
        contact_matcher_.resize(contact_buffer_size_);
    }

    finalized_ = true;
    body_staging_.clear();
    shape_staging_.clear();
    joint_staging_.clear();
}

// ============================================================================
// step()
// ============================================================================

void RigidSimulator::step(float dt, std::uintptr_t cuda_stream) {
    if (!finalized_) throw std::runtime_error("Call finalize() before step()");

    int nb = bodies_.count;
    int ns = shapes_.count;
    int nj = joints_.count;

    // 1. Snapshot body_pos_prev, body_quat_prev
    avbd::launch_snapshot_prev(
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        bodies_.pos_prev.gpu_data(), bodies_.quat_prev.gpu_data(),
        nb, cuda_stream);

    // 2. Broadphase: compute shape AABBs → find overlapping pairs
    broadphase_.query(
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        shapes_.body.gpu_data(), shapes_.geo_type.gpu_data(),
        shapes_.geo_scale.gpu_data(), shapes_.pos_local.gpu_data(),
        shapes_.quat_local.gpu_data(), shapes_.gap.gpu_data(),
        ns, cuda_stream);

    int n_bp_pairs = broadphase_.host_pair_count(cuda_stream);

    // 4. Narrowphase at predicted positions
    narrow_phase_detect(
        broadphase_.pair_list_dev(), n_bp_pairs,
        shapes_.body.gpu_data(), shapes_.geo_type.gpu_data(),
        shapes_.geo_scale.gpu_data(), shapes_.pos_local.gpu_data(),
        shapes_.quat_local.gpu_data(),
        shapes_.ke.gpu_data(), shapes_.kd.gpu_data(),
        shapes_.mu.gpu_data(), shapes_.gap.gpu_data(),
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        excluded_pair_count_ > 0 ? excluded_body_pairs_.gpu_data() : nullptr,
        excluded_pair_count_,
        contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
        contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
        contacts_.normal.gpu_data(),
        contacts_.margin0.gpu_data(), contacts_.margin1.gpu_data(),
        contacts_.material_ke.gpu_data(), contacts_.material_kd.gpu_data(),
        contacts_.material_mu.gpu_data(),
        contact_count_dev_.gpu_data(), contact_buffer_size_,
        cuda_stream);

    // Read back contact count
    contact_count_dev_.copy_to_host(cuda_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
    int n_contacts = contact_count_dev_[0];
    if (n_contacts > contact_buffer_size_) n_contacts = contact_buffer_size_;
    contacts_.contact_count = n_contacts;

    // 5. Build per-body contact lists
    avbd::launch_build_body_contact_list(
        n_contacts, nb, contacts_.per_body_cap,
        contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
        shapes_.body.gpu_data(),
        contacts_.body_contact_counts.gpu_data(),
        contacts_.body_contact_indices.gpu_data(),
        cuda_stream);

    // 6. Init contact AVBD state (or warm-start)
    if (contact_history_ && contacts_.prev_count > 0) {
        contact_matcher_.match_and_warmstart(
            n_contacts,
            contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
            contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
            contacts_.normal.gpu_data(),
            contacts_.margin0.gpu_data(), contacts_.margin1.gpu_data(),
            shapes_.body.gpu_data(),
            bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
            contacts_.penalty_k.gpu_data(), contacts_.lambda.gpu_data(),
            contacts_.C0.gpu_data(), contacts_.stick_flag.gpu_data(),
            contacts_.material_ke.gpu_data(),
            contacts_.prev_count,
            contacts_.prev_shape0.gpu_data(), contacts_.prev_shape1.gpu_data(),
            contacts_.prev_point0.gpu_data(), contacts_.prev_point1.gpu_data(),
            contacts_.prev_normal.gpu_data(),
            contacts_.prev_lambda.gpu_data(), contacts_.prev_penalty_k.gpu_data(),
            contacts_.prev_stick_flag.gpu_data(),
            cuda_stream);
    } else {
        avbd::launch_init_contact_avbd(
            n_contacts,
            contacts_.penalty_k.gpu_data(), contacts_.lambda.gpu_data(),
            contacts_.C0.gpu_data(), contacts_.stick_flag.gpu_data(),
            contacts_.material_ke.gpu_data(),
            cuda_stream);
    }

    // 6. C0 snapshot + lambda/k decay
    avbd::launch_step_C0_lambda_contacts(
        n_contacts, avbd_alpha_, avbd_gamma_,
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
        shapes_.body.gpu_data(),
        contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
        contacts_.normal.gpu_data(), contacts_.margin0.gpu_data(),
        contacts_.margin1.gpu_data(),
        contacts_.C0.gpu_data(), contacts_.penalty_k.gpu_data(),
        contacts_.lambda.gpu_data(),
        cuda_stream);

    avbd::launch_step_C0_lambda_joints(
        nj, avbd_alpha_, avbd_gamma_,
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        joints_.parent.gpu_data(), joints_.child.gpu_data(),
        joints_.X_p_pos.gpu_data(), joints_.X_p_quat.gpu_data(),
        joints_.X_c_pos.gpu_data(), joints_.X_c_quat.gpu_data(),
        joints_.type.gpu_data(),
        joints_.C0_lin.gpu_data(), joints_.C0_ang.gpu_data(),
        joints_.penalty_k_lin.gpu_data(), joints_.penalty_k_ang.gpu_data(),
        joints_.lambda_lin.gpu_data(), joints_.lambda_ang.gpu_data(),
        cuda_stream);

    // 7. Forward step: semi-implicit Euler → inertia target
    avbd::launch_forward_step(
        dt, gravity_,
        bodies_.com.gpu_data(), bodies_.inertia.gpu_data(),
        bodies_.inv_mass.gpu_data(), bodies_.inv_inertia.gpu_data(),
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        bodies_.vel.gpu_data(), bodies_.omega.gpu_data(),
        bodies_.inertia_pos.gpu_data(), bodies_.inertia_quat.gpu_data(),
        nb, cuda_stream);

    // 8. AVBD iterations
    for (int iter = 0; iter < iterations_; ++iter) {
        for (auto& group : color_groups_) {
            int gs = static_cast<int>(group.gpu_size());

            // Zero scratch buffers
            avbd::launch_zero_scratch(
                bodies_.forces.gpu_data(), bodies_.torques.gpu_data(),
                bodies_.hessian_ll.gpu_data(), bodies_.hessian_al.gpu_data(),
                bodies_.hessian_aa.gpu_data(),
                nb, cuda_stream);

            // Accumulate contact forces
            avbd::launch_accumulate_contacts(
                group.gpu_data(), gs,
                dt, avbd_alpha_, friction_epsilon_,
                contact_hard_ ? 1 : 0,
                bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
                bodies_.pos_prev.gpu_data(), bodies_.quat_prev.gpu_data(),
                bodies_.com.gpu_data(), bodies_.inv_mass.gpu_data(),
                contacts_.per_body_cap,
                contacts_.body_contact_counts.gpu_data(),
                contacts_.body_contact_indices.gpu_data(),
                contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
                shapes_.body.gpu_data(),
                contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
                contacts_.normal.gpu_data(),
                contacts_.margin0.gpu_data(), contacts_.margin1.gpu_data(),
                contacts_.penalty_k.gpu_data(), contacts_.material_kd.gpu_data(),
                contacts_.material_mu.gpu_data(),
                contacts_.lambda.gpu_data(), contacts_.C0.gpu_data(),
                bodies_.forces.gpu_data(), bodies_.torques.gpu_data(),
                bodies_.hessian_ll.gpu_data(), bodies_.hessian_al.gpu_data(),
                bodies_.hessian_aa.gpu_data(),
                cuda_stream);

            // Solve per-body 6x6 system
            avbd::launch_solve_rigid_body(
                group.gpu_data(), gs,
                dt, avbd_alpha_,
                bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
                bodies_.inertia_pos.gpu_data(), bodies_.inertia_quat.gpu_data(),
                bodies_.com.gpu_data(), bodies_.mass.gpu_data(),
                bodies_.inv_mass.gpu_data(),
                bodies_.inertia.gpu_data(), bodies_.inv_inertia.gpu_data(),
                bodies_.forces.gpu_data(), bodies_.torques.gpu_data(),
                bodies_.hessian_ll.gpu_data(), bodies_.hessian_al.gpu_data(),
                bodies_.hessian_aa.gpu_data(),
                joints_.max_joints_per_body,
                joints_.body_joint_count.gpu_data(),
                joints_.body_joint_indices.gpu_data(),
                nj,
                joints_.type.gpu_data(), joints_.parent.gpu_data(),
                joints_.child.gpu_data(),
                joints_.X_p_pos.gpu_data(), joints_.X_p_quat.gpu_data(),
                joints_.X_c_pos.gpu_data(), joints_.X_c_quat.gpu_data(),
                joints_.penalty_k_lin.gpu_data(), joints_.penalty_k_ang.gpu_data(),
                joints_.lambda_lin.gpu_data(), joints_.lambda_ang.gpu_data(),
                joints_.C0_lin.gpu_data(), joints_.C0_ang.gpu_data(),
                joints_.is_hard.gpu_data(),
                bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
                cuda_stream);
        }

        // Update duals
        avbd::launch_update_duals_contacts(
            n_contacts, avbd_alpha_, avbd_beta_,
            stick_motion_eps_, contact_hard_ ? 1 : 0,
            bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
            bodies_.pos_prev.gpu_data(), bodies_.quat_prev.gpu_data(),
            bodies_.inv_mass.gpu_data(),
            contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
            shapes_.body.gpu_data(),
            contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
            contacts_.normal.gpu_data(),
            contacts_.margin0.gpu_data(), contacts_.margin1.gpu_data(),
            contacts_.material_ke.gpu_data(), contacts_.material_mu.gpu_data(),
            contacts_.C0.gpu_data(),
            contacts_.penalty_k.gpu_data(), contacts_.lambda.gpu_data(),
            contacts_.stick_flag.gpu_data(),
            cuda_stream);

        avbd::launch_update_duals_joints(
            nj, avbd_alpha_, avbd_gamma_,
            bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
            joints_.type.gpu_data(), joints_.parent.gpu_data(),
            joints_.child.gpu_data(),
            joints_.X_p_pos.gpu_data(), joints_.X_p_quat.gpu_data(),
            joints_.X_c_pos.gpu_data(), joints_.X_c_quat.gpu_data(),
            joints_.is_hard.gpu_data(),
            joints_.C0_lin.gpu_data(), joints_.C0_ang.gpu_data(),
            joints_.penalty_k_lin.gpu_data(), joints_.penalty_k_ang.gpu_data(),
            joints_.lambda_lin.gpu_data(), joints_.lambda_ang.gpu_data(),
            cuda_stream);
    }

    // 9. Snapshot contact history for warm-start
    if (contact_history_) {
        contact_matcher_.snapshot(
            n_contacts,
            contacts_.shape0.gpu_data(), contacts_.shape1.gpu_data(),
            contacts_.point0.gpu_data(), contacts_.point1.gpu_data(),
            contacts_.normal.gpu_data(),
            contacts_.lambda.gpu_data(), contacts_.penalty_k.gpu_data(),
            contacts_.stick_flag.gpu_data(),
            contacts_.prev_shape0.gpu_data(), contacts_.prev_shape1.gpu_data(),
            contacts_.prev_point0.gpu_data(), contacts_.prev_point1.gpu_data(),
            contacts_.prev_normal.gpu_data(),
            contacts_.prev_lambda.gpu_data(), contacts_.prev_penalty_k.gpu_data(),
            contacts_.prev_stick_flag.gpu_data(),
            cuda_stream);
        contacts_.prev_count = n_contacts;
    }

    // 10. Update body velocities
    avbd::launch_update_body_velocity(
        dt, nb,
        bodies_.pos.gpu_data(), bodies_.quat.gpu_data(),
        bodies_.com.gpu_data(),
        bodies_.pos_prev.gpu_data(), bodies_.quat_prev.gpu_data(),
        bodies_.vel.gpu_data(), bodies_.omega.gpu_data(),
        apply_stick_deadzone_ ? 1 : 0,
        stick_motion_eps_, stick_motion_eps_,
        contacts_.per_body_cap,
        contacts_.body_contact_counts.gpu_data(),
        contacts_.body_contact_indices.gpu_data(),
        contacts_.stick_flag.gpu_data(),
        cuda_stream);

    // Synchronize to avoid races with external CUDA/GL consumers.
    if (cuda_stream == 0)
        cudaDeviceSynchronize();
    else
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
}

// ============================================================================
// Accessors
// ============================================================================

void RigidSimulator::get_body_poses(math::Vec3f* pos_out,
                                    math::Quatf* quat_out) const {
    int nb = bodies_.count;
    const_cast<CudaArray<math::Vec3f>&>(bodies_.pos).copy_to_host();
    const_cast<CudaArray<math::Quatf>&>(bodies_.quat).copy_to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < nb; ++i) {
        pos_out[i] = bodies_.pos.cpu_data()[i];
        quat_out[i] = bodies_.quat.cpu_data()[i];
    }
}

void RigidSimulator::get_body_velocities(math::Vec3f* vel_out,
                                         math::Vec3f* omega_out) const {
    int nb = bodies_.count;
    const_cast<CudaArray<math::Vec3f>&>(bodies_.vel).copy_to_host();
    const_cast<CudaArray<math::Vec3f>&>(bodies_.omega).copy_to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < nb; ++i) {
        vel_out[i] = bodies_.vel.cpu_data()[i];
        omega_out[i] = bodies_.omega.cpu_data()[i];
    }
}

}  // namespace rigid
}  // namespace chysx
