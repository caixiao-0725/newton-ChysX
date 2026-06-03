// SPDX-License-Identifier: Apache-2.0
//
// RigidSimulator: top-level AVBD rigid-body solver.

#pragma once

#include "../../math/matrix.cuh"
#include "../../math/quat.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

#include "contact_matching.h"
#include "rigid_body.h"
#include "rigid_broadphase.h"
#include "rigid_contact.h"
#include "rigid_joint.h"
#include "rigid_shape.h"

#include <cstdint>
#include <vector>

namespace chysx {
namespace rigid {

enum class JointTypeArg : int { Ball = 0, Fixed = 1 };

class RigidSimulator {
public:
    RigidSimulator() = default;

    // --- Scene construction (call before finalize) --------------------------

    int add_body(float mass, const math::Mat3f& inertia, const math::Vec3f& com,
                 const math::Vec3f& pos, const math::Quatf& quat);

    int add_shape_sphere(int body, float radius,
                         float ke = 1e4f, float kd = 10.f, float mu = 0.5f,
                         float gap = 0.01f);

    int add_shape_box(int body, const math::Vec3f& half_extents,
                      float ke = 1e4f, float kd = 10.f, float mu = 0.5f,
                      float gap = 0.01f);

    int add_shape_capsule(int body, float radius, float half_height,
                          float ke = 1e4f, float kd = 10.f, float mu = 0.5f,
                          float gap = 0.01f);

    void add_ground_plane(float ke = 1e4f, float kd = 10.f, float mu = 0.5f);

    int add_joint(JointTypeArg type, int parent, int child,
                  const math::Vec3f& anchor_p, const math::Quatf& frame_p,
                  const math::Vec3f& anchor_c, const math::Quatf& frame_c);

    // Upload CPU data to GPU and build adjacency/coloring.
    void finalize();

    // --- Stepping -----------------------------------------------------------

    void step(float dt, std::uintptr_t cuda_stream = 0);

    // --- Accessors ----------------------------------------------------------

    int body_count() const { return bodies_.count; }
    int shape_count() const { return shapes_.count; }
    int joint_count() const { return joints_.count; }
    int contact_count() const { return contacts_.contact_count; }

    // Copy body poses from device to host-side numpy-compatible arrays.
    // Caller must provide arrays of at least body_count elements.
    void get_body_poses(math::Vec3f* pos_out, math::Quatf* quat_out) const;
    void get_body_velocities(math::Vec3f* vel_out, math::Vec3f* omega_out) const;

    // --- Parameters ---------------------------------------------------------

    void set_iterations(int n) { iterations_ = n; }
    void set_gravity(const math::Vec3f& g) { gravity_ = g; }
    void set_contact_hard(bool hard) { contact_hard_ = hard; }
    void set_contact_history(bool enabled) { contact_history_ = enabled; }
    void set_avbd_alpha(float a) { avbd_alpha_ = a; }
    void set_avbd_gamma(float g) { avbd_gamma_ = g; }
    void set_avbd_beta(float b) { avbd_beta_ = b; }
    void set_friction_epsilon(float e) { friction_epsilon_ = e; }
    void set_stick_motion_eps(float e) { stick_motion_eps_ = e; }
    void set_stick_deadzone(bool e) { apply_stick_deadzone_ = e; }
    void set_contact_buffer_size(int n) { contact_buffer_size_ = n; }
    void set_per_body_contact_capacity(int n) { per_body_contact_cap_ = n; }
    void set_max_broadphase_pairs(int n) { max_broadphase_pairs_ = n; }

private:
    // --- CPU-side staging during construction ---
    struct BodyStaging {
        float mass; math::Mat3f inertia; math::Vec3f com;
        math::Vec3f pos; math::Quatf quat;
    };
    struct ShapeStaging {
        int body; int geo_type; math::Vec3f geo_scale;
        math::Vec3f pos_local; math::Quatf quat_local;
        float ke, kd, mu, gap;
    };
    struct JointStaging {
        int type, parent, child;
        math::Vec3f X_p_pos; math::Quatf X_p_quat;
        math::Vec3f X_c_pos; math::Quatf X_c_quat;
    };
    std::vector<BodyStaging>  body_staging_;
    std::vector<ShapeStaging> shape_staging_;
    std::vector<JointStaging> joint_staging_;
    bool finalized_ = false;

    // --- GPU-side data ---
    RigidBodyData    bodies_;
    RigidShapeData   shapes_;
    RigidJointData   joints_;
    RigidContactData contacts_;

    // Contact matching
    ContactMatcher   contact_matcher_;

    // Broadphase
    RigidBroadphase broadphase_;

    // Per-contact count device buffer (single int)
    CudaArray<int> contact_count_dev_;

    // Excluded body pairs for collision filtering (joint-connected bodies)
    CudaArray<math::Vec2i> excluded_body_pairs_;
    int excluded_pair_count_ = 0;

    // Coloring
    std::vector<CudaArray<int>> color_groups_;

    // --- Parameters ---
    math::Vec3f gravity_{0.f, -9.81f, 0.f};
    int iterations_             = 5;
    bool contact_hard_          = true;
    bool contact_history_       = false;
    float avbd_alpha_           = 0.95f;
    float avbd_gamma_           = 0.999f;
    float avbd_beta_            = 0.f;
    float friction_epsilon_     = 1e-3f;
    float stick_motion_eps_     = 1e-4f;
    bool apply_stick_deadzone_  = true;
    int contact_buffer_size_    = 4096;
    int per_body_contact_cap_   = 64;
    int max_broadphase_pairs_   = 65536;
};

}  // namespace rigid
}  // namespace chysx
