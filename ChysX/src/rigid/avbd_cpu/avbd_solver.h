// SPDX-License-Identifier: MIT
// AVBD rigid-body solver types.
// Adapted from avbd-demo3d (Chris Giles, 2026).

#pragma once

#include "avbd_maths.h"

#include <vector>

namespace chysx {
namespace avbd {

#define AVBD_PENALTY_MIN 1.0f
#define AVBD_PENALTY_MAX 10000000000.0f
#define AVBD_COLLISION_MARGIN 0.01f
#define AVBD_STICK_THRESH 0.00001f

struct Rigid;
struct Force;
struct Manifold;
struct Solver;
struct GpuManifold;
class BroadphaseGPU;
class NarrowphaseGPU;
class GraphColoringGPU;
class GpuSolver;

struct SoaData {
    int count = 0;
    std::vector<Rigid*> body_ptrs;

    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> quat_x, quat_y, quat_z, quat_w;
    std::vector<float> half_x, half_y, half_z;
    std::vector<float> radius;
    std::vector<float> mass;
    std::vector<float> friction;
    std::vector<float> moment_x, moment_y, moment_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> velang_x, velang_y, velang_z;
    std::vector<float> prevvel_x, prevvel_y, prevvel_z;

    void pack(Rigid* bodies);
    void unpack(Rigid* bodies);
};

struct Rigid {
    Solver* solver;
    Force* forces;
    Rigid* next;
    float3 positionLin;
    quat positionAng;
    float3 initialLin;
    quat initialAng;
    float3 inertialLin;
    quat inertialAng;
    float3 velocityLin;
    float3 velocityAng;
    float3 prevVelocityLin;
    float3 size;
    float mass;
    float3 moment;
    float friction;
    float radius;

    Rigid(Solver* solver, float3 size, float density, float friction,
          float3 position, float3 velocity = float3{0, 0, 0});
    ~Rigid();

    bool constrainedTo(Rigid* other) const;
};

struct Force {
    Solver* solver;
    Rigid* bodyA;
    Rigid* bodyB;
    Force* nextA;
    Force* nextB;
    Force* next;

    Force(Solver* solver, Rigid* bodyA, Rigid* bodyB);
    virtual ~Force();

    virtual bool initialize() = 0;
    virtual void updatePrimal(Rigid* body, float alpha,
                              float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                              float3& rhsLin, float3& rhsAng) = 0;
    virtual void updateDual(float alpha) = 0;
};

struct Joint : Force {
    float3 rA, rB;
    float3 C0Lin, C0Ang;
    float3 penaltyLin, penaltyAng;
    float3 lambdaLin, lambdaAng;
    float stiffnessLin, stiffnessAng, fracture;
    float torqueArm;
    bool broken;

    Joint(Solver* solver, Rigid* bodyA, Rigid* bodyB, float3 rA, float3 rB,
          float stiffnessLin = INFINITY, float stiffnessAng = 0.0f, float fracture = INFINITY);

    bool initialize() override;
    void updatePrimal(Rigid* body, float alpha,
                      float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                      float3& rhsLin, float3& rhsAng) override;
    void updateDual(float alpha) override;
};

struct Spring : Force {
    float3 rA, rB;
    float rest;
    float stiffness;

    Spring(Solver* solver, Rigid* bodyA, Rigid* bodyB, float3 rA, float3 rB,
           float stiffness, float rest = -1);

    bool initialize() override { return true; }
    void updatePrimal(Rigid* body, float alpha,
                      float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                      float3& rhsLin, float3& rhsAng) override;
    void updateDual(float alpha) override;
};

struct IgnoreCollision : Force {
    IgnoreCollision(Solver* solver, Rigid* bodyA, Rigid* bodyB)
        : Force(solver, bodyA, bodyB) {}

    bool initialize() override { return true; }
    void updatePrimal(Rigid*, float, float3x3&, float3x3&, float3x3&, float3&, float3&) override {}
    void updateDual(float) override {}
};

struct Manifold : Force {
    union FeaturePair {
        struct { char inR; char outR; char inI; char outI; };
        int key;
    };

    struct Contact {
        FeaturePair feature;
        float3 rA;
        float3 rB;
        float3 C0;
        float3 penalty;
        float3 lambda;
        bool stick;
    };

    Contact contacts[8];
    float3x3 basis;
    int numContacts;
    float friction;

    // When GPU narrowphase provides collision results, they are stored here
    // and used by initialize() instead of calling collide() on CPU.
    Contact gpu_new_contacts_[8];
    float3x3 gpu_basis_;
    int gpu_num_contacts_ = -1;  // -1 means no GPU result available

    Manifold(Solver* solver, Rigid* bodyA, Rigid* bodyB);

    bool initialize() override;
    void updatePrimal(Rigid* body, float alpha,
                      float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                      float3& rhsLin, float3& rhsAng) override;
    void updateDual(float alpha) override;

    static int collide(Rigid* bodyA, Rigid* bodyB, Contact* contacts, float3x3& basis);
};

struct Solver {
    float dt;
    float gravity;
    int iterations;

    float alpha;
    float betaLin;
    float betaAng;
    float gamma;

    // Ground plane: z = ground_z, normal = +Z.
    // When enabled, bodies are prevented from penetrating below this plane.
    bool  has_ground_plane = false;
    float ground_z = 0.0f;
    float ground_friction = 0.5f;

    void set_ground_plane(float z, float friction = 0.5f) {
        has_ground_plane = true;
        ground_z = z;
        ground_friction = friction;
    }

    // Sphere collider: a dynamic body that bypasses broadphase.
    // The sphere Rigid must be created FIRST so it sits at the end of
    // the packed SoA (index = body_count - 1).
    bool  has_sphere_collider = false;
    float sphere_radius = 0.0f;
    float sphere_friction = 0.5f;

    Rigid* bodies;
    Force* forces;

    SoaData soa_;
    BroadphaseGPU* broadphase_gpu_ = nullptr;
    NarrowphaseGPU* narrowphase_gpu_ = nullptr;
    GraphColoringGPU* graph_coloring_gpu_ = nullptr;
    GpuSolver* gpu_solver_ = nullptr;
    std::vector<int> pairs_a_, pairs_b_;

    // When true, the last step() used the GPU solver path and
    // gpu_solver_ contains up-to-date body state on the device.
    bool gpu_state_valid_ = false;

    // Tracks whether the previous frame left valid GPU state, so we can
    // skip the full CPU→GPU upload on subsequent frames.
    bool gpu_state_valid_prev_ = false;
    int prev_body_count_ = 0;

    Solver();
    ~Solver();

    Rigid* pick(float3 origin, float3 dir, float3& local);
    void clear();
    void defaultParams();
    void step();

    GpuSolver* gpu_solver() { return gpu_solver_; }
};

}  // namespace avbd
}  // namespace chysx
