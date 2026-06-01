// SPDX-License-Identifier: MIT
// Collision manifold (contact constraint). Adapted from avbd-demo3d.

#include "avbd_solver.h"

#include <cmath>

namespace chysx {
namespace avbd {

Manifold::Manifold(Solver* solver, Rigid* bodyA, Rigid* bodyB)
    : Force(solver, bodyA, bodyB), numContacts(0) {}

bool Manifold::initialize() {
    friction = std::sqrt(bodyA->friction * bodyB->friction);

    Contact newContacts[8] = {};
    int newNumContacts;

    if (gpu_num_contacts_ >= 0) {
        // GPU path: contacts have warm-start data from GPU warmstart kernel
        // (raw post-solver values from prev frame, matched by feature_key).
        newNumContacts = gpu_num_contacts_;
        for (int i = 0; i < newNumContacts; i++)
            newContacts[i] = gpu_new_contacts_[i];
        basis = gpu_basis_;
        gpu_num_contacts_ = -1;
    } else {
        newNumContacts = collide(bodyA, bodyB, newContacts, basis);

        // CPU warm-start matching (only for CPU narrowphase path)
        for (int i = 0; i < newNumContacts; i++) {
            for (int j = 0; j < numContacts; j++) {
                if (newContacts[i].feature.key == contacts[j].feature.key) {
                    float3 newRA = newContacts[i].rA;
                    float3 newRB = newContacts[i].rB;
                    newContacts[i] = contacts[j];
                    if (!contacts[j].stick) {
                        newContacts[i].rA = newRA;
                        newContacts[i].rB = newRB;
                    }
                    break;
                }
            }
        }
    }

    numContacts = newNumContacts;
    for (int i = 0; i < numContacts; i++)
        contacts[i] = newContacts[i];

    for (int i = 0; i < numContacts; i++) {
        float3 xA = transform(bodyA->positionLin, bodyA->positionAng, contacts[i].rA);
        float3 xB = transform(bodyB->positionLin, bodyB->positionAng, contacts[i].rB);
        contacts[i].C0 = basis * (xA - xB) + float3{AVBD_COLLISION_MARGIN, 0, 0};

        // Scale lambda and penalty (both CPU and GPU paths transfer raw values)
        contacts[i].lambda = contacts[i].lambda * solver->alpha * solver->gamma;
        contacts[i].penalty = clamp(contacts[i].penalty * solver->gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
    }

    return numContacts > 0;
}

void Manifold::updatePrimal(Rigid* body, float alpha,
                            float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                            float3& rhsLin, float3& rhsAng) {
    float3 dqALin = bodyA->positionLin - bodyA->initialLin;
    float3 dqAAng = bodyA->positionAng - bodyA->initialAng;
    float3 dqBLin = bodyB->positionLin - bodyB->initialLin;
    float3 dqBAng = bodyB->positionAng - bodyB->initialAng;

    for (int i = 0; i < numContacts; i++) {
        float3 rAWorld = rotate(bodyA->positionAng, contacts[i].rA);
        float3 rBWorld = rotate(bodyB->positionAng, contacts[i].rB);

        float3x3 jALin = basis;
        float3x3 jBLin = -basis;
        float3x3 jAAng = float3x3{cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2])};
        float3x3 jBAng = float3x3{cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2])};

        float3x3 K = diagonal(contacts[i].penalty.x, contacts[i].penalty.y, contacts[i].penalty.z);
        float3 C = contacts[i].C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng;

        float3 F = K * C + contacts[i].lambda;
        F[0] = min(F[0], 0.0f);

        float bounds = std::fabs(F[0]) * friction;
        float frictionScale = length(float2{F[1], F[2]});
        if (frictionScale > bounds && frictionScale > 0) {
            F[1] *= bounds / frictionScale;
            F[2] *= bounds / frictionScale;
        }

        float3x3 jLin = body == bodyA ? jALin : jBLin;
        float3x3 jAng = body == bodyA ? jAAng : jBAng;

        float3x3 jLinT = transpose(jLin);
        float3x3 jAngT = transpose(jAng);
        float3x3 jAngTk = jAngT * K;

        lhsLin += jLinT * K * jLin;
        lhsAng += jAngTk * jAng;
        lhsCross += jAngTk * jLin;

        rhsLin += jLinT * F;
        rhsAng += jAngT * F;
    }
}

void Manifold::updateDual(float alpha) {
    float3 dqALin = bodyA->positionLin - bodyA->initialLin;
    float3 dqAAng = bodyA->positionAng - bodyA->initialAng;
    float3 dqBLin = bodyB->positionLin - bodyB->initialLin;
    float3 dqBAng = bodyB->positionAng - bodyB->initialAng;

    for (int i = 0; i < numContacts; i++) {
        float3 rAWorld = rotate(bodyA->positionAng, contacts[i].rA);
        float3 rBWorld = rotate(bodyB->positionAng, contacts[i].rB);

        float3x3 jALin = basis;
        float3x3 jBLin = -basis;
        float3x3 jAAng = float3x3{cross(rAWorld, jALin[0]), cross(rAWorld, jALin[1]), cross(rAWorld, jALin[2])};
        float3x3 jBAng = float3x3{cross(rBWorld, jBLin[0]), cross(rBWorld, jBLin[1]), cross(rBWorld, jBLin[2])};

        float3x3 K = diagonal(contacts[i].penalty.x, contacts[i].penalty.y, contacts[i].penalty.z);
        float3 C = contacts[i].C0 * (1 - alpha) + jALin * dqALin + jBLin * dqBLin + jAAng * dqAAng + jBAng * dqBAng;

        float3 F = K * C + contacts[i].lambda;
        F[0] = min(F[0], 0.0f);

        float bounds = std::fabs(F[0]) * friction;
        float frictionScale = length(float2{F[1], F[2]});
        if (frictionScale > bounds && frictionScale > 0) {
            F[1] *= bounds / frictionScale;
            F[2] *= bounds / frictionScale;
        }

        contacts[i].lambda = F;

        if (F[0] < 0)
            contacts[i].penalty[0] = min(contacts[i].penalty[0] + solver->betaLin * std::fabs(C[0]), AVBD_PENALTY_MAX);
        if (frictionScale <= bounds) {
            contacts[i].penalty[1] = min(contacts[i].penalty[1] + solver->betaLin * std::fabs(C[1]), AVBD_PENALTY_MAX);
            contacts[i].penalty[2] = min(contacts[i].penalty[2] + solver->betaLin * std::fabs(C[2]), AVBD_PENALTY_MAX);
            contacts[i].stick = length(float2{C[1], C[2]}) < AVBD_STICK_THRESH;
        }
    }
}

}  // namespace avbd
}  // namespace chysx
