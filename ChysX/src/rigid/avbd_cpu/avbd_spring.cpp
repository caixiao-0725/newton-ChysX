// SPDX-License-Identifier: MIT
// Spring force. Adapted from avbd-demo3d.

#include "avbd_solver.h"

namespace chysx {
namespace avbd {

Spring::Spring(Solver* solver, Rigid* bodyA, Rigid* bodyB,
               float3 rA, float3 rB, float stiffness, float rest)
    : Force(solver, bodyA, bodyB), rA(rA), rB(rB),
      rest(rest), stiffness(stiffness) {
    if (this->rest < 0.0f) {
        float3 pA = transform(bodyA->positionLin, bodyA->positionAng, this->rA);
        float3 pB = transform(bodyB->positionLin, bodyB->positionAng, this->rB);
        this->rest = length(pA - pB);
    }
}

void Spring::updatePrimal(Rigid* body, float /*alpha*/,
                          float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                          float3& rhsLin, float3& rhsAng) {
    float3 pA = transform(bodyA->positionLin, bodyA->positionAng, rA);
    float3 pB = transform(bodyB->positionLin, bodyB->positionAng, rB);
    float3 d = pA - pB;
    float dLen = length(d);
    if (dLen <= 1.0e-6f) return;

    float3 n = d / dLen;
    float C = dLen - rest;
    float f = stiffness * C;

    float3 rWorld;
    float3 jLin;
    float3 jAng;
    if (body == bodyA) {
        rWorld = rotate(bodyA->positionAng, rA);
        jLin = n;
        jAng = cross(rWorld, n);
    } else {
        rWorld = rotate(bodyB->positionAng, rB);
        jLin = -n;
        jAng = -cross(rWorld, n);
    }

    float3 F = jLin * f;
    float3 Tau = jAng * f;
    float3x3 Kll = outer(jLin, jLin) * stiffness;
    float3x3 Kla = outer(jAng, jLin) * stiffness;
    float3x3 Kaa = outer(jAng, jAng) * stiffness;

    lhsLin += Kll;
    lhsAng += Kaa;
    lhsCross += Kla;
    rhsLin += F;
    rhsAng += Tau;
}

void Spring::updateDual(float /*alpha*/) {}

}  // namespace avbd
}  // namespace chysx
