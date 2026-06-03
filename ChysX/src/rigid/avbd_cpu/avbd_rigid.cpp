// SPDX-License-Identifier: MIT
// Rigid body construction / destruction. Adapted from avbd-demo3d.

#include "avbd_solver.h"

namespace chysx {
namespace avbd {

Rigid::Rigid(Solver* solver, float3 size, float density, float friction,
             float3 position, float3 velocity)
    : solver(solver), forces(nullptr), next(nullptr),
      positionLin(position), positionAng({0, 0, 0, 1}),
      velocityLin(velocity), velocityAng({0, 0, 0}),
      prevVelocityLin(velocity), size(size), friction(friction) {
    next = solver->bodies;
    solver->bodies = this;

    mass = size.x * size.y * size.z * density;
    moment = float3{
        (size.y * size.y + size.z * size.z) / 12.0f * mass,
        (size.x * size.x + size.z * size.z) / 12.0f * mass,
        (size.x * size.x + size.y * size.y) / 12.0f * mass};
    radius = length(size * 0.5f);
}

Rigid::~Rigid() {
    Rigid** p = &solver->bodies;
    while (*p != this) p = &(*p)->next;
    *p = next;
}

bool Rigid::constrainedTo(Rigid* other) const {
    for (Force* f = forces; f != nullptr; f = f->next)
        if ((f->bodyA == this && f->bodyB == other) ||
            (f->bodyA == other && f->bodyB == this))
            return true;
    return false;
}

}  // namespace avbd
}  // namespace chysx
