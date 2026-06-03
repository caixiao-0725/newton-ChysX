// SPDX-License-Identifier: MIT
// Force linked-list management. Adapted from avbd-demo3d.

#include "avbd_solver.h"

namespace chysx {
namespace avbd {

Force::Force(Solver* solver, Rigid* bodyA, Rigid* bodyB)
    : solver(solver), bodyA(bodyA), bodyB(bodyB), nextA(nullptr), nextB(nullptr) {
    next = solver->forces;
    solver->forces = this;

    if (bodyA) {
        nextA = bodyA->forces;
        bodyA->forces = this;
    }
    if (bodyB) {
        nextB = bodyB->forces;
        bodyB->forces = this;
    }
}

Force::~Force() {
    Force** p = &solver->forces;
    while (*p != this) p = &(*p)->next;
    *p = next;

    if (bodyA) {
        p = &bodyA->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyA ? &(*p)->nextA : &(*p)->nextB;
        *p = nextA;
    }

    if (bodyB) {
        p = &bodyB->forces;
        while (*p != this)
            p = (*p)->bodyA == bodyB ? &(*p)->nextA : &(*p)->nextB;
        *p = nextB;
    }
}

}  // namespace avbd
}  // namespace chysx
