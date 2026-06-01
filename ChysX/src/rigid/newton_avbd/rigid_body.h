// SPDX-License-Identifier: Apache-2.0
//
// SoA rigid-body state for the AVBD solver.

#pragma once

#include "../../math/matrix.cuh"
#include "../../math/quat.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace rigid {

using math::Mat3f;
using math::Quatf;
using math::Vec3f;

struct RigidBodyData {
    // --- Constant properties ---
    CudaArray<Vec3f> com;           // center-of-mass offset in body frame [m]
    CudaArray<float> mass;          // scalar mass [kg]
    CudaArray<float> inv_mass;      // 1/m (0 for kinematic/static)
    CudaArray<Mat3f> inertia;       // body-frame inertia tensor [kg·m²]
    CudaArray<Mat3f> inv_inertia;   // body-frame inverse inertia

    // --- Pose (origin + quaternion) ---
    CudaArray<Vec3f> pos;           // world-frame body origin [m]
    CudaArray<Quatf> quat;          // orientation (x,y,z,w)
    CudaArray<Vec3f> vel;           // linear velocity at COM [m/s]
    CudaArray<Vec3f> omega;         // angular velocity [rad/s]

    // --- Scratch per step ---
    CudaArray<Vec3f> pos_prev;      // pose at step start
    CudaArray<Quatf> quat_prev;
    CudaArray<Vec3f> inertia_pos;   // forward-integrated inertial target
    CudaArray<Quatf> inertia_quat;

    // --- Scratch per iteration (zeroed per color group) ---
    CudaArray<Vec3f> forces;        // accumulated contact/external forces
    CudaArray<Vec3f> torques;       // accumulated contact/external torques
    CudaArray<Mat3f> hessian_ll;    // 3×3 linear-linear block
    CudaArray<Mat3f> hessian_al;    // 3×3 angular-linear coupling
    CudaArray<Mat3f> hessian_aa;    // 3×3 angular-angular block

    int count = 0;

    void resize(int n) {
        count = n;
        com.resize(n); mass.resize(n); inv_mass.resize(n);
        inertia.resize(n); inv_inertia.resize(n);
        pos.resize(n); quat.resize(n); vel.resize(n); omega.resize(n);
        pos_prev.resize(n); quat_prev.resize(n);
        inertia_pos.resize(n); inertia_quat.resize(n);
        forces.resize(n); torques.resize(n);
        hessian_ll.resize(n); hessian_al.resize(n); hessian_aa.resize(n);
    }
};

}  // namespace rigid
}  // namespace chysx
