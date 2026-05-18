// SPDX-License-Identifier: Apache-2.0
//
// Shared device helpers for IPC-style smooth Coulomb friction
// (Li et al. 2020 "Incremental Potential Contact").  Used by both
// `static_contact.cu` (cloth vs static planes/boxes) and
// `self_collision_constraint.cu` / `contact_spmv.cu` (cloth vs cloth,
// VF + EE pairs).
//
// The Lagged-Newton linearisation of IPC isotropic friction reads
//
//     f_t   = -α · dx_t,
//     α     = μ · f_n · f1_SF_over_x(‖u_t^lag‖)
//
// where `u_t^lag = (v - n (n·v)) · dt` is the tangential slip
// accumulated over the previous step, projected onto the contact
// tangent plane.  `f1_SF_over_x` is the slip-aware smoothing of the
// `1/‖u_t‖` factor: linear ramp from `2/ε_u` at `‖u_t‖ = 0` to
// `1/ε_u` at `‖u_t‖ = ε_u`, then plain `1/‖u_t‖` past the
// regularisation band.  This caps `‖f_t‖` exactly at `μ · f_n` once
// `‖u_t‖ ≥ ε_u` (the Coulomb cone) and gives a smooth, monotone
// transition through standstill — Newton-friendly.

#pragma once

namespace chysx {
namespace collision {

__device__ inline float ipc_f1_sf_over_x(float u_norm, float eps_u) {
    if (u_norm > eps_u) {
        return 1.0f / u_norm;
    }
    // Linear ramp on [0, eps_u]: value 2/eps_u at u_norm = 0, value
    // 1/eps_u at u_norm = eps_u (matches the outer branch so the
    // function is C0-continuous across the band edge).
    return (-u_norm / eps_u + 2.0f) / eps_u;
}

}  // namespace collision
}  // namespace chysx
