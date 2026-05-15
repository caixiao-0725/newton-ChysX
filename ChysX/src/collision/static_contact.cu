// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::StaticContactSet.
//
// One thread per particle; the per-particle work is to walk the
// (small) static-shape table, pick the deepest penetration, cache
// (depth, normal) and -- if velocities were provided -- the lagged
// tangential slip `u_t = (v - n (n·v)) * dt`, then have follow-up
// scatter passes push the resulting penalty + IPC-friction
// contributions into the cloth simulator's `rhs` and `H_.diag`
// arrays.

#include "static_contact.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::StaticContactSet: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// Plane SDF.  `n` is unit; signed distance is `dot(n, x) + d`,
// outward normal is just `n` (independent of x).
__device__ inline float plane_signed_dist(const math::Vec3f& p,
                                          const PlaneShape&  pl,
                                          math::Vec3f&       out_n) {
    out_n = pl.n;
    return dot(pl.n, p) + pl.d;
}

// OBB SDF.  Returns positive distance outside, negative inside,
// and writes the unit world-space outward normal into `out_n` (for
// points strictly inside, this points toward the closest face, which
// is what the penalty wants — it's the direction we want to push the
// particle to leave the box).
//
// Standard formula from Ericson, "Real-Time Collision Detection".
__device__ inline float box_signed_dist(const math::Vec3f& p,
                                        const BoxShape&    b,
                                        math::Vec3f&       out_n) {
    const math::Vec3f rel = p - b.center;
    const float qx = dot(rel, b.ex);
    const float qy = dot(rel, b.ey);
    const float qz = dot(rel, b.ez);

    const float dx = fabsf(qx) - b.half_ext.x;
    const float dy = fabsf(qy) - b.half_ext.y;
    const float dz = fabsf(qz) - b.half_ext.z;

    if (dx <= 0.0f && dy <= 0.0f && dz <= 0.0f) {
        // Inside the box.  All dᵢ are ≤ 0; the closest face is the
        // axis with the *largest* dᵢ (smallest absolute penetration).
        // signed_dist is that dᵢ (negative).
        if (dx >= dy && dx >= dz) {
            out_n = (qx >= 0.0f) ? b.ex : (b.ex * -1.0f);
            return dx;
        } else if (dy >= dz) {
            out_n = (qy >= 0.0f) ? b.ey : (b.ey * -1.0f);
            return dy;
        } else {
            out_n = (qz >= 0.0f) ? b.ez : (b.ez * -1.0f);
            return dz;
        }
    }

    // Outside the box.  Distance to surface = ‖max(d, 0)‖₂.  The
    // outward normal in box-local coords is (sign(qᵢ) * max(dᵢ, 0)) /
    // dist; rotate back into world via R.
    const float cx = (dx > 0.0f) ? dx : 0.0f;
    const float cy = (dy > 0.0f) ? dy : 0.0f;
    const float cz = (dz > 0.0f) ? dz : 0.0f;
    const float dist = sqrtf(cx * cx + cy * cy + cz * cz);
    const float inv_d = 1.0f / fmaxf(dist, 1.0e-30f);
    const float sx = (qx > 0.0f) ?  cx : -cx;
    const float sy = (qy > 0.0f) ?  cy : -cy;
    const float sz = (qz > 0.0f) ?  cz : -cz;
    out_n = b.ex * (sx * inv_d) + b.ey * (sy * inv_d) + b.ez * (sz * inv_d);
    return dist;
}

// One thread per particle.  Walks every plane and box, picks the
// deepest penetration (largest `thickness - signed_dist`), and writes
// the resulting (n, depth) into `contacts[p]`.  When `velocities` is
// non-null we additionally compute the lagged tangential slip
// `u_t = (v - n (n·v)) * dt` and stash it (with its norm) in
// `slips[p]`.  depth ≤ 0 means "no active contact" and the downstream
// scatter passes skip it.
__global__ void detect_kernel(
    const float3* __restrict__       positions,
    const float3* __restrict__       velocities,   // may be nullptr
    int                              n_particles,
    const PlaneShape* __restrict__   planes,
    int                              n_planes,
    const BoxShape* __restrict__     boxes,
    int                              n_boxes,
    float                            thickness,
    float                            dt,
    math::Vec4f* __restrict__        contacts,
    math::Vec4f* __restrict__        slips) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const float3 pf = positions[p];
    const math::Vec3f x(pf.x, pf.y, pf.z);

    float       best_depth = 0.0f;
    math::Vec3f best_n(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < n_planes; ++i) {
        math::Vec3f n;
        const float d = plane_signed_dist(x, planes[i], n);
        const float depth = thickness - d;
        if (depth > best_depth) {
            best_depth = depth;
            best_n = n;
        }
    }

    for (int i = 0; i < n_boxes; ++i) {
        math::Vec3f n;
        const float d = box_signed_dist(x, boxes[i], n);
        const float depth = thickness - d;
        if (depth > best_depth) {
            best_depth = depth;
            best_n = n;
        }
    }

    contacts[p] = math::Vec4f(best_n.x, best_n.y, best_n.z, best_depth);

    // Tangential slip from previous step: project (v * dt) onto the
    // contact tangent plane.  When the particle is not in contact
    // (best_depth == 0) the value is irrelevant, but we still write
    // zero to keep downstream passes branch-free.
    if (slips != nullptr) {
        math::Vec3f u_t(0.0f, 0.0f, 0.0f);
        float       u_norm = 0.0f;
        if (best_depth > 0.0f && velocities != nullptr && dt > 0.0f) {
            const float3 vf = velocities[p];
            const math::Vec3f v(vf.x, vf.y, vf.z);
            const math::Vec3f dxv = v * dt;
            const float vn = dot(best_n, dxv);
            u_t = dxv - best_n * vn;
            u_norm = sqrtf(u_t.x * u_t.x + u_t.y * u_t.y + u_t.z * u_t.z);
        }
        slips[p] = math::Vec4f(u_t.x, u_t.y, u_t.z, u_norm);
    }
}

// IPC-style smoothing of `1 / ‖u_t‖`.  Linear ramp inside the
// regularisation band so that `‖f_t‖ → 0` as `‖u_t‖ → 0` (no spurious
// tangential force at standstill); plain `1/‖u_t‖` outside the band
// so the resulting force saturates at `μ · f_n` for any meaningful
// slip.  Matches `compute_projected_isotropic_friction` in the VBD
// kernels (newton/_src/solvers/vbd/rigid_vbd_kernels.py).
__device__ inline float f1_sf_over_x(float u_norm, float eps_u) {
    if (u_norm > eps_u) {
        return 1.0f / u_norm;
    }
    // Linear ramp on [0, eps_u]: at u_norm = 0 the value is 2/eps_u
    // (largest), at u_norm = eps_u it matches the outer branch
    // (1/eps_u) so the function is C0-continuous.
    return (-u_norm / eps_u + 2.0f) / eps_u;
}

// rhs[p] += -k * depth * n  +  (-α * u_t^lag).
//
// No atomic — each particle gets exactly one contact at most so there
// is no cross-thread contention; the pre-existing rhs value (gradient
// from elastic / pin / etc.) gets read once and written once per
// particle.
//
// MODIFIED: Now includes tangential friction force in RHS for improved
// stability.  The tangential force is `-α · u_t^lag` where `u_t^lag`
// is the lagged tangential slip from the previous step (computed in
// `detect_kernel`).  This provides explicit damping based on the
// previous motion, while the diagonal block `α · (I - n n^T)` in
// `bake_diag_kernel` provides implicit stiffness.  Together they form
// a spring-damper system that is more stable than stiffness-only.
__global__ void scatter_gradient_kernel(
    const math::Vec4f* __restrict__ contacts,
    const math::Vec4f* __restrict__ slips,
    int                             n_particles,
    float                           stiffness,
    float                           friction_mu,
    float                           friction_epsilon,
    math::Vec3f* __restrict__       rhs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    // Normal force: -k * depth * n
    const float kd = -stiffness * depth;
    rhs[p].x += kd * c.x;
    rhs[p].y += kd * c.y;
    rhs[p].z += kd * c.z;

    // Tangential friction force: -α * u_t^lag
    if (friction_mu > 0.0f && slips != nullptr) {
        const float f_n = stiffness * depth;
        const math::Vec4f slip = slips[p];
        const float u_norm = slip.w;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        rhs[p].x += -alpha * slip.x;
        rhs[p].y += -alpha * slip.y;
        rhs[p].z += -alpha * slip.z;
    }
}

// diag[p] += k_n * (n n^T)  +  α_p * (I - n n^T).
//
//  - Normal block:    Gauss-Newton block of the penalty energy
//    `(1/2) k_n (h - d)^2`, see static_contact.h.
//  - Tangential block: Lagged-Newton linearisation of IPC isotropic
//    Coulomb friction (Li et al. 2020).  The unknown is the cloth
//    displacement `dx`; the friction force at the solution is
//    `f_t = -α · dx_t` with `α = μ · f_n · f1_SF_over_x(‖u_t,lag‖)`,
//    self-bounded by `‖f_t‖ ≤ μ · f_n` because `f1_SF_over_x` decays
//    like `1/‖u_t‖` past the regularisation band.  Skipped when
//    `friction_mu == 0` or `slips == nullptr`.
//
// Same single-writer-per-particle property as the gradient pass --
// no atomics needed.  We bake the full 3x3 rather than only the
// upper triangle so the BlockCSR3 SpMV path doesn't need to know
// about symmetry.
__global__ void bake_diag_kernel(
    const math::Vec4f* __restrict__ contacts,
    const math::Vec4f* __restrict__ slips,        // may be nullptr
    int                             n_particles,
    float                           stiffness,
    float                           friction_mu,
    float                           friction_epsilon,
    math::Mat3f* __restrict__       diag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_particles) return;

    const math::Vec4f c = contacts[p];
    const float depth = c.w;
    if (depth <= 0.0f) return;

    const float k  = stiffness;
    const float nx = c.x;
    const float ny = c.y;
    const float nz = c.z;

    // Normal Gauss-Newton block: k_n * (n n^T).
    float a00 = k * nx * nx;
    float a01 = k * nx * ny;
    float a02 = k * nx * nz;
    float a11 = k * ny * ny;
    float a12 = k * ny * nz;
    float a22 = k * nz * nz;

    // Tangential IPC-Coulomb friction block: α * (I - n n^T).
    if (friction_mu > 0.0f) {
        const float f_n = k * depth;        // normal load magnitude
        // Without a slip cache we treat the particle as fully stuck
        // (u_norm = 0): α defaults to its sticking limit
        // `μ · f_n / (eps_u / 2)`, which still respects the Coulomb
        // bound for any `‖dx_t‖ ≤ eps_u`.
        const float u_norm = (slips != nullptr) ? slips[p].w : 0.0f;
        const float f1 = f1_sf_over_x(u_norm, friction_epsilon);
        const float alpha = friction_mu * f_n * f1;
        a00 += alpha * (1.0f - nx * nx);
        a01 += alpha * (-nx * ny);
        a02 += alpha * (-nx * nz);
        a11 += alpha * (1.0f - ny * ny);
        a12 += alpha * (-ny * nz);
        a22 += alpha * (1.0f - nz * nz);
    }

    math::Mat3f& A = diag[p];
    A.data[0] += a00;
    A.data[1] += a01;
    A.data[2] += a02;
    A.data[3] += a01;       // symmetric
    A.data[4] += a11;
    A.data[5] += a12;
    A.data[6] += a02;       // symmetric
    A.data[7] += a12;       // symmetric
    A.data[8] += a22;
}

}  // namespace

void StaticContactSet::clear() {
    n_planes_     = 0;
    n_boxes_      = 0;
    shapes_dirty_ = true;
    planes_.clear();
    boxes_.clear();
}

void StaticContactSet::add_plane(const PlaneShape& p) {
    // Tiny shape counts (typically <10) make full reallocation cheap
    // and avoid the need for a custom growable container.
    const int new_n = n_planes_ + 1;
    CudaArray<PlaneShape> next(static_cast<std::size_t>(new_n));
    if (n_planes_ > 0) {
        std::memcpy(next.cpu_data(), planes_.cpu_data(),
                    static_cast<std::size_t>(n_planes_) * sizeof(PlaneShape));
    }
    next[static_cast<std::size_t>(n_planes_)] = p;
    planes_ = std::move(next);
    n_planes_ = new_n;
    shapes_dirty_ = true;
}

void StaticContactSet::add_box(const BoxShape& b) {
    const int new_n = n_boxes_ + 1;
    CudaArray<BoxShape> next(static_cast<std::size_t>(new_n));
    if (n_boxes_ > 0) {
        std::memcpy(next.cpu_data(), boxes_.cpu_data(),
                    static_cast<std::size_t>(n_boxes_) * sizeof(BoxShape));
    }
    next[static_cast<std::size_t>(n_boxes_)] = b;
    boxes_ = std::move(next);
    n_boxes_ = new_n;
    shapes_dirty_ = true;
}

void StaticContactSet::upload_shapes_() {
    if (n_planes_ > 0) planes_.copy_to_device();
    if (n_boxes_ > 0)  boxes_.copy_to_device();
    shapes_dirty_ = false;
}

void StaticContactSet::detect(const math::Vec3f* positions,
                              int                n_particles,
                              std::uintptr_t     cuda_stream,
                              const math::Vec3f* velocities,
                              float              dt) {
    if (!active() || n_particles <= 0) return;
    if (positions == nullptr) {
        throw std::invalid_argument(
            "chysx::collision::StaticContactSet::detect: positions must be "
            "non-null");
    }

    if (shapes_dirty_) {
        upload_shapes_();
    }
    if (cached_n_particles_ != n_particles) {
        contacts_.allocate_device(static_cast<std::size_t>(n_particles));
        cached_n_particles_ = n_particles;
    }

    const bool need_slip =
        (friction_ > 0.0f) && (velocities != nullptr) && (dt > 0.0f);
    if (need_slip &&
        slips_.gpu_size() != static_cast<std::size_t>(n_particles)) {
        slips_.allocate_device(static_cast<std::size_t>(n_particles));
    }
    cached_has_slip_ = need_slip;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    detect_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        reinterpret_cast<const float3*>(positions),
        reinterpret_cast<const float3*>(velocities),
        n_particles,
        planes_.gpu_data(),
        n_planes_,
        boxes_.gpu_data(),
        n_boxes_,
        thickness_,
        dt,
        contacts_.gpu_data(),
        need_slip ? slips_.gpu_data() : nullptr);
    check_cuda(cudaGetLastError(), "detect_kernel launch");
}

void StaticContactSet::accumulate_gradient(math::Vec3f*    rhs,
                                           int             n_particles,
                                           std::uintptr_t  cuda_stream) const {
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const math::Vec4f* slip_ptr =
        (friction_ > 0.0f && cached_has_slip_) ? slips_.gpu_data() : nullptr;
    scatter_gradient_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        slip_ptr,
        n_particles,
        stiffness_,
        friction_,
        friction_epsilon_,
        rhs);
    check_cuda(cudaGetLastError(), "scatter_gradient_kernel launch");
}

void StaticContactSet::bake_diag(math::Mat3f*   diag,
                                 int             n_particles,
                                 float           dt,
                                 std::uintptr_t  cuda_stream) const {
    (void)dt;  // unused: Coulomb friction is dt-free (lagged slip)
    if (!active() || n_particles <= 0) return;
    if (cached_n_particles_ != n_particles) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const math::Vec4f* slip_ptr =
        (friction_ > 0.0f && cached_has_slip_) ? slips_.gpu_data() : nullptr;
    bake_diag_kernel<<<grid_for(n_particles), kBlockDim, 0, stream>>>(
        contacts_.gpu_data(),
        slip_ptr,
        n_particles,
        stiffness_,
        friction_,
        friction_epsilon_,
        diag);
    check_cuda(cudaGetLastError(), "bake_diag_kernel launch");
}

}  // namespace collision
}  // namespace chysx
