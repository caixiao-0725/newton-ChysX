// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::solver::PCGSolver.

#include "pcg_solver.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../collision/zero_count.h"
#include "../profile/nvtx_range.h"

namespace chysx {
namespace solver {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::solver::PCGSolver: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 256;

inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void invert_diag_kernel(const math::Mat3f* __restrict__ A_diag,
                                   math::Mat3f* __restrict__ M_inv,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    M_inv[i] = math::inverse(A_diag[i]);
}

__global__ void apply_jacobi_kernel(const math::Mat3f* __restrict__ M_inv,
                                    const math::Vec3f* __restrict__ in,
                                    math::Vec3f* __restrict__ out,
                                    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = M_inv[i] * in[i];
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    constexpr int kNumWarps = BLOCK / 32;
    val = (threadIdx.x < kNumWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template <int BLOCK>
__global__ void dot_kernel(const math::Vec3f* __restrict__ a,
                           const math::Vec3f* __restrict__ b,
                           float* __restrict__ out,
                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        const math::Vec3f av = a[i];
        const math::Vec3f bv = b[i];
        local = av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) atomicAdd(out, bsum);
}

__global__ void axpy_dev_kernel(const float* __restrict__ alpha_ptr,
                                const math::Vec3f* __restrict__ x,
                                math::Vec3f* __restrict__ y,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = *alpha_ptr;
    y[i] = y[i] + x[i] * alpha;
}

__global__ void naxpy_dev_kernel(const float* __restrict__ alpha_ptr,
                                 const math::Vec3f* __restrict__ x,
                                 math::Vec3f* __restrict__ y,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float alpha = *alpha_ptr;
    y[i] = y[i] - x[i] * alpha;
}

__global__ void update_p_kernel(const float* __restrict__ beta_ptr,
                                const math::Vec3f* __restrict__ z,
                                math::Vec3f* __restrict__ p,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float beta = *beta_ptr;
    p[i] = z[i] + p[i] * beta;
}

__global__ void scalar_div_kernel(const float* a, const float* b, float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float bv = *b;
        *out = (bv > 1e-37f || bv < -1e-37f) ? (*a / bv) : 0.0f;
    }
}

__global__ void scalar_copy_kernel(const float* src, float* dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}

// Issue all PCG kernels onto `stream`.
void emit_pcg(const sparse::BlockCSR3& A,
              DeviceSpan<math::Vec3f> b,
              DeviceSpan<math::Vec3f> x,
              int max_iterations,
              std::uintptr_t cuda_stream,
              const collision::ContactSpMVOp& contact,
              CudaArray<math::Vec3f>& r,
              CudaArray<math::Vec3f>& p,
              CudaArray<math::Vec3f>& z,
              CudaArray<math::Vec3f>& Ap,
              CudaArray<math::Mat3f>& M_inv,
              CudaArray<float>& coeff) {
    const int n = A.num_block_rows();
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int grid = grid_for(n);

    invert_diag_kernel<<<grid, kBlockDim, 0, stream>>>(
        A.diag.gpu_data(), M_inv.gpu_data(), n);

    check_cuda(cudaMemcpyAsync(r.gpu_data(), b.data(),
                               n * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync(r=b)");
    sparse::spmv(A, x,
                 DeviceSpan<math::Vec3f>::from(r),
                 -1.0f, 1.0f, cuda_stream);
    collision::apply_contact_spmv(contact, x.data(), r.gpu_data(),
                                  n, -1.0f, cuda_stream);

    apply_jacobi_kernel<<<grid, kBlockDim, 0, stream>>>(
        M_inv.gpu_data(), r.gpu_data(), z.gpu_data(), n);

    check_cuda(cudaMemcpyAsync(p.gpu_data(), z.gpu_data(),
                               n * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync(p=z)");

    check_cuda(cudaMemsetAsync(coeff.gpu_data(), 0, 4 * sizeof(float), stream),
               "cudaMemsetAsync(coeff)");
    dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
        r.gpu_data(), z.gpu_data(), &coeff.gpu_data()[0], n);

    for (int iter = 0; iter < max_iterations; ++iter) {
        sparse::spmv(A,
                     DeviceSpan<math::Vec3f>::from(p),
                     DeviceSpan<math::Vec3f>::from(Ap),
                     1.0f, 0.0f, cuda_stream);
        collision::apply_contact_spmv(contact, p.gpu_data(),
                                      Ap.gpu_data(), n, 1.0f, cuda_stream);

        check_cuda(cudaMemsetAsync(&coeff.gpu_data()[1], 0, sizeof(float),
                                   stream), "cudaMemsetAsync(coeff[1])");
        dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            p.gpu_data(), Ap.gpu_data(), &coeff.gpu_data()[1], n);

        scalar_div_kernel<<<1, 1, 0, stream>>>(
            &coeff.gpu_data()[0], &coeff.gpu_data()[1], &coeff.gpu_data()[3]);

        axpy_dev_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff.gpu_data()[3], p.gpu_data(), x.data(), n);

        naxpy_dev_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff.gpu_data()[3], Ap.gpu_data(), r.gpu_data(), n);

        apply_jacobi_kernel<<<grid, kBlockDim, 0, stream>>>(
            M_inv.gpu_data(), r.gpu_data(), z.gpu_data(), n);

        check_cuda(cudaMemsetAsync(&coeff.gpu_data()[2], 0, sizeof(float),
                                   stream), "cudaMemsetAsync(coeff[2])");
        dot_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            r.gpu_data(), z.gpu_data(), &coeff.gpu_data()[2], n);

        scalar_div_kernel<<<1, 1, 0, stream>>>(
            &coeff.gpu_data()[2], &coeff.gpu_data()[0], &coeff.gpu_data()[3]);

        update_p_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff.gpu_data()[3], z.gpu_data(), p.gpu_data(), n);

        scalar_copy_kernel<<<1, 1, 0, stream>>>(
            &coeff.gpu_data()[2], &coeff.gpu_data()[0]);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// PCGSolver
// ---------------------------------------------------------------------------

void PCGSolver::destroy_graph() noexcept {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_stream_) {
        cudaStreamDestroy(graph_stream_);
        graph_stream_ = nullptr;
    }
    graph_n_ = 0;
    graph_max_iter_ = 0;
}

PCGSolver::~PCGSolver() { destroy_graph(); }

PCGSolver::PCGSolver(PCGSolver&& o) noexcept
    : num_block_rows_(o.num_block_rows_),
      r_(std::move(o.r_)), p_(std::move(o.p_)),
      z_(std::move(o.z_)), Ap_(std::move(o.Ap_)),
      M_inv_(std::move(o.M_inv_)), coeff_(std::move(o.coeff_)),
      graph_stream_(o.graph_stream_), graph_exec_(o.graph_exec_),
      graph_n_(o.graph_n_), graph_max_iter_(o.graph_max_iter_) {
    o.graph_stream_ = nullptr;
    o.graph_exec_ = nullptr;
    o.graph_n_ = 0;
    o.graph_max_iter_ = 0;
}

PCGSolver& PCGSolver::operator=(PCGSolver&& o) noexcept {
    if (this != &o) {
        destroy_graph();
        num_block_rows_ = o.num_block_rows_;
        r_ = std::move(o.r_); p_ = std::move(o.p_);
        z_ = std::move(o.z_); Ap_ = std::move(o.Ap_);
        M_inv_ = std::move(o.M_inv_); coeff_ = std::move(o.coeff_);
        graph_stream_ = o.graph_stream_; graph_exec_ = o.graph_exec_;
        graph_n_ = o.graph_n_; graph_max_iter_ = o.graph_max_iter_;
        o.graph_stream_ = nullptr; o.graph_exec_ = nullptr;
        o.graph_n_ = 0; o.graph_max_iter_ = 0;
    }
    return *this;
}

void PCGSolver::initialize(int num_block_rows) {
    if (num_block_rows < 0) {
        throw std::invalid_argument("PCGSolver::initialize: negative size");
    }
    if (num_block_rows == num_block_rows_) return;

    r_.resize(num_block_rows);
    p_.resize(num_block_rows);
    z_.resize(num_block_rows);
    Ap_.resize(num_block_rows);
    M_inv_.resize(num_block_rows);
    coeff_.resize(4);

    num_block_rows_ = num_block_rows;
}

int PCGSolver::solve(const sparse::BlockCSR3& A,
                     DeviceSpan<math::Vec3f> b,
                     DeviceSpan<math::Vec3f> x,
                     const PCGParams& params,
                     std::uintptr_t cuda_stream,
                     collision::ContactSpMVOp contact) {
    const int n = A.num_block_rows();
    if (n == 0) return 0;

    if (static_cast<int>(b.size()) < n || static_cast<int>(x.size()) < n) {
        throw std::invalid_argument("PCGSolver::solve: b/x shorter than A rows");
    }
    if (static_cast<int>(A.diag.gpu_size()) < n) {
        throw std::invalid_argument(
            "PCGSolver::solve: A.diag has fewer than A.num_block_rows() entries; "
            "call A.build_topology(...) before solving");
    }

    if (n != num_block_rows_) {
        initialize(n);
        destroy_graph();
    }

    CHYSX_NVTX_RANGE_COLOUR("pcg::solve", 0xfff1c40f);

    collision::zero_count_ptr();

    const int max_iter = params.max_iterations;

    // CUDA Graph capture disabled: the cloth simulator reassembles
    // b (RHS) and A.diag every frame, so a captured graph replays
    // stale data.  Emit kernels directly onto the caller's stream.
    emit_pcg(A, b, x, max_iter, cuda_stream, contact,
             r_, p_, z_, Ap_, M_inv_, coeff_);

    return max_iter;
}

float PCGSolver::last_residual() {
    if (coeff_.gpu_size() == 0) return 0.0f;
    coeff_.copy_to_host();
    return coeff_.cpu_data()[0];
}

}  // namespace solver
}  // namespace chysx
