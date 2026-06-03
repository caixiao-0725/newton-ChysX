// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU parallel graph coloring: four algorithms from the Vivace paper.

#include "avbd_graph_coloring.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace chysx {
namespace avbd {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GraphColoringGPU: ") + w +
                                 ": " + cudaGetErrorString(e));
}

// Simple LCG-based per-thread RNG
__device__ unsigned rng_next(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// ---------------------------------------------------------------------------
//  Helper: max-reduce colors[] to get num_colors = max(colors[i]) + 1
// ---------------------------------------------------------------------------

__global__ void max_color_kernel(const int* colors, int n, int* out_max) {
    __shared__ int s_max;
    if (threadIdx.x == 0) s_max = -1;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMax(&s_max, colors[i]);
    __syncthreads();

    if (threadIdx.x == 0) atomicMax(out_max, s_max);
}

// ---------------------------------------------------------------------------
//  Helper: count uncolored vertices
// ---------------------------------------------------------------------------

__global__ void count_uncolored_kernel(const int* colors, int n, int* out_count) {
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && colors[i] < 0)
        atomicAdd(&s_count, 1);
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(out_count, s_count);
}

// ---------------------------------------------------------------------------
//  1. Brooks-Vizing (Vivace) coloring kernels
// ---------------------------------------------------------------------------

// Initialize palette as bitmask: each vertex gets colors {0..deg/s}
// palette is stored as an array of ints per vertex (1 bit per color).
// palette_width = number of ints needed to cover max_palette_size colors.
__global__ void vivace_init_palette_kernel(
    const int* vtx_counts, int n, int shrink_factor,
    int* palette, int palette_width,
    int* colors, unsigned* rand_state, unsigned seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    colors[i] = -1;
    rand_state[i] = seed ^ (i * 2654435761u + 1);

    int deg = vtx_counts[i];
    int palette_size = max(1, deg / max(1, shrink_factor)) + 1;

    for (int w = 0; w < palette_width; w++) {
        int bits = 0;
        for (int b = 0; b < 32; b++) {
            int color_id = w * 32 + b;
            if (color_id < palette_size)
                bits |= (1 << b);
        }
        palette[i * palette_width + w] = bits;
    }
}

// Step 1: Tentative coloring — each uncolored vertex picks a random color from palette
__global__ void vivace_tentative_kernel(
    int* colors, int* temp_colors, unsigned* rand_state,
    const int* palette, int palette_width, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) { if (i < n) temp_colors[i] = -1; return; }

    // Count available colors
    int avail = 0;
    for (int w = 0; w < palette_width; w++)
        avail += __popc(palette[i * palette_width + w]);

    if (avail == 0) { temp_colors[i] = -1; return; }

    // Pick random k-th available color
    unsigned r = rng_next(rand_state[i]);
    int pick = r % avail;
    int count = 0;
    for (int w = 0; w < palette_width; w++) {
        int bits = palette[i * palette_width + w];
        while (bits) {
            int b = __ffs(bits) - 1;
            if (count == pick) {
                temp_colors[i] = w * 32 + b;
                return;
            }
            count++;
            bits &= bits - 1;
        }
    }
    temp_colors[i] = -1;
}

// Step 2: Conflict resolution with Hungarian heuristic
__global__ void vivace_conflict_kernel(
    int* colors, const int* temp_colors,
    const int* vtx_counts, const VertexEntry* vtx_table, int stride,
    int* palette, int palette_width, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    int my_color = temp_colors[i];
    if (my_color < 0) return;

    int deg = min(vtx_counts[i], stride);
    bool conflict = false;
    for (int s = 0; s < deg; s++) {
        int nb = vtx_table[i * stride + s].other_body;
        if (nb < 0 || nb >= n) continue;
        if (temp_colors[nb] == my_color) {
            // Hungarian heuristic: higher index wins
            if (i < nb) { conflict = true; break; }
        }
    }

    if (!conflict) {
        colors[i] = my_color;
        // Remove this color from all neighbors' palettes
        int word = my_color / 32;
        int bit = my_color % 32;
        for (int s = 0; s < deg; s++) {
            int nb = vtx_table[i * stride + s].other_body;
            if (nb >= 0 && nb < n && colors[nb] < 0) {
                atomicAnd(&palette[nb * palette_width + word], ~(1 << bit));
            }
        }
    }
}

// Step 3a: Mark hungry vertices (palette empty) into a flag array
__global__ void vivace_mark_hungry_kernel(
    const int* colors, const int* palette, int palette_width,
    int* hungry_flag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) { if (i < n) hungry_flag[i] = 0; return; }

    int avail = 0;
    for (int w = 0; w < palette_width; w++)
        avail += __popc(palette[i * palette_width + w]);

    hungry_flag[i] = (avail == 0) ? 1 : 0;
}

// Step 3b: Feed the hungry — all hungry vertices get the SAME new color
__global__ void vivace_feed_kernel(
    const int* colors, const int* hungry_flag,
    int* palette, int palette_width,
    int new_color, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    if (hungry_flag[i]) {
        int word = new_color / 32;
        int bit = new_color % 32;
        if (word < palette_width)
            palette[i * palette_width + word] |= (1 << bit);
    }
}

// ---------------------------------------------------------------------------
//  2. Luby MIS coloring kernels
// ---------------------------------------------------------------------------

__global__ void luby_assign_weights_kernel(
    const int* colors, unsigned* rand_state, int* weights, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (colors[i] >= 0) { weights[i] = -1; return; }
    weights[i] = (int)(rng_next(rand_state[i]) & 0x7FFFFFFF);
}

__global__ void luby_find_mis_and_color_kernel(
    int* colors, const int* weights,
    const int* vtx_counts, const VertexEntry* vtx_table, int stride,
    int current_color, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    int my_w = weights[i];
    int deg = min(vtx_counts[i], stride);
    bool is_local_max = true;
    for (int s = 0; s < deg; s++) {
        int nb = vtx_table[i * stride + s].other_body;
        if (nb < 0 || nb >= n || colors[nb] >= 0) continue;
        if (weights[nb] > my_w || (weights[nb] == my_w && nb > i)) {
            is_local_max = false;
            break;
        }
    }
    if (is_local_max)
        colors[i] = current_color;
}

// ---------------------------------------------------------------------------
//  3. Jones-Plassmann coloring kernels
// ---------------------------------------------------------------------------

__global__ void jp_color_local_max_kernel(
    int* colors, const int* weights,
    const int* vtx_counts, const VertexEntry* vtx_table, int stride, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    int my_w = weights[i];
    int deg = min(vtx_counts[i], stride);
    bool is_local_max = true;
    for (int s = 0; s < deg; s++) {
        int nb = vtx_table[i * stride + s].other_body;
        if (nb < 0 || nb >= n || colors[nb] >= 0) continue;
        if (weights[nb] > my_w || (weights[nb] == my_w && nb > i)) {
            is_local_max = false;
            break;
        }
    }

    if (is_local_max) {
        // Assign smallest color not used by any already-colored neighbor
        unsigned used_mask = 0;
        for (int s = 0; s < deg; s++) {
            int nb = vtx_table[i * stride + s].other_body;
            if (nb < 0 || nb >= n) continue;
            int nc = colors[nb];
            if (nc >= 0 && nc < 32) used_mask |= (1u << nc);
        }
        int c = __ffs(~used_mask) - 1;
        colors[i] = c;
    }
}

// ---------------------------------------------------------------------------
//  4. LDF (Largest-Degree-First) coloring kernels
// ---------------------------------------------------------------------------

__global__ void ldf_init_degrees_kernel(
    const int* vtx_counts, int stride, int* degrees, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    degrees[i] = min(vtx_counts[i], stride);
}

__global__ void ldf_find_max_degree_and_color_kernel(
    int* colors, const int* degrees,
    const int* vtx_counts, const VertexEntry* vtx_table, int stride, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    int my_deg = degrees[i];
    int deg = min(vtx_counts[i], stride);
    bool is_local_max = true;
    for (int s = 0; s < deg; s++) {
        int nb = vtx_table[i * stride + s].other_body;
        if (nb < 0 || nb >= n || colors[nb] >= 0) continue;
        int nb_deg = degrees[nb];
        if (nb_deg > my_deg || (nb_deg == my_deg && nb > i)) {
            is_local_max = false;
            break;
        }
    }

    if (is_local_max) {
        unsigned used_mask = 0;
        for (int s = 0; s < deg; s++) {
            int nb = vtx_table[i * stride + s].other_body;
            if (nb < 0 || nb >= n) continue;
            int nc = colors[nb];
            if (nc >= 0 && nc < 32) used_mask |= (1u << nc);
        }
        colors[i] = __ffs(~used_mask) - 1;
    }
}

// Update residual degrees after removing colored vertices
__global__ void ldf_update_degrees_kernel(
    const int* colors, int* degrees,
    const int* vtx_counts, const VertexEntry* vtx_table, int stride, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || colors[i] >= 0) return;

    int d = 0;
    int deg = min(vtx_counts[i], stride);
    for (int s = 0; s < deg; s++) {
        int nb = vtx_table[i * stride + s].other_body;
        if (nb >= 0 && nb < n && colors[nb] < 0)
            d++;
    }
    degrees[i] = d;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Host implementation
// ---------------------------------------------------------------------------

void GraphColoringGPU::ensure_capacity(int n_bodies) {
    if (n_bodies <= n_bodies_) return;
    n_bodies_ = n_bodies;
    colors_.resize(n_bodies);
    temp_colors_.resize(n_bodies);
    rand_state_.resize(n_bodies);
    remaining_.resize(1);
    weights_.resize(n_bodies);
    active_.resize(n_bodies);
    degrees_.resize(n_bodies);

    int max_colors = 64;
    palette_width_ = (max_colors + 31) / 32;
    palette_.resize(n_bodies * palette_width_);
}

// ---------------------------------------------------------------------------
//  1. Brooks-Vizing (Vivace)
// ---------------------------------------------------------------------------

ColoringStats GraphColoringGPU::color_vivace(
    const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
    int n_bodies, int stride, unsigned seed)
{
    ensure_capacity(n_bodies);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // Download vtx_counts to compute shrink factor s = min non-zero degree
    std::vector<int> host_degrees(n_bodies);
    check(cudaMemcpy(host_degrees.data(), vtx_counts_dev,
                     n_bodies * sizeof(int), cudaMemcpyDeviceToHost),
          "vivace download degrees");

    int min_deg = stride;
    int max_deg = 0;
    for (int i = 0; i < n_bodies; i++) {
        int d = host_degrees[i] < stride ? host_degrees[i] : stride;
        if (d > 0 && d < min_deg) min_deg = d;
        if (d > max_deg) max_deg = d;
    }
    int shrink_factor = min_deg > 1 ? min_deg : 1;

    // Compute initial max palette size across all vertices
    int init_max_palette = 0;
    for (int i = 0; i < n_bodies; i++) {
        int d = host_degrees[i] < stride ? host_degrees[i] : stride;
        int sf = d / shrink_factor;
        int ps = (sf > 1 ? sf : 1) + 1;
        if (ps > init_max_palette) init_max_palette = ps;
    }

    // Track next available color for feed-the-hungry
    int next_new_color = init_max_palette;

    vivace_init_palette_kernel<<<grid(n_bodies), kBlock>>>(
        vtx_counts_dev, n_bodies, shrink_factor,
        palette_.gpu_data(), palette_width_,
        colors_.gpu_data(), rand_state_.gpu_data(), seed);
    check(cudaGetLastError(), "vivace_init_palette");

    // Reuse temp_colors_ as hungry_flag (same size, both int arrays)
    int* hungry_flag = temp_colors_.gpu_data();

    int rounds = 0;
    for (;;) {
        check(cudaMemset(remaining_.gpu_data(), 0, sizeof(int)), "zero remaining");
        count_uncolored_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), n_bodies, remaining_.gpu_data());
        int uncolored = 0;
        check(cudaMemcpy(&uncolored, remaining_.gpu_data(), sizeof(int),
                         cudaMemcpyDeviceToHost), "remaining D2H");
        if (uncolored == 0) break;

        // Step 1: Tentative coloring
        vivace_tentative_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), temp_colors_.gpu_data(), rand_state_.gpu_data(),
            palette_.gpu_data(), palette_width_, n_bodies);

        // Step 2: Conflict resolution
        vivace_conflict_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), temp_colors_.gpu_data(),
            vtx_counts_dev, vtx_table_dev, stride,
            palette_.gpu_data(), palette_width_, n_bodies);

        // Step 3: Feed the hungry (two-phase)
        // 3a: mark hungry vertices
        vivace_mark_hungry_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), palette_.gpu_data(), palette_width_,
            hungry_flag, n_bodies);

        // 3b: check if any hungry exist, then feed with ONE shared new color
        check(cudaMemset(remaining_.gpu_data(), 0, sizeof(int)), "zero hungry count");
        count_uncolored_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), n_bodies, remaining_.gpu_data());
        // Quick check: scan hungry_flag for any 1
        // (We use a simple approach: download and check)
        temp_colors_.copy_to_host();
        bool any_hungry = false;
        for (int i = 0; i < n_bodies; i++) {
            if (temp_colors_.cpu_data()[i] == 1) { any_hungry = true; break; }
        }

        if (any_hungry && next_new_color < palette_width_ * 32) {
            vivace_feed_kernel<<<grid(n_bodies), kBlock>>>(
                colors_.gpu_data(), hungry_flag,
                palette_.gpu_data(), palette_width_,
                next_new_color, n_bodies);
            next_new_color++;
        }

        rounds++;
        if (rounds > 200) break;
    }

    colors_.copy_to_host();

    int num_colors = 0;
    for (int i = 0; i < n_bodies; i++)
        num_colors = max(num_colors, colors_.cpu_data()[i] + 1);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return {num_colors, rounds, ms};
}

// ---------------------------------------------------------------------------
//  2. Luby MIS
// ---------------------------------------------------------------------------

ColoringStats GraphColoringGPU::color_luby(
    const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
    int n_bodies, int stride, unsigned seed)
{
    ensure_capacity(n_bodies);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    check(cudaMemset(colors_.gpu_data(), 0xFF, n_bodies * sizeof(int)), "init colors -1");

    // Init RNG
    auto init_rng = [&]() {
        for (int i = 0; i < n_bodies; i++)
            rand_state_[i] = seed ^ (i * 2654435761u + 1);
        rand_state_.copy_to_device();
    };
    init_rng();

    int current_color = 0;
    int rounds = 0;
    for (;;) {
        check(cudaMemset(remaining_.gpu_data(), 0, sizeof(int)), "zero remaining");
        count_uncolored_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), n_bodies, remaining_.gpu_data());
        int uncolored = 0;
        check(cudaMemcpy(&uncolored, remaining_.gpu_data(), sizeof(int),
                         cudaMemcpyDeviceToHost), "remaining D2H");
        if (uncolored == 0) break;

        luby_assign_weights_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), rand_state_.gpu_data(), weights_.gpu_data(), n_bodies);

        luby_find_mis_and_color_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), weights_.gpu_data(),
            vtx_counts_dev, vtx_table_dev, stride,
            current_color, n_bodies);

        current_color++;
        rounds++;
        if (rounds > 200) break;
    }

    colors_.copy_to_host();
    int num_colors = current_color;

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return {num_colors, rounds, ms};
}

// ---------------------------------------------------------------------------
//  3. Jones-Plassmann
// ---------------------------------------------------------------------------

ColoringStats GraphColoringGPU::color_jp(
    const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
    int n_bodies, int stride, unsigned seed)
{
    ensure_capacity(n_bodies);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    check(cudaMemset(colors_.gpu_data(), 0xFF, n_bodies * sizeof(int)), "init colors -1");

    for (int i = 0; i < n_bodies; i++)
        rand_state_[i] = seed ^ (i * 2654435761u + 1);
    rand_state_.copy_to_device();

    int rounds = 0;
    for (;;) {
        check(cudaMemset(remaining_.gpu_data(), 0, sizeof(int)), "zero remaining");
        count_uncolored_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), n_bodies, remaining_.gpu_data());
        int uncolored = 0;
        check(cudaMemcpy(&uncolored, remaining_.gpu_data(), sizeof(int),
                         cudaMemcpyDeviceToHost), "remaining D2H");
        if (uncolored == 0) break;

        luby_assign_weights_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), rand_state_.gpu_data(), weights_.gpu_data(), n_bodies);

        jp_color_local_max_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), weights_.gpu_data(),
            vtx_counts_dev, vtx_table_dev, stride, n_bodies);

        rounds++;
        if (rounds > 200) break;
    }

    // GPU max-reduction to compute num_colors without downloading colors_
    int max_color = -1;
    check(cudaMemset(remaining_.gpu_data(), 0xFF, sizeof(int)), "init max_color -1");
    max_color_kernel<<<grid(n_bodies), kBlock>>>(
        colors_.gpu_data(), n_bodies, remaining_.gpu_data());
    check(cudaMemcpy(&max_color, remaining_.gpu_data(), sizeof(int),
                     cudaMemcpyDeviceToHost), "max_color D2H");
    int num_colors = max_color + 1;

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return {num_colors, rounds, ms};
}

// ---------------------------------------------------------------------------
//  4. LDF (Largest-Degree-First)
// ---------------------------------------------------------------------------

ColoringStats GraphColoringGPU::color_ldf(
    const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
    int n_bodies, int stride)
{
    ensure_capacity(n_bodies);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    check(cudaMemset(colors_.gpu_data(), 0xFF, n_bodies * sizeof(int)), "init colors -1");

    ldf_init_degrees_kernel<<<grid(n_bodies), kBlock>>>(
        vtx_counts_dev, stride, degrees_.gpu_data(), n_bodies);

    int rounds = 0;
    for (;;) {
        check(cudaMemset(remaining_.gpu_data(), 0, sizeof(int)), "zero remaining");
        count_uncolored_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), n_bodies, remaining_.gpu_data());
        int uncolored = 0;
        check(cudaMemcpy(&uncolored, remaining_.gpu_data(), sizeof(int),
                         cudaMemcpyDeviceToHost), "remaining D2H");
        if (uncolored == 0) break;

        ldf_find_max_degree_and_color_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), degrees_.gpu_data(),
            vtx_counts_dev, vtx_table_dev, stride, n_bodies);

        ldf_update_degrees_kernel<<<grid(n_bodies), kBlock>>>(
            colors_.gpu_data(), degrees_.gpu_data(),
            vtx_counts_dev, vtx_table_dev, stride, n_bodies);

        rounds++;
        if (rounds > 200) break;
    }

    colors_.copy_to_host();
    int num_colors = 0;
    for (int i = 0; i < n_bodies; i++)
        num_colors = max(num_colors, colors_.cpu_data()[i] + 1);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return {num_colors, rounds, ms};
}

}  // namespace avbd
}  // namespace chysx
