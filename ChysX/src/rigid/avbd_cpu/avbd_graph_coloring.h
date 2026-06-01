// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU parallel graph coloring for AVBD rigid body Gauss-Seidel.
// Implements four algorithms from the Vivace paper (SIGGRAPH Asia 2016):
//   1. Brooks-Vizing randomized coloring (Vivace)
//   2. Luby MIS coloring
//   3. Jones-Plassmann (JP) coloring
//   4. Largest-Degree-First (LDF) coloring

#pragma once

#include "avbd_narrowphase_gpu.h"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace avbd {

struct ColoringStats {
    int num_colors;
    int num_rounds;
    float elapsed_ms;
};

class GraphColoringGPU {
public:
    GraphColoringGPU() = default;
    ~GraphColoringGPU() = default;

    GraphColoringGPU(const GraphColoringGPU&) = delete;
    GraphColoringGPU& operator=(const GraphColoringGPU&) = delete;

    /// Run Brooks-Vizing randomized coloring (Vivace, Grable & Panconesi 2000).
    /// Fastest, moderate number of colors.
    ColoringStats color_vivace(
        const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
        int n_bodies, int stride, unsigned seed = 0);

    /// Run Luby's MIS-based coloring (Luby 1985).
    /// Most colors, moderate speed.
    ColoringStats color_luby(
        const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
        int n_bodies, int stride, unsigned seed = 0);

    /// Run Jones-Plassmann coloring (Jones & Plassmann 1993).
    /// Fewer colors than Luby, moderate speed.
    ColoringStats color_jp(
        const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
        int n_bodies, int stride, unsigned seed = 0);

    /// Run Largest-Degree-First coloring (Welsh & Powell 1967).
    /// Fewest colors, slowest.
    ColoringStats color_ldf(
        const int* vtx_counts_dev, const VertexEntry* vtx_table_dev,
        int n_bodies, int stride);

    const int* colors_cpu() const { return colors_.cpu_data(); }
    const int* colors_gpu() const { return colors_.gpu_data(); }
    int num_bodies() const { return n_bodies_; }

private:
    void ensure_capacity(int n_bodies);

    int n_bodies_ = 0;

    CudaArray<int> colors_;           // [n_bodies] per-body color, -1 = uncolored
    CudaArray<int> temp_colors_;      // tentative color per vertex (Vivace)
    CudaArray<unsigned> rand_state_;  // per-vertex RNG state
    CudaArray<int> remaining_;        // single int: count of uncolored vertices
    CudaArray<int> palette_;          // [n_bodies * palette_width] bitmask palette (Vivace)
    CudaArray<int> weights_;          // [n_bodies] random weight (Luby/JP)
    CudaArray<int> active_;           // [n_bodies] 1=uncolored, 0=colored
    CudaArray<int> degrees_;          // [n_bodies] residual degree (LDF)

    int palette_width_ = 0;          // number of ints per vertex for bitmask palette
};

}  // namespace avbd
}  // namespace chysx
