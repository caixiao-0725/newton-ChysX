// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU-side box renderer using CUDA-OpenGL interop.
// Generates triangle + edge VBOs directly on GPU from solver body state,
// avoiding any device-to-host transfer for rendering.

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef TARGET_OS_MAC
#include <OpenGL/GL.h>
#else
#include <GL/gl.h>
#endif

namespace chysx {
namespace avbd {

class GpuSolver;

class GpuRenderer {
public:
    GpuRenderer();
    ~GpuRenderer();

    GpuRenderer(const GpuRenderer&) = delete;
    GpuRenderer& operator=(const GpuRenderer&) = delete;

    /// Call after solver finishes (bodies are still on GPU).
    /// Maps GL VBOs into CUDA, runs a kernel to transform box vertices,
    /// then unmaps so OpenGL can draw them.
    void update(const GpuSolver& solver,
                const float* size_x_dev, const float* size_y_dev,
                const float* size_z_dev,
                const float* mass_dev, int n_bodies);

    /// Draw filled boxes (call between glEnable(GL_DEPTH_TEST) etc.)
    void draw_triangles() const;

    /// Draw wireframe edges
    void draw_edges() const;

    /// Draw projected shadows. `shadow_mat` is a column-major 4x4 matrix
    /// already computed on the CPU (from findShadowPlane).
    void update_shadows(const GpuSolver& solver,
                        const float* size_x_dev, const float* size_y_dev,
                        const float* size_z_dev,
                        const float* mass_dev, int n_bodies,
                        const float shadow_mat[16]);
    void draw_shadow_triangles() const;

    int n_bodies() const { return n_bodies_; }

private:
    void ensure_capacity(int n_bodies);

    int n_bodies_ = 0;
    int capacity_ = 0;

    // GL buffer objects
    GLuint tri_vbo_ = 0;       // 12 tris * 3 verts * 3 floats per body
    GLuint tri_norm_vbo_ = 0;  // per-face normals
    GLuint edge_vbo_ = 0;      // 12 edges * 2 verts * 3 floats per body
    GLuint shadow_vbo_ = 0;    // same as tri_vbo_ but shadow-projected

    // CUDA graphics resources (registered from GL VBOs)
    struct cudaGraphicsResource* tri_res_ = nullptr;
    struct cudaGraphicsResource* tri_norm_res_ = nullptr;
    struct cudaGraphicsResource* edge_res_ = nullptr;
    struct cudaGraphicsResource* shadow_res_ = nullptr;

    int n_tri_verts_ = 0;
    int n_edge_verts_ = 0;
    int n_shadow_verts_ = 0;
};

}  // namespace avbd
}  // namespace chysx
