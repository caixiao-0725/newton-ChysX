// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// CUDA kernels that transform box vertices directly into GL VBOs
// via CUDA-OpenGL interop, eliminating the device→host→GL path.

#include "avbd_gpu_renderer.h"
#include "avbd_gpu_solver.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include <string>

// ---- GL extension loading (VBO functions not in gl.h on Windows) ----

#ifdef _WIN32
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei, GLuint*);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint*);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum, ptrdiff_t, const void*, GLenum);

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif

static PFNGLGENBUFFERSPROC    pglGenBuffers = nullptr;
static PFNGLDELETEBUFFERSPROC pglDeleteBuffers = nullptr;
static PFNGLBINDBUFFERPROC    pglBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC    pglBufferData = nullptr;

static void load_gl_procs() {
    if (pglGenBuffers) return;
    pglGenBuffers    = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    pglDeleteBuffers = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
    pglBindBuffer    = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    pglBufferData    = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
}

#define glGenBuffers    pglGenBuffers
#define glDeleteBuffers pglDeleteBuffers
#define glBindBuffer    pglBindBuffer
#define glBufferData    pglBufferData

#else
static void load_gl_procs() {}
#endif

namespace chysx {
namespace avbd {

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

inline void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GpuRenderer: ") + w +
                                 ": " + cudaGetErrorString(e));
}

// Unit box vertices (matches avbd_scenes.cpp kBoxV)
__constant__ float c_box_v[8][3] = {
    {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f},
    {+0.5f, +0.5f, -0.5f}, {-0.5f, +0.5f, -0.5f},
    {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f},
    {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f}};

__constant__ int c_box_tri[12][3] = {
    {0,1,2},{0,2,3}, {4,6,5},{4,7,6},
    {1,5,6},{1,6,2}, {4,0,3},{4,3,7},
    {3,2,6},{3,6,7}, {4,5,1},{4,1,0}};

__constant__ int c_box_edge[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

// ---- Device helpers ----

struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };

__device__ V3 v3add(V3 a, V3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ V3 v3sub(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ V3 v3scale(V3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ V3 v3cross(V3 a, V3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ float v3dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ V3 qrot(Q4 q, V3 v) {
    V3 u = {q.x, q.y, q.z};
    V3 t = v3scale(v3cross(u, v), 2.0f);
    return v3add(v3add(v, v3scale(t, q.w)), v3cross(u, t));
}

// half_x/y/z are the half-extents; c_box_v has ±0.5 so we scale by 2*half = size
__device__ V3 body_vertex(float px, float py, float pz,
                          float qx, float qy, float qz, float qw,
                          float hx, float hy, float hz,
                          const float local[3]) {
    V3 scaled = {local[0] * hx * 2.0f, local[1] * hy * 2.0f, local[2] * hz * 2.0f};
    Q4 q = {qx, qy, qz, qw};
    V3 rotated = qrot(q, scaled);
    return {rotated.x + px, rotated.y + py, rotated.z + pz};
}

// Generate triangle vertices + normals. One thread per body.
// hx/hy/hz are half-extents (size * 0.5).
__global__ void gen_tri_verts_kernel_v2(
    float* __restrict__ out_pos, float* __restrict__ out_norm,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy,
    const float* __restrict__ qz, const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    int n)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n) return;

    float bpx = px[body], bpy = py[body], bpz = pz[body];
    float bqx = qx[body], bqy = qy[body], bqz = qz[body], bqw = qw[body];
    float bhx = hx[body], bhy = hy[body], bhz = hz[body];

    int base = body * 36 * 3;

    for (int t = 0; t < 12; t++) {
        int i0 = c_box_tri[t][0], i1 = c_box_tri[t][1], i2 = c_box_tri[t][2];
        V3 a = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i0]);
        V3 b = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i1]);
        V3 c = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i2]);

        V3 ab = v3sub(b, a);
        V3 ac = v3sub(c, a);
        V3 nv = v3cross(ab, ac);
        float len = sqrtf(v3dot(nv, nv));
        if (len > 1e-8f) nv = v3scale(nv, 1.0f / len);

        int off = base + t * 9;
        out_pos[off+0] = a.x; out_pos[off+1] = a.y; out_pos[off+2] = a.z;
        out_pos[off+3] = b.x; out_pos[off+4] = b.y; out_pos[off+5] = b.z;
        out_pos[off+6] = c.x; out_pos[off+7] = c.y; out_pos[off+8] = c.z;

        out_norm[off+0] = nv.x; out_norm[off+1] = nv.y; out_norm[off+2] = nv.z;
        out_norm[off+3] = nv.x; out_norm[off+4] = nv.y; out_norm[off+5] = nv.z;
        out_norm[off+6] = nv.x; out_norm[off+7] = nv.y; out_norm[off+8] = nv.z;
    }
}

// One thread per body → 12 edges × 2 verts = 24 verts per body
__global__ void gen_edge_verts_kernel(
    float* __restrict__ out,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy,
    const float* __restrict__ qz, const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    int n)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n) return;

    float bpx = px[body], bpy = py[body], bpz = pz[body];
    float bqx = qx[body], bqy = qy[body], bqz = qz[body], bqw = qw[body];
    float bhx = hx[body], bhy = hy[body], bhz = hz[body];

    int base = body * 24 * 3;

    for (int e = 0; e < 12; e++) {
        int i0 = c_box_edge[e][0], i1 = c_box_edge[e][1];
        V3 a = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i0]);
        V3 b = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i1]);

        int off = base + e * 6;
        out[off+0] = a.x; out[off+1] = a.y; out[off+2] = a.z;
        out[off+3] = b.x; out[off+4] = b.y; out[off+5] = b.z;
    }
}

// Shadow-projected triangle vertices (dynamic bodies only, mass > 0)
__global__ void gen_shadow_verts_kernel(
    float* __restrict__ out,
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz,
    const float* __restrict__ qx, const float* __restrict__ qy,
    const float* __restrict__ qz, const float* __restrict__ qw,
    const float* __restrict__ hx, const float* __restrict__ hy, const float* __restrict__ hz,
    const float* __restrict__ mass_dev,
    const float sm00, const float sm01, const float sm02, const float sm03,
    const float sm10, const float sm11, const float sm12, const float sm13,
    const float sm20, const float sm21, const float sm22, const float sm23,
    const float sm30, const float sm31, const float sm32, const float sm33,
    int n)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n) return;
    if (mass_dev[body] <= 0.0f) return;  // static bodies don't cast shadows

    float bpx = px[body], bpy = py[body], bpz = pz[body];
    float bqx = qx[body], bqy = qy[body], bqz = qz[body], bqw = qw[body];
    float bhx = hx[body], bhy = hy[body], bhz = hz[body];

    int base = body * 36 * 3;

    for (int t = 0; t < 12; t++) {
        int i0 = c_box_tri[t][0], i1 = c_box_tri[t][1], i2 = c_box_tri[t][2];
        V3 verts[3];
        verts[0] = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i0]);
        verts[1] = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i1]);
        verts[2] = body_vertex(bpx, bpy, bpz, bqx, bqy, bqz, bqw, bhx, bhy, bhz, c_box_v[i2]);

        int off = base + t * 9;
        for (int v = 0; v < 3; v++) {
            float x = verts[v].x, y = verts[v].y, z = verts[v].z;
            float ox = sm00*x + sm01*y + sm02*z + sm03;
            float oy = sm10*x + sm11*y + sm12*z + sm13;
            float oz = sm20*x + sm21*y + sm22*z + sm23;
            float ow = sm30*x + sm31*y + sm32*z + sm33;
            if (fabsf(ow) > 1e-6f) { ox /= ow; oy /= ow; oz /= ow; }
            out[off + v*3 + 0] = ox;
            out[off + v*3 + 1] = oy;
            out[off + v*3 + 2] = oz;
        }
    }
}

}  // anonymous namespace

// ---- Host implementation ----

GpuRenderer::GpuRenderer() = default;

GpuRenderer::~GpuRenderer() {
    if (tri_res_) cudaGraphicsUnregisterResource(tri_res_);
    if (tri_norm_res_) cudaGraphicsUnregisterResource(tri_norm_res_);
    if (edge_res_) cudaGraphicsUnregisterResource(edge_res_);
    if (shadow_res_) cudaGraphicsUnregisterResource(shadow_res_);
    load_gl_procs();
    if (tri_vbo_) glDeleteBuffers(1, &tri_vbo_);
    if (tri_norm_vbo_) glDeleteBuffers(1, &tri_norm_vbo_);
    if (edge_vbo_) glDeleteBuffers(1, &edge_vbo_);
    if (shadow_vbo_) glDeleteBuffers(1, &shadow_vbo_);
}

void GpuRenderer::ensure_capacity(int n) {
    if (n <= capacity_) return;

    load_gl_procs();

    // Unregister old CUDA resources
    if (tri_res_) { cudaGraphicsUnregisterResource(tri_res_); tri_res_ = nullptr; }
    if (tri_norm_res_) { cudaGraphicsUnregisterResource(tri_norm_res_); tri_norm_res_ = nullptr; }
    if (edge_res_) { cudaGraphicsUnregisterResource(edge_res_); edge_res_ = nullptr; }
    if (shadow_res_) { cudaGraphicsUnregisterResource(shadow_res_); shadow_res_ = nullptr; }

    // Delete old VBOs
    if (tri_vbo_) { glDeleteBuffers(1, &tri_vbo_); tri_vbo_ = 0; }
    if (tri_norm_vbo_) { glDeleteBuffers(1, &tri_norm_vbo_); tri_norm_vbo_ = 0; }
    if (edge_vbo_) { glDeleteBuffers(1, &edge_vbo_); edge_vbo_ = 0; }
    if (shadow_vbo_) { glDeleteBuffers(1, &shadow_vbo_); shadow_vbo_ = 0; }

    // Grow with headroom
    capacity_ = n + n / 4 + 64;
    int tri_bytes = capacity_ * 36 * 3 * sizeof(float);
    int edge_bytes = capacity_ * 24 * 3 * sizeof(float);

    glGenBuffers(1, &tri_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, tri_vbo_);
    glBufferData(GL_ARRAY_BUFFER, tri_bytes, nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &tri_norm_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, tri_norm_vbo_);
    glBufferData(GL_ARRAY_BUFFER, tri_bytes, nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &edge_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, edge_vbo_);
    glBufferData(GL_ARRAY_BUFFER, edge_bytes, nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &shadow_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, shadow_vbo_);
    glBufferData(GL_ARRAY_BUFFER, tri_bytes, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register with CUDA
    check(cudaGraphicsGLRegisterBuffer(&tri_res_, tri_vbo_,
          cudaGraphicsMapFlagsWriteDiscard), "register tri_vbo");
    check(cudaGraphicsGLRegisterBuffer(&tri_norm_res_, tri_norm_vbo_,
          cudaGraphicsMapFlagsWriteDiscard), "register tri_norm_vbo");
    check(cudaGraphicsGLRegisterBuffer(&edge_res_, edge_vbo_,
          cudaGraphicsMapFlagsWriteDiscard), "register edge_vbo");
    check(cudaGraphicsGLRegisterBuffer(&shadow_res_, shadow_vbo_,
          cudaGraphicsMapFlagsWriteDiscard), "register shadow_vbo");
}

void GpuRenderer::update(const GpuSolver& solver,
                         const float* size_x_dev, const float* size_y_dev,
                         const float* size_z_dev,
                         const float* mass_dev, int n_bodies) {
    if (n_bodies <= 0) return;
    ensure_capacity(n_bodies);
    n_bodies_ = n_bodies;
    n_tri_verts_ = n_bodies * 36;
    n_edge_verts_ = n_bodies * 24;

    // Map VBOs for CUDA
    cudaGraphicsResource* resources[] = {tri_res_, tri_norm_res_, edge_res_};
    check(cudaGraphicsMapResources(3, resources, 0), "map resources");

    float* tri_ptr = nullptr;
    float* norm_ptr = nullptr;
    float* edge_ptr = nullptr;
    size_t sz;
    check(cudaGraphicsResourceGetMappedPointer((void**)&tri_ptr, &sz, tri_res_), "get tri ptr");
    check(cudaGraphicsResourceGetMappedPointer((void**)&norm_ptr, &sz, tri_norm_res_), "get norm ptr");
    check(cudaGraphicsResourceGetMappedPointer((void**)&edge_ptr, &sz, edge_res_), "get edge ptr");

    // Access solver GPU arrays directly (const-cast needed since solver is const)
    auto& s = const_cast<GpuSolver&>(solver);

    gen_tri_verts_kernel_v2<<<grid(n_bodies), kBlock>>>(
        tri_ptr, norm_ptr,
        s.pos_x_dev(), s.pos_y_dev(), s.pos_z_dev(),
        s.quat_x_dev(), s.quat_y_dev(), s.quat_z_dev(), s.quat_w_dev(),
        size_x_dev, size_y_dev, size_z_dev,
        n_bodies);

    gen_edge_verts_kernel<<<grid(n_bodies), kBlock>>>(
        edge_ptr,
        s.pos_x_dev(), s.pos_y_dev(), s.pos_z_dev(),
        s.quat_x_dev(), s.quat_y_dev(), s.quat_z_dev(), s.quat_w_dev(),
        size_x_dev, size_y_dev, size_z_dev,
        n_bodies);

    check(cudaGraphicsUnmapResources(3, resources, 0), "unmap resources");
}

void GpuRenderer::draw_triangles() const {
    if (n_tri_verts_ <= 0) return;

    load_gl_procs();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, tri_vbo_);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, tri_norm_vbo_);
    glNormalPointer(GL_FLOAT, 0, nullptr);

    glDrawArrays(GL_TRIANGLES, 0, n_tri_verts_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void GpuRenderer::draw_edges() const {
    if (n_edge_verts_ <= 0) return;

    load_gl_procs();

    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, edge_vbo_);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);

    glDrawArrays(GL_LINES, 0, n_edge_verts_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void GpuRenderer::update_shadows(const GpuSolver& solver,
                                 const float* size_x_dev, const float* size_y_dev,
                                 const float* size_z_dev,
                                 const float* mass_dev, int n_bodies,
                                 const float sm[16]) {
    if (n_bodies <= 0) return;
    ensure_capacity(n_bodies);
    n_shadow_verts_ = n_bodies * 36;

    check(cudaGraphicsMapResources(1, &shadow_res_, 0), "map shadow");

    float* shadow_ptr = nullptr;
    size_t sz;
    check(cudaGraphicsResourceGetMappedPointer((void**)&shadow_ptr, &sz, shadow_res_), "get shadow ptr");

    // Zero the buffer first (static bodies won't write)
    check(cudaMemsetAsync(shadow_ptr, 0, n_bodies * 36 * 3 * sizeof(float), 0), "memset shadow");

    auto& s = const_cast<GpuSolver&>(solver);

    // Shadow matrix is column-major in OpenGL convention.
    // Our kernel expects row-major access: sm[row*4 + col].
    // OpenGL column-major: sm[col*4 + row], so sm[0]=col0row0, sm[1]=col0row1, etc.
    // We pass individual elements in row-major order.
    gen_shadow_verts_kernel<<<grid(n_bodies), kBlock>>>(
        shadow_ptr,
        s.pos_x_dev(), s.pos_y_dev(), s.pos_z_dev(),
        s.quat_x_dev(), s.quat_y_dev(), s.quat_z_dev(), s.quat_w_dev(),
        size_x_dev, size_y_dev, size_z_dev,
        mass_dev,
        sm[0], sm[4], sm[8],  sm[12],
        sm[1], sm[5], sm[9],  sm[13],
        sm[2], sm[6], sm[10], sm[14],
        sm[3], sm[7], sm[11], sm[15],
        n_bodies);

    check(cudaGraphicsUnmapResources(1, &shadow_res_, 0), "unmap shadow");
}

void GpuRenderer::draw_shadow_triangles() const {
    if (n_shadow_verts_ <= 0) return;

    load_gl_procs();

    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, shadow_vbo_);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);

    glDrawArrays(GL_TRIANGLES, 0, n_shadow_verts_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
}

}  // namespace avbd
}  // namespace chysx
