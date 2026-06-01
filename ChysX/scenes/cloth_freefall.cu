// SPDX-License-Identifier: Apache-2.0
//
// Scene: two cloths + box obstacle, different colors.

#include <cstring>
#include <vector>

#include "cloth/cloth_simulator.h"
#include "collision/static_contact.h"
#include "io/bgeo_writer.h"
#include "math/vec.cuh"
#include "memory/cuda_array.h"
#include "render/scene.h"

namespace {

using namespace chysx;
using namespace chysx::render;

static constexpr int GRID_W = 21;
static constexpr int GRID_H = 21;

struct ClothPiece {
    CudaArray<math::Vec3f> pos_buf;
    CudaArray<math::Vec3f> vel_buf;
    CudaArray<float> inv_mass_buf;
    cloth::ClothSimulator sim;

    std::vector<math::Vec3f> host_pos;
    std::vector<math::Vec3i> host_tris;
    int n_points = 0;
    int n_tris = 0;

    std::vector<float> flat_pos;
    std::vector<int> flat_tris;
};

static void generate_grid(int w, int h, float size, float cx, float cy,
                          float z,
                          std::vector<math::Vec3f>& positions,
                          std::vector<math::Vec3i>& triangles) {
    positions.resize(static_cast<size_t>(w) * h);
    float dx = size / static_cast<float>(w - 1);
    float dy = size / static_cast<float>(h - 1);
    float x0 = cx - size * 0.5f;
    float y0 = cy - size * 0.5f;

    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            positions[j * w + i] = {x0 + i * dx, y0 + j * dy, z};

    triangles.clear();
    triangles.reserve(static_cast<size_t>((w - 1) * (h - 1) * 2));
    for (int j = 0; j < h - 1; ++j) {
        for (int i = 0; i < w - 1; ++i) {
            int v00 = j * w + i, v10 = j * w + i + 1;
            int v01 = (j + 1) * w + i, v11 = (j + 1) * w + i + 1;
            triangles.push_back({v00, v10, v11});
            triangles.push_back({v00, v11, v01});
        }
    }
}

static void setup_cloth(ClothPiece& c, float size, float cx, float cy,
                        float z, float stretch_k, float shear_k,
                        float bend_k, float density, float damping,
                        float box_cx, float box_cy, float box_cz,
                        float box_hx, float box_hy, float box_hz) {
    generate_grid(GRID_W, GRID_H, size, cx, cy, z,
                  c.host_pos, c.host_tris);
    c.n_points = static_cast<int>(c.host_pos.size());
    c.n_tris = static_cast<int>(c.host_tris.size());

    c.pos_buf.resize(c.n_points);
    c.vel_buf.resize(c.n_points);
    c.inv_mass_buf.resize(c.n_points);

    std::memcpy(c.pos_buf.cpu_data(), c.host_pos.data(),
                c.n_points * sizeof(math::Vec3f));
    c.pos_buf.copy_to_device();
    std::memset(c.vel_buf.cpu_data(), 0,
                c.n_points * sizeof(math::Vec3f));
    c.vel_buf.copy_to_device();
    for (int i = 0; i < c.n_points; ++i) c.inv_mass_buf[i] = 1.0f;
    c.inv_mass_buf.copy_to_device();

    cloth::ClothMaterial mat;
    mat.density = density;
    mat.damping = damping;
    mat.gz = -9.81f;
    c.sim.set_material(mat);

    c.sim.set_external_buffers(c.pos_buf.gpu_ptr(), c.vel_buf.gpu_ptr(),
                               c.n_points, c.inv_mass_buf.gpu_ptr());
    c.sim.set_mesh(c.host_tris.data(), c.n_tris);
    c.sim.redistribute_mass_area_weighted(
        density, c.inv_mass_buf.gpu_ptr(), c.n_points);

    c.sim.build_springs_from_current_positions(stretch_k);
    c.sim.build_fem_stretch_from_current_positions(stretch_k);
    c.sim.build_fem_shear_from_current_positions(shear_k);
    c.sim.build_bending_from_current_positions(bend_k);

    int pins[2] = {(GRID_H - 1) * GRID_W, (GRID_H - 1) * GRID_W + GRID_W - 1};
    math::Vec3f targets[2] = {c.host_pos[pins[0]], c.host_pos[pins[1]]};
    c.sim.set_pins(pins, targets, 2, 1.0e6f);

    collision::PlaneShape ground;
    ground.n = {0, 0, 1}; ground.d = 0;
    c.sim.add_static_plane(ground);
    c.sim.set_static_contact_thickness(0.01f);
    c.sim.set_static_contact_stiffness(1.0e4f);

    if (box_hx > 0) {
        collision::BoxShape box;
        box.center = {box_cx, box_cy, box_cz};
        box.half_ext = {box_hx, box_hy, box_hz};
        box.ex = {1, 0, 0}; box.ey = {0, 1, 0}; box.ez = {0, 0, 1};
        c.sim.add_static_box(box);
    }

    c.flat_pos.resize(static_cast<size_t>(c.n_points) * 3);
    c.flat_tris.resize(static_cast<size_t>(c.n_tris) * 3);
    for (int i = 0; i < c.n_tris; ++i) {
        c.flat_tris[i * 3] = c.host_tris[i].x;
        c.flat_tris[i * 3 + 1] = c.host_tris[i].y;
        c.flat_tris[i * 3 + 2] = c.host_tris[i].z;
    }
}

static void update_flat_pos(ClothPiece& c) {
    c.pos_buf.copy_to_host();
    auto* p = c.pos_buf.cpu_data();
    for (int i = 0; i < c.n_points; ++i) {
        c.flat_pos[i * 3] = p[i].x;
        c.flat_pos[i * 3 + 1] = p[i].y;
        c.flat_pos[i * 3 + 2] = p[i].z;
    }
}

// ====================================================================
// Scene 1: Two cloths + box
// ====================================================================

class TwoClothBoxScene : public Scene {
public:
    const char* name() const override { return "Two Cloths + Box"; }

    void setup() override {
        io::generate_box_mesh(0, 0, 0.6f, 0.2f, 0.2f, 0.05f,
                              box_pos_, box_tris_);
        box_np_ = static_cast<int>(box_pos_.size()) / 3;
        box_nt_ = static_cast<int>(box_tris_.size()) / 3;

        a_ = std::make_unique<ClothPiece>();
        b_ = std::make_unique<ClothPiece>();
        setup_cloth(*a_, 0.8f, -0.3f, 0, 1.5f, 5000, 500, 0.01f, 0.1f, 2,
                    0, 0, 0.6f, 0.2f, 0.2f, 0.05f);
        setup_cloth(*b_, 0.8f,  0.3f, 0, 1.2f, 5000, 500, 0.01f, 0.1f, 2,
                    0, 0, 0.6f, 0.2f, 0.2f, 0.05f);
        update_flat_pos(*a_);
        update_flat_pos(*b_);
    }

    void step(float dt) override {
        a_->sim.step(dt);
        b_->sim.step(dt);
        update_flat_pos(*a_);
        update_flat_pos(*b_);
    }

    void draw_meshes(std::vector<DrawMesh>& out) override {
        out.push_back({a_->flat_pos.data(), a_->n_points,
                       a_->flat_tris.data(), a_->n_tris,
                       0.2f, 0.4f, 0.9f});
        out.push_back({b_->flat_pos.data(), b_->n_points,
                       b_->flat_tris.data(), b_->n_tris,
                       0.2f, 0.9f, 0.3f});
        out.push_back({box_pos_.data(), box_np_,
                       box_tris_.data(), box_nt_,
                       0.9f, 0.2f, 0.2f});
    }

private:
    std::unique_ptr<ClothPiece> a_, b_;
    std::vector<float> box_pos_;
    std::vector<int> box_tris_;
    int box_np_ = 0, box_nt_ = 0;
};

// ====================================================================
// Scene 2: Single cloth drop (simpler)
// ====================================================================

class SingleClothScene : public Scene {
public:
    const char* name() const override { return "Single Cloth Drop"; }

    void setup() override {
        c_ = std::make_unique<ClothPiece>();
        setup_cloth(*c_, 1.0f, 0, 0, 1.5f, 5000, 500, 0.01f, 0.1f, 2,
                    0, 0, 0, 0, 0, 0);
        update_flat_pos(*c_);
    }

    void step(float dt) override {
        c_->sim.step(dt);
        update_flat_pos(*c_);
    }

    void draw_meshes(std::vector<DrawMesh>& out) override {
        out.push_back({c_->flat_pos.data(), c_->n_points,
                       c_->flat_tris.data(), c_->n_tris,
                       0.3f, 0.6f, 1.0f});
    }

private:
    std::unique_ptr<ClothPiece> c_;
};

}  // anonymous namespace

extern "C" void chysx_register_cloth_scenes() {
    using namespace chysx::render;
    register_scene("Two Cloths + Box",
                    []() -> Scene* { return new TwoClothBoxScene(); });
    register_scene("Single Cloth Drop",
                    []() -> Scene* { return new SingleClothScene(); });
}
