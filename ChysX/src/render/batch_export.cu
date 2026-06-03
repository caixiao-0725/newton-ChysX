// SPDX-License-Identifier: Apache-2.0
//
// Batch export entry point — runs a scene headlessly and exports
// each frame as OBJ + bgeo.

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "io/bgeo_writer.h"
#include "io/obj_io.h"
#include "render/scene.h"

#ifndef ASSET_PATH
#define ASSET_PATH "./"
#endif

static constexpr int N_FRAMES = 120;
static constexpr float DT = 1.0f / 60.0f;

int main() {
    chysx::render::register_all_scenes();
    const auto& reg = chysx::render::scene_registry();
    if (reg.empty()) {
        std::cerr << "No scenes registered" << std::endl;
        return 1;
    }

    auto* scene = reg[0].create();
    scene->setup();

    std::string out_dir = std::string(ASSET_PATH) + "output/";
    std::cout << "=== ChysX Batch Export ===" << std::endl;
    std::cout << "Scene: " << scene->name() << std::endl;
    std::cout << "Frames: " << N_FRAMES << "  dt=" << DT << std::endl;

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        scene->step(DT);

        std::vector<chysx::render::DrawMesh> meshes;
        scene->draw_meshes(meshes);

        // Build BgeoMeshPiece from DrawMesh
        std::vector<chysx::io::BgeoMeshPiece> pieces;
        for (const auto& m : meshes) {
            pieces.push_back({m.positions, m.n_points,
                              m.triangles, m.n_tris,
                              m.color_r, m.color_g, m.color_b});
        }

        char filename[256];
        std::snprintf(filename, sizeof(filename),
                      "%sframe_%04d.bgeo", out_dir.c_str(), frame);
        chysx::io::BgeoWriter::write_multi(filename, pieces);

        // Per-piece OBJ
        for (int p = 0; p < static_cast<int>(meshes.size()); ++p) {
            const auto& m = meshes[p];
            std::snprintf(filename, sizeof(filename),
                          "%spiece%d_%04d.obj", out_dir.c_str(), p, frame);
            chysx::io::save_obj(filename, m.positions, m.n_points,
                                m.triangles, m.n_tris);
        }

        if ((frame + 1) % 10 == 0 || frame == 0) {
            std::printf("Frame %d/%d\n", frame + 1, N_FRAMES);
        }
    }

    delete scene;
    std::cout << "=== Done ===" << std::endl;
    return 0;
}
