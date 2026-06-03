// SPDX-License-Identifier: Apache-2.0
//
// Scene base class for the ChysX viewer.
//
// Each scene manages its own ClothSimulator instances + CUDA buffers
// and exposes a simple interface: setup → step → draw data.
// Scenes are registered as function pointers (like avbd-demo3d) for
// minimal boilerplate; a polymorphic base enables richer per-scene
// state while keeping the registration mechanism simple.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "math/vec.cuh"

namespace chysx {
namespace render {

// Per-object draw data handed from a Scene to the viewer each frame.
struct DrawMesh {
    const float* positions;   // flat xyz, n_points * 3
    int n_points;
    const int* triangles;     // flat 0-based, n_tris * 3
    int n_tris;
    float color_r, color_g, color_b;
    bool wireframe = false;   // draw as wireframe overlay
};

// Abstract scene interface.
class Scene {
public:
    virtual ~Scene() = default;

    // Human-readable name for the UI combo box.
    virtual const char* name() const = 0;

    // Called once when the scene is selected.  Allocate CUDA buffers,
    // configure simulators, build constraints.
    virtual void setup() = 0;

    // Advance the simulation by one step.
    virtual void step(float dt) = 0;

    // After step(), populate `out` with meshes the viewer should draw.
    // The pointers inside DrawMesh must remain valid until the next
    // call to step() or setup().
    virtual void draw_meshes(std::vector<DrawMesh>& out) = 0;

    // Optional: expose tunable parameters to ImGui.
    virtual void ui() {}

    // Optional: reset to initial state without full re-setup.
    virtual void reset() { setup(); }

    // Optional: custom OpenGL drawing after all DrawMeshes are rendered
    // (lines, points, shadow passes, etc.).
    virtual void draw_custom() {}

    // Optional: input forwarding from the viewer.  The ray is in world
    // space, computed from the camera.  Return true to consume the event.
    virtual bool on_mouse_down(int button,
                               float ray_ox, float ray_oy, float ray_oz,
                               float ray_dx, float ray_dy, float ray_dz) {
        (void)button; (void)ray_ox; (void)ray_oy; (void)ray_oz;
        (void)ray_dx; (void)ray_dy; (void)ray_dz;
        return false;
    }
    virtual void on_mouse_move(float ray_ox, float ray_oy, float ray_oz,
                               float ray_dx, float ray_dy, float ray_dz) {
        (void)ray_ox; (void)ray_oy; (void)ray_oz;
        (void)ray_dx; (void)ray_dy; (void)ray_dz;
    }
    virtual void on_mouse_up(int button) { (void)button; }
    virtual bool on_key_down(int key) { (void)key; return false; }
};

// Scene registry — a simple function-pointer table.
struct SceneEntry {
    const char* name;
    Scene* (*create)();   // factory: allocates a new Scene on the heap
};

// Populated in scenes/*.cu — call register_scene() at file scope.
void register_scene(const char* name, Scene* (*factory)());

// Access the global registry.
const std::vector<SceneEntry>& scene_registry();

// Force-link all scene translation units.
// Each scene .cu file defines its own extern "C" void chysx_register_scenes_XXX();
// and register_all_scenes() calls them all.
void register_all_scenes();

}  // namespace render
}  // namespace chysx
