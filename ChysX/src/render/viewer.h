// SPDX-License-Identifier: Apache-2.0
//
// ChysX real-time viewer: SDL2 + OpenGL + ImGui.
//
// Draws triangle meshes produced by Scene::draw_meshes() with simple
// per-face flat shading, orbit camera, and an ImGui control panel for
// scene switching, pause/step, and per-scene tuning.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "scene.h"

struct SDL_Window;
typedef void* SDL_GLContext;

namespace chysx {
namespace render {

struct ViewerConfig {
    int width  = 1280;
    int height = 720;
    const char* title = "ChysX Viewer";
    float dt = 1.0f / 60.0f;
};

class Viewer {
public:
    explicit Viewer(const ViewerConfig& cfg = {});
    ~Viewer();

    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    // Run the main loop (blocks until window is closed).
    void run();

private:
    bool init_sdl();
    void shutdown_sdl();

    void main_loop();
    void handle_events();
    void update_camera();
    void draw_scene();
    void draw_grid();
    void draw_mesh(const DrawMesh& m);
    void draw_ui();

    void switch_scene(int index);
    bool screen_to_ray(int sx, int sy,
                       float& ox, float& oy, float& oz,
                       float& dx, float& dy, float& dz) const;

    ViewerConfig cfg_;
    SDL_Window* window_ = nullptr;
    SDL_GLContext context_ = nullptr;
    bool running_ = false;
    bool paused_ = false;
    int current_scene_ = 0;
    std::unique_ptr<Scene> scene_;

    // camera
    float cam_distance_ = 3.0f;
    float cam_azimuth_  = 1.2f;   // radians
    float cam_elevation_ = 0.5f;
    float cam_target_[3] = {0.0f, 0.0f, 0.6f};

    // mouse state
    bool mouse_left_ = false;
    bool mouse_right_ = false;
    int mouse_x_ = 0, mouse_y_ = 0;

    // window size (updated on resize)
    int win_w_, win_h_;
};

}  // namespace render
}  // namespace chysx
