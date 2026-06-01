// SPDX-License-Identifier: Apache-2.0

#include "viewer.h"

#include <cmath>
#include <cstdio>
#include <iostream>

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

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

namespace chysx {
namespace render {

Viewer::Viewer(const ViewerConfig& cfg)
    : cfg_(cfg), win_w_(cfg.width), win_h_(cfg.height) {}

Viewer::~Viewer() {
    shutdown_sdl();
}

bool Viewer::init_sdl() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    window_ = SDL_CreateWindow(
        cfg_.title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        cfg_.width, cfg_.height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window_) {
        std::cerr << "SDL_CreateWindow: " << SDL_GetError() << std::endl;
        return false;
    }

    context_ = SDL_GL_CreateContext(window_);
    SDL_GL_SetSwapInterval(1);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window_, context_);
    ImGui_ImplOpenGL3_Init("#version 120");

    return true;
}

void Viewer::shutdown_sdl() {
    if (context_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        SDL_GL_DeleteContext(context_);
        context_ = nullptr;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

void Viewer::run() {
    if (!init_sdl()) return;

    register_all_scenes();

    const auto& reg = scene_registry();
    if (!reg.empty()) {
        switch_scene(2);
    }

    running_ = true;
    while (running_) {
        main_loop();
    }
}

void Viewer::main_loop() {
    handle_events();

    // Simulation
    if (!paused_ && scene_) {
        scene_->step(cfg_.dt);
    }

    // Render
    SDL_GL_GetDrawableSize(window_, &win_w_, &win_h_);
    glViewport(0, 0, win_w_, win_h_);

    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    update_camera();
    draw_grid();
    draw_scene();

    // ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
    draw_ui();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(window_);
}

void Viewer::handle_events() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);

        if (e.type == SDL_QUIT) {
            running_ = false;
        }
        if (e.type == SDL_WINDOWEVENT &&
            e.window.event == SDL_WINDOWEVENT_CLOSE) {
            running_ = false;
        }

        auto& io = ImGui::GetIO();
        if (io.WantCaptureMouse || io.WantCaptureKeyboard)
            continue;

        if (e.type == SDL_MOUSEBUTTONDOWN) {
            float rox, roy, roz, rdx, rdy, rdz;
            bool hasRay = screen_to_ray(e.button.x, e.button.y,
                                        rox, roy, roz, rdx, rdy, rdz);
            bool consumed = false;
            if (hasRay && scene_)
                consumed = scene_->on_mouse_down(e.button.button,
                                                 rox, roy, roz, rdx, rdy, rdz);
            if (!consumed) {
                if (e.button.button == SDL_BUTTON_LEFT) mouse_left_ = true;
                if (e.button.button == SDL_BUTTON_RIGHT) mouse_right_ = true;
            }
            mouse_x_ = e.button.x;
            mouse_y_ = e.button.y;
        }
        if (e.type == SDL_MOUSEBUTTONUP) {
            if (scene_) scene_->on_mouse_up(e.button.button);
            if (e.button.button == SDL_BUTTON_LEFT) mouse_left_ = false;
            if (e.button.button == SDL_BUTTON_RIGHT) mouse_right_ = false;
        }
        if (e.type == SDL_MOUSEMOTION) {
            int dx = e.motion.x - mouse_x_;
            int dy = e.motion.y - mouse_y_;
            mouse_x_ = e.motion.x;
            mouse_y_ = e.motion.y;

            if (scene_) {
                float rox, roy, roz, rdx2, rdy2, rdz2;
                if (screen_to_ray(e.motion.x, e.motion.y,
                                  rox, roy, roz, rdx2, rdy2, rdz2))
                    scene_->on_mouse_move(rox, roy, roz, rdx2, rdy2, rdz2);
            }

            if (mouse_left_) {
                cam_azimuth_ -= dx * 0.005f;
                cam_elevation_ += dy * 0.005f;
                if (cam_elevation_ > 1.5f) cam_elevation_ = 1.5f;
                if (cam_elevation_ < -0.2f) cam_elevation_ = -0.2f;
            }
            if (mouse_right_) {
                float cos_a = std::cos(cam_azimuth_);
                float sin_a = std::sin(cam_azimuth_);
                float scale = cam_distance_ * 0.002f;
                cam_target_[0] += (-cos_a * dx + sin_a * dy * std::sin(cam_elevation_)) * scale;
                cam_target_[1] += (-sin_a * dx - cos_a * dy * std::sin(cam_elevation_)) * scale;
                cam_target_[2] += dy * std::cos(cam_elevation_) * scale;
            }
        }
        if (e.type == SDL_MOUSEWHEEL) {
            cam_distance_ *= (e.wheel.y > 0) ? 0.9f : 1.1f;
            if (cam_distance_ < 0.1f) cam_distance_ = 0.1f;
            if (cam_distance_ > 1000.0f) cam_distance_ = 1000.0f;
        }
        if (e.type == SDL_KEYDOWN) {
            bool consumed = false;
            if (scene_) consumed = scene_->on_key_down(e.key.keysym.sym);
            if (!consumed) {
                if (e.key.keysym.sym == SDLK_SPACE) paused_ = !paused_;
                if (e.key.keysym.sym == SDLK_r && scene_) {
                    scene_->reset();
                }
            }
        }
    }
}

void Viewer::update_camera() {
    float aspect = static_cast<float>(win_w_) /
                   static_cast<float>(win_h_ > 0 ? win_h_ : 1);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float fov_y = 45.0f;
    float near_p = 0.01f, far_p = 2000.0f;
    float top = near_p * std::tan(fov_y * 0.5f * 3.14159f / 180.0f);
    float right = top * aspect;
    glFrustum(-right, right, -top, top, near_p, far_p);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float ex = cam_target_[0] + cam_distance_ * std::cos(cam_elevation_) * std::cos(cam_azimuth_);
    float ey = cam_target_[1] + cam_distance_ * std::cos(cam_elevation_) * std::sin(cam_azimuth_);
    float ez = cam_target_[2] + cam_distance_ * std::sin(cam_elevation_);

    // gluLookAt equivalent
    float fx = cam_target_[0] - ex;
    float fy = cam_target_[1] - ey;
    float fz = cam_target_[2] - ez;
    float len = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= len; fy /= len; fz /= len;

    float ux = 0, uy = 0, uz = 1;  // world up = Z
    // side = f × up
    float sx = fy*uz - fz*uy;
    float sy = fz*ux - fx*uz;
    float sz = fx*uy - fy*ux;
    len = std::sqrt(sx*sx + sy*sy + sz*sz);
    sx /= len; sy /= len; sz /= len;
    // u = s × f
    ux = sy*fz - sz*fy;
    uy = sz*fx - sx*fz;
    uz = sx*fy - sy*fx;

    float m[16] = {
        sx,  ux, -fx, 0,
        sy,  uy, -fy, 0,
        sz,  uz, -fz, 0,
        0,   0,   0,  1
    };
    glMultMatrixf(m);
    glTranslatef(-ex, -ey, -ez);
}

void Viewer::draw_grid() {
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINES);
    const int extent = 50;
    for (int i = -extent; i <= extent; ++i) {
        float f = static_cast<float>(i);
        glVertex3f(f, static_cast<float>(-extent), 0.0f);
        glVertex3f(f, static_cast<float>(extent), 0.0f);
        glVertex3f(static_cast<float>(-extent), f, 0.0f);
        glVertex3f(static_cast<float>(extent), f, 0.0f);
    }
    glEnd();
}

void Viewer::draw_scene() {
    if (!scene_) return;

    std::vector<DrawMesh> meshes;
    scene_->draw_meshes(meshes);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    float light_pos[] = {2.0f, -3.0f, 5.0f, 0.0f};
    float light_amb[] = {0.3f, 0.3f, 0.3f, 1.0f};
    float light_dif[] = {0.8f, 0.8f, 0.8f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);

    for (const auto& m : meshes) {
        draw_mesh(m);
    }

    glDisable(GL_LIGHTING);

    if (scene_) scene_->draw_custom();
}

void Viewer::draw_mesh(const DrawMesh& m) {
    const float* P = m.positions;
    const int* T = m.triangles;

    glColor3f(m.color_r, m.color_g, m.color_b);

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < m.n_tris; ++i) {
        int i0 = T[i * 3], i1 = T[i * 3 + 1], i2 = T[i * 3 + 2];
        float ax = P[i1*3]-P[i0*3], ay = P[i1*3+1]-P[i0*3+1], az = P[i1*3+2]-P[i0*3+2];
        float bx = P[i2*3]-P[i0*3], by = P[i2*3+1]-P[i0*3+1], bz = P[i2*3+2]-P[i0*3+2];
        float nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }
        glNormal3f(nx, ny, nz);
        glVertex3f(P[i0*3], P[i0*3+1], P[i0*3+2]);
        glVertex3f(P[i1*3], P[i1*3+1], P[i1*3+2]);
        glVertex3f(P[i2*3], P[i2*3+1], P[i2*3+2]);
    }
    glEnd();

    // Wireframe overlay
    if (m.wireframe) {
        glDisable(GL_LIGHTING);
        glColor3f(0.0f, 0.0f, 0.0f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_TRIANGLES);
        for (int i = 0; i < m.n_tris; ++i) {
            int i0 = T[i*3], i1 = T[i*3+1], i2 = T[i*3+2];
            glVertex3f(P[i0*3], P[i0*3+1], P[i0*3+2]);
            glVertex3f(P[i1*3], P[i1*3+1], P[i1*3+2]);
            glVertex3f(P[i2*3], P[i2*3+1], P[i2*3+2]);
        }
        glEnd();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_LIGHTING);
    }
}

void Viewer::draw_ui() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("ChysX Viewer");

    const auto& reg = scene_registry();
    if (!reg.empty()) {
        const char* cur_name = (current_scene_ < static_cast<int>(reg.size()))
                               ? reg[current_scene_].name : "None";
        if (ImGui::BeginCombo("Scene", cur_name)) {
            for (int i = 0; i < static_cast<int>(reg.size()); ++i) {
                bool selected = (i == current_scene_);
                if (ImGui::Selectable(reg[i].name, selected)) {
                    if (i != current_scene_) {
                        switch_scene(i);
                    }
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    if (ImGui::Button(paused_ ? "Resume" : "Pause")) {
        paused_ = !paused_;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step") && scene_) {
        scene_->step(cfg_.dt);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset") && scene_) {
        scene_->reset();
    }

    ImGui::Separator();
    ImGui::Text("Camera: dist=%.2f", cam_distance_);
    ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

    if (scene_) {
        ImGui::Separator();
        scene_->ui();
    }

    ImGui::End();
}

bool Viewer::screen_to_ray(int sx, int sy,
                           float& ox, float& oy, float& oz,
                           float& dx, float& dy, float& dz) const {
    if (win_w_ <= 0 || win_h_ <= 0) return false;
    float aspect = static_cast<float>(win_w_) / static_cast<float>(win_h_);
    float ndcX = static_cast<float>(sx) / static_cast<float>(win_w_) * 2.0f - 1.0f;
    float ndcY = 1.0f - static_cast<float>(sy) / static_cast<float>(win_h_) * 2.0f;

    float ex = cam_target_[0] + cam_distance_ * std::cos(cam_elevation_) * std::cos(cam_azimuth_);
    float ey = cam_target_[1] + cam_distance_ * std::cos(cam_elevation_) * std::sin(cam_azimuth_);
    float ez = cam_target_[2] + cam_distance_ * std::sin(cam_elevation_);

    float fx = cam_target_[0] - ex;
    float fy = cam_target_[1] - ey;
    float fz = cam_target_[2] - ez;
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;

    // right = forward x up(0,0,1)
    float rx = fy * 1.0f - fz * 0.0f;
    float ry = fz * 0.0f - fx * 1.0f;
    float rz = fx * 0.0f - fy * 0.0f;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    if (rl < 1e-8f) { rx = 0; ry = 1; rz = 0; rl = 1; }
    rx /= rl; ry /= rl; rz /= rl;
    // up = right x forward
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    float fov_y = 45.0f;
    float tanHalf = std::tan(0.5f * fov_y * 3.14159265f / 180.0f);
    float px = ndcX * aspect * tanHalf;
    float py = ndcY * tanHalf;

    ox = ex; oy = ey; oz = ez;
    float rdx = fx + rx * px + ux * py;
    float rdy = fy + ry * px + uy * py;
    float rdz = fz + rz * px + uz * py;
    float rdl = std::sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
    dx = rdx / rdl; dy = rdy / rdl; dz = rdz / rdl;
    return true;
}

void Viewer::switch_scene(int index) {
    const auto& reg = scene_registry();
    if (index < 0 || index >= static_cast<int>(reg.size())) return;
    current_scene_ = index;
    scene_.reset(reg[index].create());
    scene_->setup();
    paused_ = false;
}

}  // namespace render
}  // namespace chysx
