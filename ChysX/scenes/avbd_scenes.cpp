// SPDX-License-Identifier: Apache-2.0
// AVBD rigid-body demo scenes for the ChysX viewer.
// Adapted from avbd-demo3d (Chris Giles, 2026).

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <imgui.h>

#include "render/scene.h"
#include "rigid/avbd_cpu/avbd_solver.h"
#include "rigid/avbd_cpu/avbd_gpu_solver.h"
#include "rigid/avbd_cpu/avbd_gpu_renderer.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace chysx::avbd;
using chysx::render::DrawMesh;
using chysx::render::Scene;
using chysx::render::register_scene;

namespace {

// ---- Box vertex / triangle tables ----

static const float kBoxV[8][3] = {
    {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f},
    {+0.5f, +0.5f, -0.5f}, {-0.5f, +0.5f, -0.5f},
    {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f},
    {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f}};

static const unsigned kBoxT[12][3] = {
    {0,1,2},{0,2,3}, {4,6,5},{4,7,6},
    {1,5,6},{1,6,2}, {4,0,3},{4,3,7},
    {3,2,6},{3,6,7}, {4,5,1},{4,1,0}};

static const unsigned kBoxE[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

inline float3 bodyVertexWorld(const Rigid* body, const float v[3]) {
    float3 local{v[0] * body->size.x, v[1] * body->size.y, v[2] * body->size.z};
    return transform(body->positionLin, body->positionAng, local);
}

// ---- Shadow helpers ----

inline void makePlaneFromPointNormal(float3 p, float3 n, float plane[4]) {
    float3 nn = normalize(n);
    plane[0] = nn.x; plane[1] = nn.y; plane[2] = nn.z;
    plane[3] = -(nn.x * p.x + nn.y * p.y + nn.z * p.z);
}

inline void makeShadowMatrix(float out[16], const float light[4], const float plane[4]) {
    const float d = plane[0]*light[0] + plane[1]*light[1] + plane[2]*light[2] + plane[3]*light[3];
#define SM(r,c) out[(c)*4+(r)]
    SM(0,0)=d-light[0]*plane[0]; SM(0,1)=-light[0]*plane[1]; SM(0,2)=-light[0]*plane[2]; SM(0,3)=-light[0]*plane[3];
    SM(1,0)=-light[1]*plane[0]; SM(1,1)=d-light[1]*plane[1]; SM(1,2)=-light[1]*plane[2]; SM(1,3)=-light[1]*plane[3];
    SM(2,0)=-light[2]*plane[0]; SM(2,1)=-light[2]*plane[1]; SM(2,2)=d-light[2]*plane[2]; SM(2,3)=-light[2]*plane[3];
    SM(3,0)=-light[3]*plane[0]; SM(3,1)=-light[3]*plane[1]; SM(3,2)=-light[3]*plane[2]; SM(3,3)=d-light[3]*plane[3];
#undef SM
}

inline float3 applyProjectiveMatrix(const float m[16], float3 p) {
    float x = m[0]*p.x + m[4]*p.y + m[8]*p.z  + m[12];
    float y = m[1]*p.x + m[5]*p.y + m[9]*p.z  + m[13];
    float z = m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14];
    float w = m[3]*p.x + m[7]*p.y + m[11]*p.z + m[15];
    if (std::fabs(w) > 1.0e-6f) { x/=w; y/=w; z/=w; }
    return {x, y, z};
}

inline bool findShadowPlane(const Solver* solver, float3& planePoint, float3& planeNormal) {
    if (solver->has_ground_plane) {
        planePoint  = {0, 0, solver->ground_z};
        planeNormal = {0, 0, 1};
        return true;
    }
    const float3 up{0, 0, 1};
    float bestScore = 0.0f;
    bool found = false;
    for (const Rigid* body = solver->bodies; body; body = body->next) {
        if (body->mass > 0.0f) continue;
        float3 half = body->size * 0.5f;
        float3 axes[3] = {
            rotate(body->positionAng, float3{1,0,0}),
            rotate(body->positionAng, float3{0,1,0}),
            rotate(body->positionAng, float3{0,0,1})};
        for (int axis = 0; axis < 3; ++axis) {
            int i1 = (axis+1)%3, i2 = (axis+2)%3;
            float area = 4.0f * half[i1] * half[i2];
            if (area <= 0.0f) continue;
            for (int s = 0; s < 2; ++s) {
                float sign = s == 0 ? -1.0f : 1.0f;
                float3 n = axes[axis] * sign;
                float upness = dot(n, up);
                if (upness <= 0.15f) continue;
                float score = area * upness;
                if (!found || score > bestScore) {
                    found = true;
                    bestScore = score;
                    planeNormal = n;
                    planePoint = body->positionLin + n * half[axis];
                }
            }
        }
    }
    return found;
}

// ---- AVBDScene base ----

using SceneSetupFn = std::function<void(Solver*)>;

class AVBDScene : public Scene {
public:
    AVBDScene(const char* sceneName, SceneSetupFn setupFn)
        : name_(sceneName), setupFn_(std::move(setupFn)) {
        solver_ = new Solver();
    }
    ~AVBDScene() override { releaseDrag(); delete gpu_renderer_; delete solver_; }

    const char* name() const override { return name_; }

    void setup() override {
        releaseDrag();
        setupFn_(solver_);
    }

    void step(float /*dt*/) override {
        solver_->step();
        updateGpuRender();
    }

    void reset() override {
        releaseDrag();
        setupFn_(solver_);
    }

    void draw_meshes(std::vector<DrawMesh>& /*out*/) override {}

    void drawGroundPlane() {
        if (!solver_->has_ground_plane) return;
        float z = solver_->ground_z;
        float ext = 200.0f;
        glColor4f(0.85f, 0.87f, 0.90f, 1.0f);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex3f(-ext, -ext, z);
        glVertex3f( ext, -ext, z);
        glVertex3f( ext,  ext, z);
        glVertex3f(-ext,  ext, z);
        glEnd();
    }

    void drawSphereCollider() {
        if (!solver_->has_sphere_collider) return;
        float R = solver_->sphere_radius;
        float spx, spy, spz, sqx, sqy, sqz, sqw;
        int idx = solver_->soa_.count - 1;

        if (solver_->gpu_state_valid_ && solver_->gpu_solver()) {
            solver_->gpu_solver()->download_body_pose(
                idx, spx, spy, spz, sqx, sqy, sqz, sqw);
        } else {
            Rigid* b = solver_->bodies;
            while (b && b->next) b = b->next;
            if (!b) return;
            spx = b->positionLin.x; spy = b->positionLin.y; spz = b->positionLin.z;
            sqx = b->positionAng.x; sqy = b->positionAng.y;
            sqz = b->positionAng.z; sqw = b->positionAng.w;
        }

        constexpr int SLICES = 24, STACKS = 16;
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.9f, 0.3f, 0.2f, 0.6f);
        glEnable(GL_LIGHTING);

        glBegin(GL_TRIANGLES);
        for (int j = 0; j < STACKS; j++) {
            float phi0  = M_PI * (float)j / STACKS - M_PI * 0.5f;
            float phi1  = M_PI * (float)(j+1) / STACKS - M_PI * 0.5f;
            float cp0 = cosf(phi0), sp0 = sinf(phi0);
            float cp1 = cosf(phi1), sp1 = sinf(phi1);
            for (int i = 0; i < SLICES; i++) {
                float th0 = 2.0f * M_PI * (float)i / SLICES;
                float th1 = 2.0f * M_PI * (float)(i+1) / SLICES;
                float ct0 = cosf(th0), st0 = sinf(th0);
                float ct1 = cosf(th1), st1 = sinf(th1);
                float3 v00 = {cp0*ct0, cp0*st0, sp0};
                float3 v10 = {cp1*ct0, cp1*st0, sp1};
                float3 v01 = {cp0*ct1, cp0*st1, sp0};
                float3 v11 = {cp1*ct1, cp1*st1, sp1};
                auto emit = [&](float3 n) {
                    glNormal3f(n.x, n.y, n.z);
                    glVertex3f(spx + n.x*R, spy + n.y*R, spz + n.z*R);
                };
                emit(v00); emit(v10); emit(v11);
                emit(v00); emit(v11); emit(v01);
            }
        }
        glEnd();
        glDisable(GL_BLEND);
    }

    void draw_custom() override {
        // GPU path: all bodies rendered via VBO from CUDA-GL interop
        if (gpu_renderer_ && solver_->gpu_state_valid_ && gpu_renderer_->n_bodies() > 0) {
            GpuRenderer* renderer = gpu_renderer_;
            glEnable(GL_LINE_SMOOTH);
            glLineWidth(2.0f);
            glDisable(GL_LIGHTING);
            glShadeModel(GL_FLAT);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);

            // Filled boxes
            glColor4f(0.80f, 0.84f, 0.90f, 1.0f);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f);

            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            glEnable(GL_COLOR_MATERIAL);
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
            float light_pos[] = {2.0f, -3.0f, 5.0f, 0.0f};
            float light_amb[] = {0.2f, 0.2f, 0.2f, 1.0f};
            float light_dif[] = {0.9f, 0.9f, 0.9f, 1.0f};
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
            glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);

            renderer->draw_triangles();
            drawGroundPlane();
            drawSphereCollider();
            glDisable(GL_LIGHTING);
            glDisable(GL_POLYGON_OFFSET_FILL);

            // Projected shadows (compute shadow matrix on CPU, project on GPU)
            drawProjectedShadowsGpu(renderer);

            // Wireframe edges
            glColor4f(0.10f, 0.12f, 0.14f, 1.0f);
            renderer->draw_edges();

            // Forces still drawn on CPU (joints/springs are few)
            glPointSize(3.0f);
            for (const Force* f = solver_->forces; f; f = f->next) {
                if (const Joint* j = dynamic_cast<const Joint*>(f))
                    drawJoint(j);
                else if (const Spring* s = dynamic_cast<const Spring*>(f))
                    drawSpring(s);
                else if (const Manifold* m = dynamic_cast<const Manifold*>(f))
                    drawManifold(m);
            }
        } else {
            // CPU fallback path (original)
            glEnable(GL_LINE_SMOOTH);
            glLineWidth(2.0f);
            glPointSize(3.0f);
            glDisable(GL_LIGHTING);
            glShadeModel(GL_FLAT);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);

            drawGroundPlane();
            drawSphereCollider();

            for (const Rigid* body = solver_->bodies; body; body = body->next)
                if (body->mass <= 0.0f)
                    drawBody(body);

            drawProjectedShadows();

            for (const Rigid* body = solver_->bodies; body; body = body->next)
                if (body->mass > 0.0f)
                    drawBody(body);

            for (const Force* f = solver_->forces; f; f = f->next) {
                if (const Joint* j = dynamic_cast<const Joint*>(f))
                    drawJoint(j);
                else if (const Spring* s = dynamic_cast<const Spring*>(f))
                    drawSpring(s);
                else if (const Manifold* m = dynamic_cast<const Manifold*>(f))
                    drawManifold(m);
            }
        }
    }

    bool on_mouse_down(int button,
                       float ray_ox, float ray_oy, float ray_oz,
                       float ray_dx, float ray_dy, float ray_dz) override {
        if (button == SDL_BUTTON_LEFT) {
            float3 origin{ray_ox, ray_oy, ray_oz};
            float3 dir{ray_dx, ray_dy, ray_dz};
            float3 localHit;
            Rigid* body = solver_->pick(origin, dir, localHit);
            if (body) {
                float3 worldHit = transform(body->positionLin, body->positionAng, localHit);
                dragRayDist_ = max(dot(worldHit - origin, dir), 0.1f);
                const float dragStiffness = 5000.0f;
                drag_ = new Joint(solver_, nullptr, body, worldHit, localHit, dragStiffness, 0.0f);
                dragging_ = true;
                lastRayO_ = origin;
                lastRayD_ = dir;
                return true;
            }
        }
        if (button == SDL_BUTTON_MIDDLE) {
            shootBox(float3{ray_ox, ray_oy, ray_oz},
                     float3{ray_dx, ray_dy, ray_dz});
            return true;
        }
        return false;
    }

    void on_mouse_move(float ray_ox, float ray_oy, float ray_oz,
                       float ray_dx, float ray_dy, float ray_dz) override {
        lastRayO_ = float3{ray_ox, ray_oy, ray_oz};
        lastRayD_ = float3{ray_dx, ray_dy, ray_dz};
        if (drag_ && dragging_)
            drag_->rA = lastRayO_ + lastRayD_ * dragRayDist_;
    }

    void on_mouse_up(int button) override {
        if (button == SDL_BUTTON_LEFT)
            releaseDrag();
    }

    bool on_key_down(int key) override {
        if (key == SDLK_f) {
            shootBox(lastRayO_, lastRayD_);
            return true;
        }
        return false;
    }

    void ui() override {
        ImGui::Text("Drag: Left Mouse");
        ImGui::Text("Shoot: Middle Mouse / F");
        ImGui::Separator();
        ImGui::SliderFloat("Box Friction", &boxFriction_, 0.0f, 2.0f);
        ImGui::SliderFloat3("Box Size", &boxSize_.x, 0.1f, 5.0f);
        ImGui::SliderFloat("Box Velocity", &boxVelocity_, 0.0f, 40.0f);
        ImGui::Separator();
        ImGui::SliderFloat("Gravity", &solver_->gravity, -20.0f, 20.0f);
        ImGui::SliderFloat("Dt", &solver_->dt, 0.001f, 0.1f);
        ImGui::SliderInt("Iterations", &solver_->iterations, 1, 50);
        ImGui::Checkbox("Show Contacts", &showContacts_);
    }

private:
    const char* name_;
    SceneSetupFn setupFn_;
    Solver* solver_ = nullptr;
    GpuRenderer* gpu_renderer_ = nullptr;

    // Drag state
    Joint* drag_ = nullptr;
    bool dragging_ = false;
    float dragRayDist_ = 0.0f;
    float3 lastRayO_{0,0,0};
    float3 lastRayD_{0,0,1};

    // Shoot parameters
    float3 boxSize_{1,1,1};
    float boxVelocity_ = 20.0f;
    float boxFriction_ = 0.5f;
    float boxDensity_ = 1.0f;

    bool showContacts_ = true;

    void updateGpuRender() {
        if (!solver_->gpu_state_valid_ || !solver_->gpu_solver()) return;
        if (!gpu_renderer_)
            gpu_renderer_ = new GpuRenderer();
        int render_count = solver_->soa_.count;
        if (solver_->has_sphere_collider) render_count--;
        gpu_renderer_->update(
            *solver_->gpu_solver(),
            solver_->gpu_solver()->half_x_dev(),
            solver_->gpu_solver()->half_y_dev(),
            solver_->gpu_solver()->half_z_dev(),
            solver_->gpu_solver()->mass_dev(),
            render_count);
    }

    void releaseDrag() {
        if (drag_) { delete drag_; drag_ = nullptr; }
        dragging_ = false;
    }

    void shootBox(float3 origin, float3 dir) {
        float spawnOffset = 2.0f + 0.5f * length(boxSize_);
        float3 spawnPos = origin + dir * spawnOffset;
        float3 velocity = dir * boxVelocity_;
        new Rigid(solver_, boxSize_, boxDensity_, boxFriction_, spawnPos, velocity);
    }

    // ---- Drawing helpers ----

    static void drawBody(const Rigid* body) {
        glColor4f(0.80f, 0.84f, 0.90f, 1.0f);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        glBegin(GL_TRIANGLES);
        for (int i = 0; i < 12; ++i) {
            float3 a = bodyVertexWorld(body, kBoxV[kBoxT[i][0]]);
            float3 b = bodyVertexWorld(body, kBoxV[kBoxT[i][1]]);
            float3 c = bodyVertexWorld(body, kBoxV[kBoxT[i][2]]);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
            glVertex3f(c.x, c.y, c.z);
        }
        glEnd();
        glDisable(GL_POLYGON_OFFSET_FILL);

        glColor4f(0.10f, 0.12f, 0.14f, 1.0f);
        glBegin(GL_LINES);
        for (int i = 0; i < 12; ++i) {
            float3 a = bodyVertexWorld(body, kBoxV[kBoxE[i][0]]);
            float3 b = bodyVertexWorld(body, kBoxV[kBoxE[i][1]]);
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
        }
        glEnd();
    }

    static void drawJoint(const Joint* joint) {
        float3 v0 = joint->bodyA ?
            transform(joint->bodyA->positionLin, joint->bodyA->positionAng, joint->rA) : joint->rA;
        float3 v1 = transform(joint->bodyB->positionLin, joint->bodyB->positionAng, joint->rB);
        glColor3f(0.75f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glEnd();
    }

    static void drawSpring(const Spring* spring) {
        float3 v0 = transform(spring->bodyA->positionLin, spring->bodyA->positionAng, spring->rA);
        float3 v1 = transform(spring->bodyB->positionLin, spring->bodyB->positionAng, spring->rB);
        glColor3f(0.75f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glEnd();
    }

    void drawManifold(const Manifold* manifold) const {
        if (!showContacts_) return;
        glColor3f(0.75f, 0.0f, 0.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < manifold->numContacts; ++i) {
            float3 v0 = transform(manifold->bodyA->positionLin, manifold->bodyA->positionAng,
                                  manifold->contacts[i].rA);
            float3 v1 = transform(manifold->bodyB->positionLin, manifold->bodyB->positionAng,
                                  manifold->contacts[i].rB);
            glVertex3f(v0.x, v0.y, v0.z);
            glVertex3f(v1.x, v1.y, v1.z);
        }
        glEnd();
    }

    void drawProjectedShadowsGpu(GpuRenderer* renderer) {
        GLint stencilBits = 0;
        glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
        bool useStencil = stencilBits > 0;

        float3 planePoint, planeNormal;
        if (!findShadowPlane(solver_, planePoint, planeNormal)) return;

        float plane[4];
        makePlaneFromPointNormal(planePoint, planeNormal, plane);

        float3 l = normalize(float3{0.45f, 0.95f, 1.0f});
        float light[4] = {l.x, l.y, l.z, 0.0f};

        float shadowMat[16];
        makeShadowMatrix(shadowMat, light, plane);

        // Update shadow VBO on GPU
        int shadow_count = solver_->soa_.count;
        if (solver_->has_sphere_collider) shadow_count--;
        renderer->update_shadows(
            *solver_->gpu_solver(),
            solver_->gpu_solver()->half_x_dev(),
            solver_->gpu_solver()->half_y_dev(),
            solver_->gpu_solver()->half_z_dev(),
            solver_->gpu_solver()->mass_dev(),
            shadow_count, shadowMat);

        GLboolean lightingWas = glIsEnabled(GL_LIGHTING);
        GLboolean polyOffWas  = glIsEnabled(GL_POLYGON_OFFSET_FILL);
        GLboolean stencilWas  = glIsEnabled(GL_STENCIL_TEST);
        GLboolean depthWrite  = GL_TRUE;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
        GLboolean colorMask[4]; glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);

        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);

        if (useStencil) {
            glEnable(GL_STENCIL_TEST);
            glClear(GL_STENCIL_BUFFER_BIT);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glStencilMask(0xFF);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            renderer->draw_shadow_triangles();
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glColor3f(0.72f, 0.72f, 0.72f);
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
            renderer->draw_shadow_triangles();
        } else {
            glColor3f(0.72f, 0.72f, 0.72f);
            renderer->draw_shadow_triangles();
        }

        glDepthMask(depthWrite);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        if (useStencil) glStencilMask(0xFF);
        polyOffWas  ? glEnable(GL_POLYGON_OFFSET_FILL) : glDisable(GL_POLYGON_OFFSET_FILL);
        lightingWas ? glEnable(GL_LIGHTING)            : glDisable(GL_LIGHTING);
        stencilWas && useStencil ? glEnable(GL_STENCIL_TEST) : glDisable(GL_STENCIL_TEST);
    }

    void drawProjectedShadows() {
        GLint stencilBits = 0;
        glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
        bool useStencil = stencilBits > 0;

        float3 planePoint, planeNormal;
        if (!findShadowPlane(solver_, planePoint, planeNormal)) return;

        float plane[4];
        makePlaneFromPointNormal(planePoint, planeNormal, plane);

        float3 l = normalize(float3{0.45f, 0.95f, 1.0f});
        float light[4] = {l.x, l.y, l.z, 0.0f};

        float shadowMat[16];
        makeShadowMatrix(shadowMat, light, plane);

        GLboolean lightingWas = glIsEnabled(GL_LIGHTING);
        GLboolean blendWas    = glIsEnabled(GL_BLEND);
        GLboolean cullWas     = glIsEnabled(GL_CULL_FACE);
        GLboolean polyOffWas  = glIsEnabled(GL_POLYGON_OFFSET_FILL);
        GLboolean stencilWas  = glIsEnabled(GL_STENCIL_TEST);
        GLboolean depthWrite  = GL_TRUE;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
        GLboolean colorMask[4]; glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);

        auto drawCasters = [&]() {
            for (const Rigid* body = solver_->bodies; body; body = body->next) {
                if (body->mass <= 0.0f) continue;
                glBegin(GL_TRIANGLES);
                for (int i = 0; i < 12; ++i) {
                    float3 a = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, kBoxV[kBoxT[i][0]]));
                    float3 b = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, kBoxV[kBoxT[i][1]]));
                    float3 c = applyProjectiveMatrix(shadowMat, bodyVertexWorld(body, kBoxV[kBoxT[i][2]]));
                    glVertex3f(a.x, a.y, a.z);
                    glVertex3f(b.x, b.y, b.z);
                    glVertex3f(c.x, c.y, c.z);
                }
                glEnd();
            }
        };

        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);

        if (useStencil) {
            glEnable(GL_STENCIL_TEST);
            glClear(GL_STENCIL_BUFFER_BIT);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glStencilMask(0xFF);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            drawCasters();
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glColor3f(0.72f, 0.72f, 0.72f);
            glStencilMask(0xFF);
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
            drawCasters();
        } else {
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glColor3f(0.72f, 0.72f, 0.72f);
            drawCasters();
        }

        // Restore state
        glDepthMask(depthWrite);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        if (useStencil) glStencilMask(0xFF);
        polyOffWas  ? glEnable(GL_POLYGON_OFFSET_FILL) : glDisable(GL_POLYGON_OFFSET_FILL);
        cullWas     ? glEnable(GL_CULL_FACE)           : glDisable(GL_CULL_FACE);
        lightingWas ? glEnable(GL_LIGHTING)            : glDisable(GL_LIGHTING);
        blendWas    ? glEnable(GL_BLEND)               : glDisable(GL_BLEND);
        stencilWas && useStencil ? glEnable(GL_STENCIL_TEST) : glDisable(GL_STENCIL_TEST);
    }
};

// ---- Scene setup functions ----

void setupPyramid(Solver* s) {
    const int SIZE = 50;
    s->clear();
    s->set_ground_plane(0.0f, 0.5f);

    // Sphere collider (created FIRST so it ends up at SoA index = body_count-1)
    const float R = 5.0f;
    const float sphereDensity = 500.0f;
    Rigid* sphere = new Rigid(s, {2*R, 2*R, 2*R}, sphereDensity, 0.5f,
                              {-40.0f, 0.0f, R + 5.0f}, {15.0f, 0.0f, 0.0f});
    float I = 0.4f * sphere->mass * R * R;
    sphere->moment = {I, I, I};
    s->has_sphere_collider = true;
    s->sphere_radius = R;
    s->sphere_friction = 0.5f;

    float rho = 0.1f;
    int SIZE_I = 10;
    int SIZE_J = 40;
    for(int i = 0; i < SIZE_I; i++){
        for(int j = 0; j < SIZE_J; j++){
            for (int y = 0; y < SIZE; y++)
                for (int x = 0; x < SIZE - y; x++) {
                    new Rigid(s, { 1, 0.5f, 0.5f }, rho, 0.5f,
                        { x * 1.01f + y * 0.5f - SIZE / 2.0f + i * 55.f, -4.0f*j, y * 0.5f });

                }
        }
    }
}

void setupPyramid2(Solver* s) {
    const int SIZE = 50;
    s->clear();
    s->set_ground_plane(0.0f, 0.5f);

    float rho = 0.1f;

    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE - y; x++) {
            new Rigid(s, { 1, 0.5f, 0.5f }, rho, 0.5f,
                { x * 1.01f + y * 0.5f - SIZE / 2.0f, 0.0f, y * 0.5f });
        }

}

void setupRope(Solver* s) {
    s->clear();
    s->set_ground_plane(-19.5f, 0.5f);
    Rigid* prev = nullptr;
    for (int i = 0; i < 20; i++) {
        Rigid* curr = new Rigid(s, {1, 0.5f, 0.5f}, i == 0 ? 0.0f : 1.0f, 0.5f,
                                {static_cast<float>(i), 0.0f, 10.0f});
        if (prev) new Joint(s, prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0});
        prev = curr;
    }
}

void setupHeavyRope(Solver* s) {
    const int N = 20;
    const float SZ = 5.0f;
    s->clear();
    s->set_ground_plane(-19.5f, 0.5f);
    Rigid* prev = nullptr;
    for (int i = 0; i < N; i++) {
        float3 sz = i == N-1 ? float3{SZ, SZ, SZ} : float3{1, 0.5f, 0.5f};
        Rigid* curr = new Rigid(s, sz, i == 0 ? 0.0f : 1.0f, 0.5f,
                                {static_cast<float>(i) + (i == N-1 ? SZ/2 : 0), 0.0f, 10.0f});
        if (prev)
            new Joint(s, prev, curr, {0.5f, 0, 0},
                      i == N-1 ? float3{-SZ/2, 0, 0} : float3{-0.5f, 0, 0});
        prev = curr;
    }
}

void setupStack(Solver* s) {
    s->clear();
    s->set_ground_plane(0.5f, 0.5f);
    for (int i = 0; i < 110; i++)
        new Rigid(s, {1, 1, 1}, 1.0f, 0.5f, {0, 0, i * 1.0f + 1.0f});
}

void setupSoftBody(Solver* s) {
    s->clear();
    s->set_ground_plane(0.5f, 0.5f);

    const float Klin = 1000.0f;
    const float Kang = 250.0f;
    const int W = 4, D = 4, H = 4, N = 3;
    const float sz = 0.8f, half = sz * 0.5f;
    const float baseZ = 8.0f, stackGap = 2.0f;

    for (int n = 0; n < N; n++) {
        Rigid* grid[W][D][H];
        float stackZ = n * (H * sz + stackGap);

        for (int x = 0; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++) {
                    float px = (x - (W-1)*0.5f) * sz;
                    float py = (y - (D-1)*0.5f) * sz;
                    float pz = baseZ + stackZ + z * sz;
                    grid[x][y][z] = new Rigid(s, {sz, sz, sz}, 1.0f, 0.5f, {px, py, pz});
                }

        for (int x = 1; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++)
                    new Joint(s, grid[x-1][y][z], grid[x][y][z], {half,0,0}, {-half,0,0}, Klin, Kang);
        for (int x = 0; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++)
                    new Joint(s, grid[x][y-1][z], grid[x][y][z], {0,half,0}, {0,-half,0}, Klin, Kang);
        for (int x = 0; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++)
                    new Joint(s, grid[x][y][z-1], grid[x][y][z], {0,0,half}, {0,0,-half}, Klin, Kang);

        for (int x = 1; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++) {
                    new IgnoreCollision(s, grid[x-1][y][z-1], grid[x][y][z]);
                    new IgnoreCollision(s, grid[x][y][z-1], grid[x-1][y][z]);
                }
        for (int x = 0; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 1; z < H; z++) {
                    new IgnoreCollision(s, grid[x][y-1][z-1], grid[x][y][z]);
                    new IgnoreCollision(s, grid[x][y][z-1], grid[x][y-1][z]);
                }
        for (int x = 1; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++) {
                    new IgnoreCollision(s, grid[x-1][y-1][z], grid[x][y][z]);
                    new IgnoreCollision(s, grid[x][y-1][z], grid[x-1][y][z]);
                }
    }
}

void setupBridge(Solver* s) {
    const int N = 40;
    const float plankLen = 1.0f, plankW = 4.0f, plankH = 0.5f;
    const float halfL = plankLen * 0.5f, halfW = plankW * 0.5f;

    s->clear();
    s->set_ground_plane(0.5f, 0.5f);

    Rigid* prev = nullptr;
    for (int i = 0; i < N; i++) {
        Rigid* curr = new Rigid(s, {plankLen, plankW, plankH},
                                (i == 0 || i == N-1) ? 0.0f : 1.0f, 0.5f,
                                {static_cast<float>(i) - N/2.0f, 0.0f, 10.0f});
        if (prev) {
            new Joint(s, prev, curr, {halfL,  halfW, 0}, {-halfL,  halfW, 0}, INFINITY, 0.0f);
            new Joint(s, prev, curr, {halfL, -halfW, 0}, {-halfL, -halfW, 0}, INFINITY, 0.0f);
        }
        prev = curr;
    }

    for (int x = 0; x < N/4; x++)
        for (int y = 0; y < N/8; y++)
            new Rigid(s, {1, 1, 1}, 1.0f, 0.5f,
                      {static_cast<float>(x) - N/8.0f, 0.0f, static_cast<float>(y) + 12.0f});
}

void setupBreakable(Solver* s) {
    const int N = 10, M = 5;
    const float breakForce = 90.0f;

    s->clear();
    s->set_ground_plane(0.5f, 0.5f);

    Rigid* prev = nullptr;
    for (int i = 0; i <= N; i++) {
        Rigid* curr = new Rigid(s, {1, 1, 0.5f}, 1.0f, 0.5f,
                                {static_cast<float>(i) - N/2.0f, 0.0f, 6.0f});
        if (prev)
            new Joint(s, prev, curr, {0.5f,0,0}, {-0.5f,0,0}, INFINITY, INFINITY, breakForce);
        prev = curr;
    }

    new Rigid(s, {1, 1, 5}, 0.0f, 0.5f, {-N/2.0f, 0, 2.5f});
    new Rigid(s, {1, 1, 5}, 0.0f, 0.5f, { N/2.0f, 0, 2.5f});

    for (int i = 0; i < M; i++)
        new Rigid(s, {2, 1, 1}, 1.0f, 0.5f, {0, 0, i * 2.0f + 8.0f});
}

void setupSpring(Solver* s) {
    s->clear();
    s->set_ground_plane(0.5f, 0.5f);
    Rigid* anchor = new Rigid(s, {1, 1, 1}, 0.0f, 0.5f, {0, 0, 14.0f});
    Rigid* block  = new Rigid(s, {2, 2, 2}, 1.0f, 0.5f, {0, 0, 8.0f});
    new Spring(s, anchor, block, {0,0,0}, {0,0,0}, 100.0f, 4.0f);
}

}  // anonymous namespace

extern "C" void chysx_register_avbd_scenes() {
    register_scene("AVBD: Pyramid",    []() -> Scene* { return new AVBDScene("AVBD: Pyramid",    setupPyramid); });
    register_scene("AVBD: Pyramid 2",  []() -> Scene* { return new AVBDScene("AVBD: Pyramid 2",  setupPyramid2); });
    register_scene("AVBD: Rope",       []() -> Scene* { return new AVBDScene("AVBD: Rope",       setupRope); });
    register_scene("AVBD: Heavy Rope", []() -> Scene* { return new AVBDScene("AVBD: Heavy Rope", setupHeavyRope); });
    register_scene("AVBD: Stack",      []() -> Scene* { return new AVBDScene("AVBD: Stack",      setupStack); });
    register_scene("AVBD: Soft Body",  []() -> Scene* { return new AVBDScene("AVBD: Soft Body",  setupSoftBody); });
    register_scene("AVBD: Bridge",     []() -> Scene* { return new AVBDScene("AVBD: Bridge",     setupBridge); });
    register_scene("AVBD: Breakable",  []() -> Scene* { return new AVBDScene("AVBD: Breakable",  setupBreakable); });
    register_scene("AVBD: Spring",     []() -> Scene* { return new AVBDScene("AVBD: Spring",     setupSpring); });
}
