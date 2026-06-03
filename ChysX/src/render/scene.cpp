// SPDX-License-Identifier: Apache-2.0

#include "scene.h"

extern "C" void chysx_register_cloth_scenes();
extern "C" void chysx_register_avbd_scenes();

namespace chysx {
namespace render {

static std::vector<SceneEntry>& registry_impl() {
    static std::vector<SceneEntry> r;
    return r;
}

void register_scene(const char* name, Scene* (*factory)()) {
    registry_impl().push_back({name, factory});
}

const std::vector<SceneEntry>& scene_registry() {
    return registry_impl();
}

void register_all_scenes() {
    chysx_register_cloth_scenes();
    chysx_register_avbd_scenes();
}

}  // namespace render
}  // namespace chysx
