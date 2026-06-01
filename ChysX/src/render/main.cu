// SPDX-License-Identifier: Apache-2.0
//
// ChysX Viewer — entry point.

#include "viewer.h"

int main(int /*argc*/, char* /*argv*/[]) {
    chysx::render::ViewerConfig cfg;
    cfg.width = 1280;
    cfg.height = 720;
    cfg.title = "ChysX Viewer";
    cfg.dt = 1.0f / 100.0f;

    chysx::render::Viewer viewer(cfg);
    viewer.run();
    return 0;
}
