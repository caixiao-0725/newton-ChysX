// SPDX-License-Identifier: Apache-2.0
//
// Simple Wavefront OBJ triangle mesh reader / writer.

#pragma once

#include <string>
#include <vector>

namespace chysx {
namespace io {

struct ObjMesh {
    std::vector<float> positions;  // flat xyz, length = n_points * 3
    std::vector<int>   triangles;  // flat vertex indices (0-based), length = n_tris * 3
};

// Load a simple OBJ (only "v" and "f" lines).  Supports both plain
// integer indices and complex "v/vt/vn" face entries.  All faces must
// be triangles.  Returns false on failure.
bool load_obj(const std::string& filepath, ObjMesh& mesh);

// Write a triangle mesh to OBJ.  `positions` is flat xyz, length =
// n_points * 3.  `triangles` is flat 0-based indices, length =
// n_tris * 3.  Returns false on failure.
bool save_obj(const std::string& filepath,
              const float* positions, int n_points,
              const int* triangles, int n_tris);

}  // namespace io
}  // namespace chysx
