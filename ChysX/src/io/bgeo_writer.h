// SPDX-License-Identifier: Apache-2.0
//
// Minimal Houdini .bgeo (binary JSON) triangle-mesh writer.
//
// Implements only the subset needed to export triangle meshes with
// per-point float32 attributes (P, Cd) and per-primitive float32
// attributes.  Follows the same binary encoding as cuda-cloth's
// src/IO/json.cpp and HouGeo.cpp.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace chysx {
namespace io {

// A single mesh piece for multi-object export.
struct BgeoMeshPiece {
    const float* positions;  // flat xyz, length = n_points * 3
    int n_points;
    const int* triangles;    // flat 0-based indices, length = n_tris * 3
    int n_tris;
    float color_r, color_g, color_b;  // uniform color for this piece
};

class BgeoWriter {
public:
    // Write a single triangle mesh to a .bgeo file.
    //
    // positions:   per-vertex xyz, length = n_points * 3
    // triangles:   per-face vertex indices (0-based), length = n_tris * 3
    // point_colors: optional per-vertex rgb, length = n_points * 3 (may be null)
    // face_colors:  optional per-face rgb, length = n_tris * 3 (may be null)
    static bool write(const std::string& filepath,
                      const float* positions, int n_points,
                      const int* triangles, int n_tris,
                      const float* point_colors = nullptr,
                      const float* face_colors = nullptr);

    // Write multiple mesh pieces merged into a single .bgeo file.
    // Each piece gets its own uniform color assigned to per-point Cd.
    // Vertex indices are automatically offset.
    static bool write_multi(const std::string& filepath,
                            const std::vector<BgeoMeshPiece>& pieces);

private:
    std::ofstream out_;

    static constexpr uint8_t JID_ARRAY_BEGIN   = 0x5B;
    static constexpr uint8_t JID_ARRAY_END     = 0x5D;
    static constexpr uint8_t JID_MAP_BEGIN     = 0x7B;
    static constexpr uint8_t JID_MAP_END       = 0x7D;
    static constexpr uint8_t JID_STRING        = 0x27;
    static constexpr uint8_t JID_INT32         = 0x13;
    static constexpr uint8_t JID_REAL32        = 0x19;
    static constexpr uint8_t JID_MAGIC         = 0x7F;
    static constexpr uint8_t JID_UNIFORM_ARRAY = 0x40;
    static constexpr uint32_t BINARY_MAGIC     = 0x624A534E;

    explicit BgeoWriter(const std::string& filepath);
    bool good() const { return out_.good(); }

    void writeMagic();
    void writeId(uint8_t id);
    void writeLength(int64_t length);

    void beginArray();
    void endArray();
    void beginMap();
    void endMap();
    void writeString(const std::string& s);
    void writeInt32(int32_t v);
    void writeUniformArrayFloat(const float* data, int64_t count);
    void writeUniformArrayInt(const int32_t* data, int64_t count);

    void writeAttribute(const std::string& name,
                        const std::string& storage,
                        int tuple_size, int n_elements,
                        const void* data, bool is_position);

    // Internal: write the full bgeo structure given merged flat arrays
    void writeGeo(const float* positions, int n_points,
                  const int* triangles, int n_tris,
                  const float* point_colors,
                  const float* face_colors);
};

// Generate an axis-aligned box mesh (12 triangles, 8 vertices).
// center: xyz center of the box
// half_extents: half-size in each axis
// out_positions: resized to 8*3, out_triangles: resized to 12*3
void generate_box_mesh(float cx, float cy, float cz,
                       float hx, float hy, float hz,
                       std::vector<float>& out_positions,
                       std::vector<int>& out_triangles);

}  // namespace io
}  // namespace chysx
