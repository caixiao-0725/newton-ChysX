// SPDX-License-Identifier: Apache-2.0

#include "bgeo_writer.h"

#include <cstring>
#include <iostream>

namespace chysx {
namespace io {

BgeoWriter::BgeoWriter(const std::string& filepath)
    : out_(filepath, std::ios::binary) {
    if (out_.good()) {
        writeMagic();
    }
}

void BgeoWriter::writeMagic() {
    writeId(JID_MAGIC);
    out_.write(reinterpret_cast<const char*>(&BINARY_MAGIC), 4);
}

void BgeoWriter::writeId(uint8_t id) {
    out_.write(reinterpret_cast<const char*>(&id), 1);
}

void BgeoWriter::writeLength(int64_t length) {
    if (length < 0xF1) {
        uint8_t b = static_cast<uint8_t>(length);
        out_.write(reinterpret_cast<const char*>(&b), 1);
    } else if (length < 0xFFFF) {
        uint8_t tag = 0xF2;
        uint16_t val = static_cast<uint16_t>(length);
        out_.write(reinterpret_cast<const char*>(&tag), 1);
        out_.write(reinterpret_cast<const char*>(&val), 2);
    } else if (length < 0xFFFFFFFF) {
        uint8_t tag = 0xF4;
        uint32_t val = static_cast<uint32_t>(length);
        out_.write(reinterpret_cast<const char*>(&tag), 1);
        out_.write(reinterpret_cast<const char*>(&val), 4);
    } else {
        uint8_t tag = 0xF8;
        out_.write(reinterpret_cast<const char*>(&tag), 1);
        out_.write(reinterpret_cast<const char*>(&length), 8);
    }
}

void BgeoWriter::beginArray() { writeId(JID_ARRAY_BEGIN); }
void BgeoWriter::endArray()   { writeId(JID_ARRAY_END); }
void BgeoWriter::beginMap()   { writeId(JID_MAP_BEGIN); }
void BgeoWriter::endMap()     { writeId(JID_MAP_END); }

void BgeoWriter::writeString(const std::string& s) {
    writeId(JID_STRING);
    writeLength(static_cast<int64_t>(s.size()));
    out_.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void BgeoWriter::writeInt32(int32_t v) {
    writeId(JID_INT32);
    out_.write(reinterpret_cast<const char*>(&v), 4);
}

void BgeoWriter::writeUniformArrayFloat(const float* data, int64_t count) {
    writeId(JID_UNIFORM_ARRAY);
    int8_t type = static_cast<int8_t>(JID_REAL32);
    out_.write(reinterpret_cast<const char*>(&type), 1);
    writeLength(count);
    out_.write(reinterpret_cast<const char*>(data),
               count * static_cast<int64_t>(sizeof(float)));
}

void BgeoWriter::writeUniformArrayInt(const int32_t* data, int64_t count) {
    writeId(JID_UNIFORM_ARRAY);
    int8_t type = static_cast<int8_t>(JID_INT32);
    out_.write(reinterpret_cast<const char*>(&type), 1);
    writeLength(count);
    out_.write(reinterpret_cast<const char*>(data),
               count * static_cast<int64_t>(sizeof(int32_t)));
}

void BgeoWriter::writeAttribute(
    const std::string& name,
    const std::string& storage,
    int tuple_size, int n_elements,
    const void* data, bool is_position) {

    beginArray();

    // definition
    beginArray();
    writeString("scope");   writeString("public");
    writeString("type");    writeString("numeric");
    writeString("name");    writeString(name);
    writeString("options");
    beginMap();
    if (is_position) {
        writeString("type");
        beginMap();
        writeString("type");   writeString("string");
        writeString("value");  writeString("point");
        endMap();
    }
    endMap();
    endArray();

    // data
    beginArray();
    writeString("size");     writeInt32(tuple_size);
    writeString("storage");  writeString(storage);
    writeString("values");
    beginArray();
    writeString("size");     writeInt32(tuple_size);
    writeString("storage");  writeString(storage);
    writeString("pagesize"); writeInt32(1024);
    writeString("rawpagedata");
    if (storage == "fpreal32") {
        writeUniformArrayFloat(
            static_cast<const float*>(data),
            static_cast<int64_t>(n_elements) * tuple_size);
    } else if (storage == "int32") {
        writeUniformArrayInt(
            static_cast<const int32_t*>(data),
            static_cast<int64_t>(n_elements) * tuple_size);
    }
    endArray();
    endArray();

    endArray();
}

void BgeoWriter::writeGeo(
    const float* positions, int n_points,
    const int* triangles, int n_tris,
    const float* point_colors,
    const float* face_colors) {

    int n_vertices = n_tris * 3;

    beginArray();

    // counts
    writeString("pointcount");    writeInt32(n_points);
    writeString("vertexcount");   writeInt32(n_vertices);
    writeString("primitivecount"); writeInt32(n_tris);

    // topology
    writeString("topology");
    beginArray();
    writeString("pointref");
    beginArray();
    writeString("indices");
    writeUniformArrayInt(triangles, n_vertices);
    endArray();
    endArray();

    // attributes
    writeString("attributes");
    beginArray();

    // point attributes
    writeString("pointattributes");
    beginArray();
    writeAttribute("P", "fpreal32", 3, n_points, positions, true);
    if (point_colors) {
        writeAttribute("Cd", "fpreal32", 3, n_points, point_colors, false);
    }
    endArray();

    // primitive attributes
    if (face_colors) {
        writeString("primitiveattributes");
        beginArray();
        writeAttribute("Cd", "fpreal32", 3, n_tris, face_colors, false);
        endArray();
    }

    endArray();  // attributes

    // primitives
    writeString("primitives");
    beginArray();
    if (n_tris > 0) {
        beginArray();
        beginArray();
        writeString("type"); writeString("p_r");
        endArray();
        beginArray();
        writeString("s_v"); writeInt32(0);
        writeString("n_p"); writeInt32(n_tris);
        writeString("n_v");
        std::vector<int32_t> n_v(static_cast<size_t>(n_tris), 3);
        writeUniformArrayInt(n_v.data(), n_tris);
        endArray();
        endArray();
    }
    endArray();  // primitives

    endArray();  // root
}

bool BgeoWriter::write(
    const std::string& filepath,
    const float* positions, int n_points,
    const int* triangles, int n_tris,
    const float* point_colors,
    const float* face_colors) {

    BgeoWriter w(filepath);
    if (!w.good()) {
        std::cerr << "BgeoWriter: cannot open " << filepath << std::endl;
        return false;
    }

    w.writeGeo(positions, n_points, triangles, n_tris,
               point_colors, face_colors);
    return w.out_.good();
}

bool BgeoWriter::write_multi(
    const std::string& filepath,
    const std::vector<BgeoMeshPiece>& pieces) {

    // Compute totals
    int total_points = 0;
    int total_tris = 0;
    for (const auto& p : pieces) {
        total_points += p.n_points;
        total_tris += p.n_tris;
    }

    // Merge positions
    std::vector<float> merged_pos;
    merged_pos.reserve(static_cast<size_t>(total_points) * 3);
    for (const auto& p : pieces) {
        merged_pos.insert(merged_pos.end(),
                          p.positions, p.positions + p.n_points * 3);
    }

    // Merge triangles with vertex index offset
    std::vector<int> merged_tris;
    merged_tris.reserve(static_cast<size_t>(total_tris) * 3);
    int vertex_offset = 0;
    for (const auto& p : pieces) {
        for (int i = 0; i < p.n_tris * 3; ++i) {
            merged_tris.push_back(p.triangles[i] + vertex_offset);
        }
        vertex_offset += p.n_points;
    }

    // Build per-point colors: each piece gets its uniform color
    std::vector<float> point_colors;
    point_colors.reserve(static_cast<size_t>(total_points) * 3);
    for (const auto& p : pieces) {
        for (int i = 0; i < p.n_points; ++i) {
            point_colors.push_back(p.color_r);
            point_colors.push_back(p.color_g);
            point_colors.push_back(p.color_b);
        }
    }

    return write(filepath,
                 merged_pos.data(), total_points,
                 merged_tris.data(), total_tris,
                 point_colors.data(), nullptr);
}

void generate_box_mesh(float cx, float cy, float cz,
                       float hx, float hy, float hz,
                       std::vector<float>& out_positions,
                       std::vector<int>& out_triangles) {
    out_positions.resize(8 * 3);
    // 8 corners: iterate (sign_x, sign_y, sign_z)
    for (int i = 0; i < 8; ++i) {
        float sx = (i & 1) ? hx : -hx;
        float sy = (i & 2) ? hy : -hy;
        float sz = (i & 4) ? hz : -hz;
        out_positions[i * 3 + 0] = cx + sx;
        out_positions[i * 3 + 1] = cy + sy;
        out_positions[i * 3 + 2] = cz + sz;
    }
    // 6 faces, 2 triangles each = 12 triangles
    // Vertex ordering:
    //   0=(---) 1=(+--) 2=(-+-) 3=(++-) 4=(--+) 5=(+-+) 6=(-++) 7=(+++)
    static constexpr int faces[12][3] = {
        {0, 2, 3}, {0, 3, 1},  // -Z face
        {4, 5, 7}, {4, 7, 6},  // +Z face
        {0, 1, 5}, {0, 5, 4},  // -Y face
        {2, 6, 7}, {2, 7, 3},  // +Y face
        {0, 4, 6}, {0, 6, 2},  // -X face
        {1, 3, 7}, {1, 7, 5},  // +X face
    };
    out_triangles.resize(12 * 3);
    for (int i = 0; i < 12; ++i) {
        out_triangles[i * 3 + 0] = faces[i][0];
        out_triangles[i * 3 + 1] = faces[i][1];
        out_triangles[i * 3 + 2] = faces[i][2];
    }
}

}  // namespace io
}  // namespace chysx
