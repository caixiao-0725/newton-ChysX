// SPDX-License-Identifier: Apache-2.0

#include "obj_io.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace chysx {
namespace io {

bool load_obj(const std::string& filepath, ObjMesh& mesh) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        std::cerr << "obj_io: cannot open " << filepath << std::endl;
        return false;
    }

    mesh.positions.clear();
    mesh.triangles.clear();

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            mesh.positions.push_back(x);
            mesh.positions.push_back(y);
            mesh.positions.push_back(z);
        } else if (token == "f") {
            // Handle both "f 1 2 3" and "f 1/2/3 4/5/6 7/8/9"
            std::string vert;
            int indices[3];
            int count = 0;
            while (iss >> vert && count < 3) {
                std::istringstream vss(vert);
                std::string idx_str;
                std::getline(vss, idx_str, '/');
                indices[count++] = std::stoi(idx_str) - 1;  // 1-based → 0-based
            }
            if (count == 3) {
                mesh.triangles.push_back(indices[0]);
                mesh.triangles.push_back(indices[1]);
                mesh.triangles.push_back(indices[2]);
            }
        }
    }

    std::cout << "[obj_io] loaded " << filepath
              << "  verts=" << mesh.positions.size() / 3
              << "  tris=" << mesh.triangles.size() / 3 << std::endl;
    return true;
}

bool save_obj(const std::string& filepath,
              const float* positions, int n_points,
              const int* triangles, int n_tris) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "obj_io: cannot open " << filepath << " for writing"
                  << std::endl;
        return false;
    }

    out << "# ChysX OBJ export  verts=" << n_points
        << " tris=" << n_tris << "\n";

    for (int i = 0; i < n_points; ++i) {
        out << "v " << positions[i * 3]
            << " " << positions[i * 3 + 1]
            << " " << positions[i * 3 + 2] << "\n";
    }

    for (int i = 0; i < n_tris; ++i) {
        out << "f " << (triangles[i * 3] + 1)
            << " " << (triangles[i * 3 + 1] + 1)
            << " " << (triangles[i * 3 + 2] + 1) << "\n";
    }

    std::cout << "[obj_io] saved " << filepath << std::endl;
    return true;
}

}  // namespace io
}  // namespace chysx
