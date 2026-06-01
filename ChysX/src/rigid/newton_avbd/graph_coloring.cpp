// SPDX-License-Identifier: Apache-2.0
//
// Greedy graph coloring for rigid body Gauss-Seidel iteration.

#include "graph_coloring.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace chysx {
namespace rigid {

std::vector<std::vector<int>> color_rigid_bodies(
    int body_count,
    const std::vector<std::pair<int, int>>& edges)
{
    if (body_count <= 0) return {};

    // Build adjacency list
    std::vector<std::vector<int>> adj(body_count);
    for (auto& [u, v] : edges) {
        if (u >= 0 && u < body_count && v >= 0 && v < body_count && u != v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    // Sort by degree descending (largest-first heuristic)
    std::vector<int> order(body_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });

    std::vector<int> color(body_count, -1);
    std::vector<bool> used;
    int max_color = 0;

    for (int bi : order) {
        // Find smallest color not used by any neighbor
        used.assign(max_color + 1, false);
        for (int nb : adj[bi]) {
            if (color[nb] >= 0 && color[nb] <= max_color) {
                used[color[nb]] = true;
            }
        }
        int c = 0;
        while (c < static_cast<int>(used.size()) && used[c]) ++c;
        color[bi] = c;
        if (c >= max_color) max_color = c + 1;
    }

    // Group bodies by color
    std::vector<std::vector<int>> groups(max_color);
    for (int i = 0; i < body_count; ++i) {
        groups[color[i]].push_back(i);
    }

    return groups;
}

}  // namespace rigid
}  // namespace chysx
