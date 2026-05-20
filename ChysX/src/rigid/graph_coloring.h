// SPDX-License-Identifier: Apache-2.0
//
// CPU-side greedy graph coloring for rigid-body GS iteration order.

#pragma once

#include <vector>

namespace chysx {
namespace rigid {

// Greedy graph coloring from a joint edge list.
// Returns color_groups: groups[c] = vector of body indices in color c.
// Bodies not touched by any joint get their own single-body group.
std::vector<std::vector<int>> color_rigid_bodies(
    int body_count,
    const std::vector<std::pair<int, int>>& edges);

}  // namespace rigid
}  // namespace chysx
