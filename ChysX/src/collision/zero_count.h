// SPDX-License-Identifier: Apache-2.0
//
// Device-side zero counter used as a fallback count pointer for
// unconditional kernel launches.  When a collision subsystem is
// inactive or default-constructed its `count_dev` may be nullptr;
// substituting `zero_count_ptr()` lets the kernel read *count_ptr
// == 0 and have every thread early-exit, keeping the kernel launch
// sequence identical across frames for CUDA Graph capture.

#pragma once

#include <cuda_runtime.h>

namespace chysx {
namespace collision {

inline const int* zero_count_ptr() {
    static const int* ptr = [] {
        int* p = nullptr;
        cudaMalloc(&p, sizeof(int));
        cudaMemset(p, 0, sizeof(int));
        return p;
    }();
    return ptr;
}

}  // namespace collision
}  // namespace chysx
