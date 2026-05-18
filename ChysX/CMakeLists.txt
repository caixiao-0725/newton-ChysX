cmake_minimum_required(VERSION 3.18)
project(chysx_native LANGUAGES CXX CUDA)

# Emit compile_commands.json for clangd / IDEs.  Real builds go through
# scikit-build-core, but if you build with a plain `cmake -B build` this
# file ends up in build/ and can be symlinked / copied to the project
# root so clangd picks it up.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 for both host and device code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Need Python.h + the pybind11 CMake config (provided by the build env)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11 from whichever Python was used to invoke the build.
# scikit-build-core puts pybind11 into PYTHONPATH for us.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# CUDA runtime library (for cudaGetLastError, cudaStream_t, ...).
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_chysx_native MODULE
    # ---- memory / sparse / solver primitives ------------------------
    src/memory/cuda_array.cu
    src/sparse/block_csr.cu
    src/solver/pcg_solver.cu

    # ---- collision: BVH machinery (broadphase data structures) -----
    src/collision/bvh/lbvh.cu
    src/collision/bvh/quant_bvh.cu

    # ---- collision: contact detectors + penalty applicators --------
    src/collision/mesh_topology.cu
    src/collision/self_collision.cu
    src/collision/contact_spmv.cu
    src/collision/static_contact.cu
    src/collision/untangle.cu

    # ---- collision: animated SDF body penalty contact --------------
    src/collision/sdf/sdf_volume.cu
    src/collision/sdf/sdf_contact.cu

    # ---- constraints: cloth elasticity ------------------------------
    src/constraint/cloth/pin_constraint.cu
    src/constraint/cloth/spring_constraint.cu
    src/constraint/cloth/triangle_stretch_constraint.cu
    src/constraint/cloth/bending_constraint.cu

    # ---- constraints: collision-driven (SpMV sidecar) ---------------
    src/constraint/collision/self_collision_constraint.cu
    src/constraint/collision/untangle_constraint.cu

    # ---- top-level cloth simulator + python bindings ----------------
    src/cloth/cloth_simulator.cu
    src/bindings.cpp
)

target_include_directories(_chysx_native PRIVATE src)
target_link_libraries(_chysx_native PRIVATE CUDA::cudart)

set_target_properties(_chysx_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# MSVC defaults to the system ANSI codepage (CP936 on zh-CN locales)
# when parsing source files, which mis-decodes UTF-8 multi-byte
# sequences in comments (the trailing byte 0x96 of e.g. '|' U+2016
# eats the following newline and "deletes" the next line of code).
# Force UTF-8 source interpretation across host and device translation
# units so the codebase's mathematical comments stay readable.
if(MSVC)
    target_compile_options(_chysx_native PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# scikit-build-core copies installed targets into the wheel.
install(TARGETS _chysx_native LIBRARY DESTINATION chysx)
