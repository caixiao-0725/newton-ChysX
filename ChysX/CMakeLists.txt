cmake_minimum_required(VERSION 3.18)
project(chysx LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ====================================================================
# chysx_core — static library with all physics / solver / collision code
# ====================================================================

add_library(chysx_core STATIC
    src/memory/cuda_array.cu
    src/sparse/block_csr.cu
    src/solver/pcg_solver.cu

    src/collision/bvh/lbvh.cu
    src/collision/bvh/quant_bvh.cu
    src/collision/mesh_topology.cu
    src/collision/self_collision.cu
    src/collision/contact_spmv.cu
    src/collision/static_contact.cu
    src/collision/untangle.cu
    src/collision/mesh_contact.cu
    src/collision/sdf/sdf_volume.cu
    src/collision/sdf/sdf_contact.cu

    src/constraint/cloth/pin_constraint.cu
    src/constraint/cloth/spring_constraint.cu
    src/constraint/cloth/triangle_stretch_constraint.cu
    src/constraint/cloth/bending_constraint.cu
    src/constraint/collision/self_collision_constraint.cu
    src/constraint/collision/untangle_constraint.cu

    src/rigid/newton_avbd/avbd_kernels.cu
    src/rigid/newton_avbd/contact_matching.cu
    src/rigid/newton_avbd/graph_coloring.cpp
    src/rigid/newton_avbd/narrow_phase.cu
    src/rigid/newton_avbd/rigid_broadphase.cu
    src/rigid/newton_avbd/rigid_simulator.cu

    src/rigid/avbd_cpu/avbd_solver.cpp
    src/rigid/avbd_cpu/avbd_rigid.cpp
    src/rigid/avbd_cpu/avbd_force.cpp
    src/rigid/avbd_cpu/avbd_joint.cpp
    src/rigid/avbd_cpu/avbd_spring.cpp
    src/rigid/avbd_cpu/avbd_manifold.cpp
    src/rigid/avbd_cpu/avbd_collide.cpp
    src/rigid/avbd_cpu/avbd_broadphase_gpu.cu
    src/rigid/avbd_cpu/avbd_narrowphase_gpu.cu
    src/rigid/avbd_cpu/avbd_graph_coloring.cu
    src/rigid/avbd_cpu/avbd_gpu_solver.cu

    src/cloth/cloth_simulator.cu
)

target_include_directories(chysx_core PUBLIC src)
target_link_libraries(chysx_core PUBLIC CUDA::cudart)

option(AVBD_VALIDATE_BROADPHASE "Compare CPU and GPU broadphase results" OFF)
if(AVBD_VALIDATE_BROADPHASE)
    target_compile_definitions(chysx_core PUBLIC AVBD_VALIDATE_BROADPHASE)
endif()

option(AVBD_VALIDATE_NARROWPHASE "Compare CPU and GPU narrowphase results" OFF)
if(AVBD_VALIDATE_NARROWPHASE)
    target_compile_definitions(chysx_core PUBLIC AVBD_VALIDATE_NARROWPHASE)
endif()

set_target_properties(chysx_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    target_compile_options(chysx_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# ====================================================================
# chysx_scenes — static library with scene definitions + scene registry
# ====================================================================

add_library(chysx_scenes STATIC
    src/render/scene.cpp
    src/io/obj_io.cpp
    src/io/bgeo_writer.cpp
    scenes/cloth_freefall.cu
    scenes/avbd_scenes.cpp
    src/rigid/avbd_cpu/avbd_gpu_renderer.cu
)

target_include_directories(chysx_scenes PUBLIC src src/io)
target_link_libraries(chysx_scenes PUBLIC chysx_core)

set_target_properties(chysx_scenes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    target_compile_options(chysx_scenes PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# ====================================================================
# _chysx_native — pybind11 Python extension module (optional)
# ====================================================================

option(CHYSX_BUILD_PYTHON "Build the pybind11 Python module" OFF)

if(CHYSX_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_chysx_native MODULE src/bindings.cpp)
    target_link_libraries(_chysx_native PRIVATE chysx_core)

    if(MSVC)
        target_compile_options(_chysx_native PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endif()

    install(TARGETS _chysx_native LIBRARY DESTINATION chysx)
endif()

# ====================================================================
# chysx_scene — batch export executable (no GUI)
# ====================================================================

option(CHYSX_BUILD_SCENES "Build batch export executable" ON)

if(CHYSX_BUILD_SCENES)
    add_compile_definitions(ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

    add_executable(chysx_scene
        src/render/batch_export.cu
    )

    target_link_libraries(chysx_scene PRIVATE chysx_scenes)

    set_target_properties(chysx_scene PROPERTIES
        LINKER_LANGUAGE CUDA
    )

    if(MSVC)
        target_compile_options(chysx_scene PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endif()
endif()

# ====================================================================
# chysx_viewer — real-time OpenGL + ImGui viewer
# ====================================================================

option(CHYSX_BUILD_VIEWER "Build the real-time viewer" ON)

if(CHYSX_BUILD_VIEWER)
    # SDL2
    add_subdirectory(external/SDL EXCLUDE_FROM_ALL)

    # ImGui as a static library
    file(GLOB IMGUI_SRC
        external/imgui/*.cpp
        external/imgui/backends/imgui_impl_sdl2.cpp
        external/imgui/backends/imgui_impl_opengl3.cpp
    )
    add_library(imgui STATIC ${IMGUI_SRC})
    target_include_directories(imgui PUBLIC
        external/imgui
        external/imgui/backends
        external/SDL/include
    )
    target_link_libraries(imgui PUBLIC SDL2::SDL2 SDL2::SDL2main)

    # avbd_scenes.cpp uses GL and SDL2/ImGui headers for custom rendering.
    target_link_libraries(chysx_scenes PRIVATE imgui)
    find_package(OpenGL REQUIRED)
    target_link_libraries(chysx_scenes PRIVATE OpenGL::GL)

    # Viewer executable (.cu entry so nvcc performs the device link step)
    add_executable(chysx_viewer
        src/render/main.cu
        src/render/viewer.cpp
    )

    target_include_directories(chysx_viewer PRIVATE src)

    target_link_libraries(chysx_viewer PRIVATE chysx_scenes imgui)

    find_package(OpenGL REQUIRED)
    target_link_libraries(chysx_viewer PRIVATE OpenGL::GL)

    set_target_properties(chysx_viewer PROPERTIES
        LINKER_LANGUAGE CUDA
    )

    if(MSVC)
        target_compile_options(chysx_viewer PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endif()

    # Copy SDL2.dll next to the exe on Windows
    if(WIN32)
        add_custom_command(TARGET chysx_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:SDL2::SDL2>
                $<TARGET_FILE_DIR:chysx_viewer>
        )
    endif()
endif()
