cmake_minimum_required(VERSION 3.5)
project(tool_cloth_dynamics)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "set env var")

# Properly setting environment variables for Apple Clang
if (APPLE)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include" "/opt/homebrew/opt/libomp/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib" "/opt/homebrew/opt/libomp/lib")

    set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
    set(ENV{CXX} "/usr/local/opt/llvm/bin/clang-omp++")
    set(ENV{LDFLAGS} "-L/usr/local/opt/libomp/lib")
    set(ENV{CPPFLAGS} "-I/usr/local/opt/libomp/include")

    # Set OpenMP flags and libraries before find_package
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_C_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")

    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

    link_directories("/usr/local/Cellar/libomp/16.0.6/lib/")
endif()

# -O3 is a GCC/Clang spelling; MSVC rejects it.
if (NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_NAME tool_cloth_dynamics)
set(EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/external)
set(LIB dependencies/lib)

add_definitions(-DSOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\")

# OpenMP is a speedup, not a requirement: the 14 `#pragma omp` sites
# degrade to serial when the compiler ignores them, and the three
# omp_* API calls are guarded by CLOTH_HAVE_OPENMP. Requiring it here
# blocked the Windows build outright, since the mingw-llvm toolchain
# ships no libomp.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    add_compile_definitions(CLOTH_HAVE_OPENMP)
else()
    message(STATUS "OpenMP not found — building single-threaded")
endif()

if (APPLE AND OpenMP_CXX_FOUND)
    # Homebrew libomp needs the flags spelled out for Apple Clang.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_C_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")

    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

    link_directories("/usr/local/Cellar/libomp/16.0.6/lib/")
endif()

#if (APPLE)
#    add_compile_options(-DUSING_MACBOOK)
#endif ()

if (USE_DEBUG)
    add_compile_options(-DUSE_DEBUG)
endif ()


message("cxx Flags: " ${CMAKE_CXX_FLAGS})

source_group("Shaders" FILES src/gpu/shaders)
source_group("Assets" FILES src/assets)

add_subdirectory(${EXTERNAL_ROOT})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)








###############
get_directory_property(EXTERNAL_HEADER
        DIRECTORY ${EXTERNAL_ROOT}
        DEFINITION EXTERNAL_HEADER)

include_directories(dependencies/include)
include_directories(src/assets)
include_directories(src/gpu)
include_directories(external/LBFGSpp/include)
# external/fcl, external/fcl/external/libccd/src and dependencies/lib were
# listed here but do not exist in the tree, and no source includes an FCL or
# libccd header (mesh_collisions.md proposes FCL for cloth-vs-arbitrary-mesh
# collision; it was never integrated). Dropping the stale entries; adding the
# dependency later is a separate decision.


link_directories(${LIB})

# OmegaEngine cpp
file(GLOB HEADERS "src/code/*.h" "src/code/engine/*.h" "src/code/simulation/*.h" "src/code/supports/*.h" "src/code/optimization/*.h" "src/code/slang_solver/*.h")
file(GLOB SRCFILES "src/code/main.cpp" "src/code/*.h" "src/code/*.c" "src/code/engine/*.cpp" "src/code/engine/*.h" "src/code/simulation/*.cpp" "src/code/simulation/*.h" "src/code/supports/*.cpp" "src/code/supports/*.h" "src/code/optimization/*.cpp" "src/code/optimization/*.h" "src/code/slang_solver/*.h")
source_group("Sources" FILES ${SRCFILES})

# GPU AVBD backend. Same kernels either way -- the Lean codegen emits
# Slang, which becomes .metallib on Apple and .spv everywhere else --
# so only the dispatcher differs, and both implement the identical
# `cloth::AvbdSolver` API from AvbdSolver.h.
#
# When neither backend is available the AVBD call sites compile out
# (see CLOTH_HAVE_GPU_AVBD in Simulation.cpp) and DiffCloth's Eigen
# LLT + CG path is the only solver.
set(AVBD_BACKEND "none")
if (APPLE)
    set(SLANG_SOLVER_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/MetalCGSolver.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdSolver.mm")
    set_source_files_properties(${SLANG_SOLVER_SRC}
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    list(APPEND SRCFILES ${SLANG_SOLVER_SRC})
    set(AVBD_BACKEND "metal")
else()
    find_package(Vulkan)
    if (Vulkan_FOUND)
        list(APPEND SRCFILES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdSolverVk.cpp")
        set(AVBD_BACKEND "vulkan")
    endif()
endif()

if (NOT AVBD_BACKEND STREQUAL "none")
    # Pure-C++ AVBD backward shim (CHI-14 Brick C). Pulled into the
    # DiffCloth build so `Simulation::stepBackwardAvbd` (Brick D) can
    # call it from the optimisation path. Previously Apple-gated even
    # though it contains no Objective-C.
    list(APPEND SRCFILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdBackwardShim.cpp")
endif()
message(STATUS "AVBD GPU backend: ${AVBD_BACKEND}")

add_executable(${EXECUTABLE_NAME} ${SRCFILES}  )

if (NOT AVBD_BACKEND STREQUAL "none")
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE CLOTH_HAVE_GPU_AVBD)
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
endif()
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${EXTERNAL_HEADER})

if (APPLE)
    target_link_directories(${PROJECT_NAME} PRIVATE "/usr/local/Cellar/libomp/16.0.6/lib/")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(${EXECUTABLE_NAME}
        ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
elseif (AVBD_BACKEND STREQUAL "vulkan")
    target_link_libraries(${EXECUTABLE_NAME} Vulkan::Vulkan)
endif()
