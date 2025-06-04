cmake_minimum_required(VERSION 3.28)
# ================================
# CUDA Setup
# ================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Optional: make sure CUDA path is correct
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

project(cuda_lemmatizer LANGUAGES CXX CUDA)

# ================================
# RAPIDS CMake Init (cuDF, RMM)
# ================================
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-25.04/RAPIDS.cmake
        ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(cuda_lemmatizer)

rapids_cpm_init()
rapids_cpm_find(cudf 25.04.00
        CPM_ARGS
        GIT_REPOSITORY https://github.com/rapidsai/cudf.git
        GIT_TAG branch-25.04
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR cpp
        OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARKS OFF"
        "CUDF_USE_ARROW_STATIC ON"
        "JITIFY_USE_CACHE ON"
        "AUTO_DETECT_CUDA_ARCHITECTURES OFF"
        "CUDF_KVIKIO_REMOTE_IO OFF"
)

# ================================
# ICU (Unchanged)
# ================================
find_package(ICU REQUIRED COMPONENTS uc i18n)

# ================================
# JNI
# ================================
find_package(JNI REQUIRED)
include_directories(
        /usr/lib/jvm/java-21-openjdk-amd64/include
        /usr/lib/jvm/java-21-openjdk-amd64/include/linux
)
#include_directories(${JNI_INCLUDE_DIRS})

# ================================
# Lemmatizer Shared Library
# ================================
add_library(lemmatizer SHARED
        src/cuda/GpuLemmatizer.cpp
        src/cuda/lemmatizer.cpp
        src/cuda/lemmatizer_kernel.cu
        src/cuda/trie.cpp
)

target_include_directories(lemmatizer PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatizer PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
)

set_target_properties(lemmatizer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# Optional C++ Debug Main
# ================================
add_executable(cuda_exe
        src/cuda/main.cu
        src/cuda/icu_lowercase.cpp
)

target_include_directories(cuda_exe PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(cuda_exe PRIVATE
        ${ICU_LIBRARIES}
)
