cmake_minimum_required(VERSION 3.28)
# ================================
# CUDA Setup
# ================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Optional: make sure CUDA path is correct
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
set(rapids-cmake-version 25.08)

project(cuda_lemmatizer LANGUAGES CXX CUDA)
add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# ================================
# cuDF, RMM
# ================================
find_package(cudf REQUIRED)
find_package(rmm REQUIRED)

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
        cudf::cudf
        lemmatizer
)

# ================================
# CPU baseline lemmatizer benchmark
# ================================
add_executable(lemmatize_cpu
        src/cuda/bench_cpu.cpp
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_cpu PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_cpu PRIVATE
        ${ICU_LIBRARIES}
)

# ================================
# GPU lemmatizer benchmark
# ================================
add_executable(lemmatize_gpu
        src/cuda/bench_gpu.cu
        src/cuda/lemmatizer_kernel.cu
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_gpu PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# GPU stride lemmatizer benchmark
# ================================
add_executable(lemmatize_gpu_stride
        src/cuda/bench_gpu_stride.cu
        src/cuda/lemmatizer_kernel.cu
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_gpu_stride PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_stride PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_stride PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# GPU bsearch lemmatizer benchmark
# ================================
add_executable(lemmatize_gpu_bsearch
        src/cuda/bench_gpu_bsearch.cu
        src/cuda/lemmatizer_kernel.cu
        src/cuda/trie.cpp
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_gpu_bsearch PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_bsearch PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_bsearch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# GPU loop (max-throughput) benchmark
# ================================
add_executable(lemmatize_gpu_loop
        src/cuda/bench_gpu_loop.cu
        src/cuda/lemmatizer_kernel.cu
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_gpu_loop PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_loop PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_loop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
