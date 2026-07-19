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
set(rapids-cmake-version 26.02.1)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

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
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

# ================================
# CPU baseline lemmatizer benchmark
# ================================
find_package(OpenMP REQUIRED)
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
        OpenMP::OpenMP_CXX
)

# ================================
# GPU raw benchmark (one word/line input, metrics only)
# ================================
add_executable(lemmatize_gpu_raw
        src/cuda/bench_gpu_raw.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_raw PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_raw PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_raw PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# GPU optimized benchmark (mmap, pinned memory, async streams, event sync)
# ================================
add_executable(lemmatize_gpu_opt
        src/cuda/bench_gpu_opt.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_opt PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_opt PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
add_executable(lemmatize_gpu_opt1
        src/cuda/bench_gpu_opt1.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_opt1 PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_opt1 PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_opt1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
# Unified GPU benchmark driver
# ================================
add_executable(lemmatize_gpu_driver
        src/cuda/bench_gpu_driver.cu
        src/cuda/lemmatizer_kernel.cu
        src/cuda/trie.cpp
        src/cuda/icu_lowercase.cpp
)

target_include_directories(lemmatize_gpu_driver PRIVATE
        ${ICU_INCLUDE_DIRS}
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_driver PRIVATE
        ${ICU_LIBRARIES}
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
        OpenMP::OpenMP_CXX
)

set_target_properties(lemmatize_gpu_driver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


# ================================
add_executable(lemmatize_gpu_opt2
        src/cuda/bench_gpu_opt2.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_opt2 PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_opt2 PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_opt2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
add_executable(lemmatize_gpu_opt1_utf8
        src/cuda/bench_gpu_opt1_utf8.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_opt1_utf8 PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_opt1_utf8 PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_opt1_utf8 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
add_executable(lemmatize_gpu_opt1_utf8_loop
        src/cuda/bench_gpu_opt1_utf8_loop.cu
        src/cuda/lemmatizer_kernel.cu
)

target_include_directories(lemmatize_gpu_opt1_utf8_loop PRIVATE
        include
        src/cuda
)

target_link_libraries(lemmatize_gpu_opt1_utf8_loop PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)

set_target_properties(lemmatize_gpu_opt1_utf8_loop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
add_executable(micro src/cuda/micro.cu)
set_target_properties(micro PROPERTIES CUDA_ARCHITECTURES 89)

add_executable(micro_l1 src/cuda/micro_single.cu)
set_target_properties(micro_l1 PROPERTIES CUDA_ARCHITECTURES 89)
target_compile_options(micro_l1 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>)
target_compile_definitions(micro_l1 PRIVATE ARRAY_KB=64)

add_executable(micro_l2 src/cuda/micro_single.cu)
set_target_properties(micro_l2 PROPERTIES CUDA_ARCHITECTURES 89)
target_compile_options(micro_l2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg>)
target_compile_definitions(micro_l2 PRIVATE ARRAY_MB=32)

add_executable(micro_dram src/cuda/micro_single.cu)
set_target_properties(micro_dram PROPERTIES CUDA_ARCHITECTURES 89)
target_compile_options(micro_dram PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg>)
target_compile_definitions(micro_dram PRIVATE ARRAY_MB=256)

# ================================
add_executable(lemmatize_gpu_opt3
        src/cuda/bench_gpu_opt3.cu
        src/cuda/lemmatizer_kernel.cu
)
target_include_directories(lemmatize_gpu_opt3 PRIVATE
        include
        src/cuda
)
target_link_libraries(lemmatize_gpu_opt3 PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)
set_target_properties(lemmatize_gpu_opt3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ================================
add_executable(lemmatize_gpu_cudf
        src/cuda/bench_gpu_cudf.cu
        src/cuda/lemmatizer_kernel.cu
)
target_include_directories(lemmatize_gpu_cudf PRIVATE include src/cuda)
target_link_libraries(lemmatize_gpu_cudf PRIVATE
        cudf::cudf
        /usr/lib/x86_64-linux-gnu/libstdc++.so.6
)
set_target_properties(lemmatize_gpu_cudf PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)