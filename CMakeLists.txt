cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
project(cuda_exe LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#list(APPEND CMAKE_PREFIX_PATH "cudf/cpp/build/include")
#
#find_package(cudf REQUIRED)
#find_package(ICU REQUIRED)
#include_directories(${ICU_INCLUDE_DIRS})
#target_link_libraries(ICU::uc ICU::i18n ICU::data)

find_package(ICU REQUIRED COMPONENTS uc i18n)

if(ICU_FOUND)
    message(STATUS "ICU found: ${ICU_VERSION}")
    message(STATUS "ICU include dir: ${ICU_INCLUDE_DIRS}")
    message(STATUS "ICU libs: ${ICU_LIBRARIES}")
else()
    message(FATAL_ERROR "ICU not found")
endif()

add_executable(cuda_exe src/cuda/main.cu
        src/cuda/icu_lowercase.cpp
        include/icu_lowercase.h
        src/cuda/lemmatizer.cpp
        include/lemmatizer.h
        src/cuda/trie.cpp
        include/trie.h
        include/structs.h
        src/cuda/lemmatizer_kernel.cu
        include/lemmatizer_kernel.h
        include/lemmatizer_kernel.cuh
        src/cuda/GpuLemmatizer.cpp)

include_directories(
        /usr/lib/jvm/java-21-openjdk-amd64/include
        /usr/lib/jvm/java-21-openjdk-amd64/include/linux
)

target_include_directories(cuda_exe PRIVATE ${ICU_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}  # <--- make sure this is here
        include
        src/cuda)
target_link_libraries(cuda_exe PRIVATE ${ICU_LIBRARIES})

set_target_properties(cuda_exe PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)




add_library(lemmatizer SHARED
        src/cuda/GpuLemmatizer.cpp
        src/cuda/lemmatizer.cpp
        src/cuda/lemmatizer_kernel.cu
        src/cuda/trie.cpp
)

target_include_directories(lemmatizer PRIVATE
        ${ICU_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}  # <--- make sure this is here
        include
        src/cuda
        /home/kowalski0805/.jdks/azul-17.0.14/include
        /home/kowalski0805/.jdks/azul-17.0.14/include/linux
)

set_target_properties(lemmatizer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)