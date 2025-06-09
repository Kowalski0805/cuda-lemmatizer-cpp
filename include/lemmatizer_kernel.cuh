//
// Created by Illya on 20.04.2025.
//

#ifndef LEMMATIZER_KERNEL_CUH
#define LEMMATIZER_KERNEL_CUH
#pragma once
#include "structs.h"

__global__ void normalize_kernel(char* d_input, char* d_output,
                                 const char* dict_keys, const char* dict_vals,
                                 int num_words, int dict_size);

__global__ void lookup_kernel(
    cudf::column_device_view d_input,
    int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    thrust::pair<char const*, cudf::size_type>* d_output
);

__device__ thrust::pair<char const*, cudf::size_type> d_lookup_kernel(
    cudf::column_device_view d_input,
    const int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    const int idx
);

__global__ void lemmatize_kernel(cudf::column_device_view d_words,
                                 cudf::size_type const* d_offsets,
                                 char* d_chars);

__global__ void sizes_kernel(cudf::column_device_view d_words,
                             cudf::size_type* d_sizes);

void launch_sizes_kernel(cudf::column_device_view d_words, int* d_sizes, cudaStream_t stream);
void launch_lemmatize_kernel(cudf::column_device_view d_words, const int* d_offsets, char* d_chars, cudaStream_t stream);

#endif //LEMMATIZER_KERNEL_CUH
