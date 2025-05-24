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

__global__ void lookup_kernel(const char* input, int num_words,
                              const GpuState* states,
                              const GpuTransition* transitions,
                              const char* lemmas,
                              char* output);

#endif //LEMMATIZER_KERNEL_CUH
