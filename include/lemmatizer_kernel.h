//
// Created by Illya on 20.04.2025.
//

#ifndef LEMMATIZER_KERNEL_H
#define LEMMATIZER_KERNEL_H
#pragma once

#include "structs.h"

extern "C" void launch_lookup_kernel(
    cudf::column_device_view const &d_input,
    int num_words,
    const GpuState* d_states,
    const GpuTransition* d_transitions,
    const char* d_lemmas,
    thrust::pair<char const*, cudf::size_type>* d_output
);
#endif //LEMMATIZER_KERNEL_H
