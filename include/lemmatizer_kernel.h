//
// Created by Illya on 20.04.2025.
//

#ifndef LEMMATIZER_KERNEL_H
#define LEMMATIZER_KERNEL_H
#pragma once

#include "structs.h"

extern "C" void launch_lookup_kernel(
    const char* d_input,
    int num_words,
    const GpuState* d_states,
    const GpuTransition* d_transitions,
    const char* d_lemmas,
    char* d_output
);
#endif //LEMMATIZER_KERNEL_H
