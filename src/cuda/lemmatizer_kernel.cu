//
// Created by Illya on 20.04.2025.
//

#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <fstream>
#include "structs.h"


// Kernel that performs wordform-to-lemma lookup using GPU-transcoded DAWG
__global__ void dawg_lookup_kernel(const char* input_words, int num_words,
                                   const GpuState* states,
                                   const GpuTransition* transitions,
                                   const char* lemma_buffer,
                                   char* output_lemmas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    const char* word = input_words + idx * MAX_WORD_LEN;
    char* output = output_lemmas + idx * MAX_WORD_LEN;

    int state = 0;
    for (int i = 0; i < MAX_WORD_LEN && word[i] != '\0'; ++i) {
        char ch = word[i];
        const GpuState& s = states[state];
        bool found = false;

        for (int j = 0; j < s.num_transitions; ++j) {
            const GpuTransition& t = transitions[s.transition_start_idx + j];
            if (t.c == ch) {
                state = t.next_state;
                found = true;
                break;
            }
        }

        if (!found) {
            state = -1;
            break;
        }
    }

    if (state != -1 && states[state].lemma_offset >= 0) {
        const char* lemma = &lemma_buffer[states[state].lemma_offset];
        for (int i = 0; i < MAX_WORD_LEN && lemma[i] != '\0'; ++i) {
            output[i] = lemma[i];
        }
    } else {
        // fallback: copy input to output
        for (int i = 0; i < MAX_WORD_LEN; ++i) {
            output[i] = word[i];
            if (word[i] == '\0') break;
        }
    }
}

__device__ bool str_eq(const char* a, const char* b) {
    for (int i = 0; i < MAX_WORD_LEN; ++i) {
        if (a[i] != b[i]) return false;
        if (a[i] == '\0') break;
    }
    return true;
}

__global__ void normalize_kernel(char* d_input, char* d_output, const char* dict_keys, const char* dict_vals, int num_words, int dict_size) {
    const u_int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    char* input_word = d_input + idx * MAX_WORD_LEN;
    char* output_word = d_output + idx * MAX_WORD_LEN;

    for (int i = 0; i < dict_size; ++i) {
        const char* key = dict_keys + i * MAX_WORD_LEN;
        const char* val = dict_vals + i * MAX_WORD_LEN;

        if (str_eq(input_word, key)) {
            memcpy(output_word, val, MAX_WORD_LEN);
            return;
        }
    }

    // Not found → copy input as fallback
    memcpy(output_word, input_word, MAX_WORD_LEN);
}

__global__ void lookup_kernel(
    const char* input,  // input words (flat array)
    int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    char* output         // output buffer (flat array)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    const char* word = input + idx * MAX_WORD_LEN;
    char* result = output + idx * MAX_WORD_LEN;

    int state = 0;

    // Traverse the trie
    for (int i = 0; i < MAX_WORD_LEN && word[i] != '\0'; ++i) {
        char ch = word[i];
        const GpuState& s = states[state];
        bool found = false;

        for (int j = 0; j < s.num_transitions; ++j) {
            const GpuTransition& t = transitions[s.transition_start_idx + j];
            if (t.c == ch) {
                state = t.next_state;
                found = true;
                break;
            }
        }

        if (!found) {
            // Fallback to input if word not found
            for (int k = 0; k < MAX_WORD_LEN; ++k) {
                result[k] = word[k];
                if (word[k] == '\0') break;
            }
            return;
        }
    }

    // Final state reached — copy lemma if found
    const GpuState& final_state = states[state];
    if (final_state.lemma_offset >= 0) {
        for (int i = 0; i < MAX_WORD_LEN; ++i) {
            char c = lemmas[final_state.lemma_offset + i];
            result[i] = c;
            if (c == '\0') break;
        }
    } else {
        // No lemma — fallback
        for (int k = 0; k < MAX_WORD_LEN; ++k) {
            result[k] = word[k];
            if (word[k] == '\0') break;
        }
    }
}

extern "C" void launch_lookup_kernel(
    const char* d_input, int num_words,
    const GpuState* d_states, const GpuTransition* d_transitions,
    const char* d_lemmas, char* d_output)
{
    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;
    lookup_kernel<<<blocks, threads>>>(d_input, num_words, d_states, d_transitions, d_lemmas, d_output);
    cudaDeviceSynchronize();
}