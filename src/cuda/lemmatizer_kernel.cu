//
// Created by Illya on 20.04.2025.
//

#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/string_view.hpp>

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
    cudf::column_device_view d_input,
    const int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    thrust::pair<char const*, cudf::size_type>* d_output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    auto const word = d_input.element<cudf::string_view>(idx);
    int state = 0;

    // Traverse the trie
    for (int i = 0; i < word.size_bytes(); ++i) {
        char ch = word.data()[i];
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
            cudf::string_view s = d_input.element<cudf::string_view>(idx);
            d_output[idx] = thrust::make_pair(s.data(), s.size_bytes());
            return;
        }
    }

    // Final state reached — copy lemma if found
    const GpuState& final_state = states[state];
    if (final_state.lemma_offset >= 0) {
        int len = 0;
        while (lemmas[final_state.lemma_offset + len] != '\0' && len < MAX_WORD_LEN) {
            ++len;
        }
        d_output[idx] = thrust::make_pair(lemmas + final_state.lemma_offset, len);
    } else {
        // No lemma — fallback
        cudf::string_view s = d_input.element<cudf::string_view>(idx);
        d_output[idx] = thrust::make_pair(s.data(), s.size_bytes());
    }
}

__device__ thrust::pair<char const*, cudf::size_type> d_lookup_kernel(
    cudf::column_device_view d_input,
    const int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    const int idx
) {
    if (idx >= num_words) return thrust::make_pair(nullptr, 0);

    auto const word = d_input.element<cudf::string_view>(idx);
    int state = 0;

    // Traverse the trie
    for (int i = 0; i < word.size_bytes(); ++i) {
        char ch = word.byte_offset(i);
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
            cudf::string_view s = d_input.element<cudf::string_view>(idx);
            return thrust::make_pair(s.data(), s.size_bytes());
        }
    }

    // Final state reached — copy lemma if found
    const GpuState& final_state = states[state];
    if (final_state.lemma_offset >= 0) {
        for (int i = 0; i < MAX_WORD_LEN; ++i) {
            if (char c = lemmas[final_state.lemma_offset + i]; c == '\0') {
                return thrust::make_pair(lemmas + final_state.lemma_offset, i);
            }
        }
    } else {
        // No lemma — fallback
        cudf::string_view s = d_input.element<cudf::string_view>(idx);
        return thrust::make_pair(s.data(), s.size_bytes());
    }
    return thrust::make_pair(nullptr, 0);
}

extern "C" void launch_lookup_kernel(
    cudf::column_device_view const &d_input, int num_words,
    const GpuState* d_states, const GpuTransition* d_transitions,
    const char* d_lemmas, thrust::pair<char const*, cudf::size_type>* d_output)
{
    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;
    lookup_kernel<<<blocks, threads>>>(d_input, num_words, d_states, d_transitions, d_lemmas, d_output);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void sizes_kernel(cudf::column_device_view d_words,
                             cudf::size_type* d_sizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_words.size()) return;

    auto word = d_words.element<cudf::string_view>(idx);

    // TODO: обчисли потрібну довжину результату для word
    d_sizes[idx] = word.size_bytes();  // (тимчасово просто повертає оригінал)
}


__global__ void lemmatize_kernel(cudf::column_device_view d_words,
                                 cudf::size_type const* d_offsets,
                                 char* d_chars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_words.size()) return;

    auto word = d_words.element<cudf::string_view>(idx);
    char* out = d_chars + d_offsets[idx];

    // TODO: заміни на результат лематизації (тимчасово — копія входу)
    memcpy(out, word.data(), word.size_bytes());
}

void launch_sizes_kernel(cudf::column_device_view d_words, int* d_sizes, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (d_words.size() + block_size - 1) / block_size;
    sizes_kernel<<<num_blocks, block_size, 0, stream>>>(d_words, d_sizes);
}

void launch_lemmatize_kernel(cudf::column_device_view d_words, const int* d_offsets, char* d_chars, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (d_words.size() + block_size - 1) / block_size;
    lemmatize_kernel<<<num_blocks, block_size, 0, stream>>>(d_words, d_offsets, d_chars);
}