//
// Created by Illya on 19.04.2025.
//

#include "lemmatizer.h"
#include "structs.h"
#include "trie.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>

#include "lemmatizer_kernel.h"
#include "../../include/icu_lowercase.h"

struct GpuState;
struct GpuTransition;
static bool is_initialized = false;
static std::vector<GpuState> h_states;
static std::vector<GpuTransition> h_transitions;
static std::vector<char> h_lemmas;

static GpuState* d_states = nullptr;
static GpuTransition* d_transitions = nullptr;
static char* d_lemmas = nullptr;

void init_trie_data() {
    if (is_initialized) return;

    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);

    cudaMalloc(&d_states, h_states.size() * sizeof(GpuState));
    cudaMalloc(&d_transitions, h_transitions.size() * sizeof(GpuTransition));
    cudaMalloc(&d_lemmas, h_lemmas.size());

    cudaMemcpy(d_states, h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas, h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    is_initialized = true;
}

extern "C" void lemmatize_batch(const char** input_words, int num_words, char** output_words) {
    init_trie_data();

    std::vector<char> h_input(num_words * MAX_WORD_LEN, 0);
    std::vector<char> h_output(num_words * MAX_WORD_LEN, 0);

    for (int i = 0; i < num_words; ++i) {
        std::string word = input_words[i];  // assuming it's already lowercased in Java
        strncpy(&h_input[i * MAX_WORD_LEN], word.c_str(), MAX_WORD_LEN);
    }

    char* d_input = nullptr;
    char* d_output = nullptr;

    cudaMalloc(&d_input, h_input.size());
    cudaMalloc(&d_output, h_output.size());

    cudaMemcpy(d_input, h_input.data(), h_input.size(), cudaMemcpyHostToDevice);

    launch_lookup_kernel(d_input, num_words, d_states, d_transitions, d_lemmas, d_output);

    cudaMemcpy(h_output.data(), d_output, h_output.size(), cudaMemcpyDeviceToHost);

    // Return results per word
    for (int i = 0; i < num_words; ++i) {
        strncpy(output_words[i], &h_output[i * MAX_WORD_LEN], MAX_WORD_LEN);
        output_words[i][MAX_WORD_LEN - 1] = '\0'; // ensure null-termination
    }

    cudaFree(d_input);
    cudaFree(d_output);
}
