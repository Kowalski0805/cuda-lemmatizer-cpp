#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <dawgdic/dictionary.h>
#include "icu_lowercase.h"
#include "lemmatizer_kernel.cuh"
#include "structs.h"
#include "trie.h"

int main_gpu_dict() {
    // std::vector<std::pair<std::string, std::string>> dict = {
    //     {"котик", "кіт"},
    //     {"собака", "собака"},
    //     {"місто", "місто"},
    // };
    // std::vector<char> h_dict_keys, h_dict_vals;
    // int dict_size = 0;
    // if (!load_flat_dict("dict_keys.bin", "dict_vals.bin", h_dict_keys, h_dict_vals, dict_size)) {
    //     std::cerr << "Failed to load dictionary from binary files.\n";
    //     return 1;
    // }

    std::vector<char> h_dict_keys, h_dict_vals;
    int dict_size = 0;

    if (!load_csv_dict("ukr_morph_dict.csv", h_dict_keys, h_dict_vals, dict_size)) {
        std::cerr << "Failed to load CSV dictionary\n";
        return 1;
    }

    std::vector<std::string> input = {"заводського", "китайка", "корейська", "ходив"};
    int num_words = input.size();

    // Flatten input and dict to GPU-friendly char arrays
    std::vector<char> h_input(num_words * MAX_WORD_LEN, 0);
    std::vector<char> h_output(num_words * MAX_WORD_LEN, 0);

    for (int i = 0; i < std::min(10, dict_size); ++i) {
        std::cout << "dict: "
                  << &h_dict_keys[i * MAX_WORD_LEN]
                  << " → "
                  << &h_dict_vals[i * MAX_WORD_LEN]
                  << "\n";
    }

    for (int i = 0; i < num_words; ++i)
        strncpy(&h_input[i * MAX_WORD_LEN], input[i].c_str(), MAX_WORD_LEN);

    // Allocate GPU memory
    char *d_input, *d_output, *d_dict_keys, *d_dict_vals;
    cudaMalloc(&d_input, h_input.size());
    cudaMalloc(&d_output, h_output.size());
    cudaMalloc(&d_dict_keys, h_dict_keys.size());
    cudaMalloc(&d_dict_vals, h_dict_vals.size());

    cudaMemcpy(d_input, h_input.data(), h_input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dict_keys, h_dict_keys.data(), h_dict_keys.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dict_vals, h_dict_vals.data(), h_dict_vals.size(), cudaMemcpyHostToDevice);

    // Run kernel
    normalize_kernel<<<1, 256>>>(d_input, d_output, d_dict_keys, d_dict_vals, num_words, dict_size);
    cudaMemcpy(h_output.data(), d_output, h_output.size(), cudaMemcpyDeviceToHost);

    std::cout << "Normalized output:\n";
    for (int i = 0; i < num_words; ++i)
        std::cout << "- " << &h_input[i * MAX_WORD_LEN] << " → " << &h_output[i * MAX_WORD_LEN] << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dict_keys);
    cudaFree(d_dict_vals);

    return 0;
}

int main_dawg() {
    std::vector<char> h_dict_vals, h_dict_keys;
    dawgdic::Dictionary dict;
    int dict_size = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // if (!load_csv_dict_to_dawg("ukr_morph_dict.csv", dict, h_dict_vals, dict_size)) {
    //     std::cerr << "Failed to load CSV dictionary\n";
    //     return 1;
    // }
    // if (!load_csv_dict("ukr_morph_dict.csv", h_dict_keys, h_dict_vals, dict_size)) {
    //     std::cerr << "Failed to load CSV dictionary\n";
    //     return 1;
    // }

    load_dawg("morph_uk.dawg", dict);
    load_bin_vector("dict_vals.bin", h_dict_vals);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to load dictionary: " << elapsed.count() << " seconds\n";

    // save_dawg("morph_uk.dawg", dict);
    // // save_bin_vector("dict_keys.bin", h_dict_keys);
    // save_bin_vector("dict_vals.bin", h_dict_vals);

    auto start1 = std::chrono::high_resolution_clock::now();


    std::cout << dict.Find("теплому") << "\n";
    std::cout << dict.Find("ящірки") << "\n";
    std::cout << dict.Find("синього") << "\n";
    std::cout << dict.Find("українська") << "\n";
    std::cout << dict.Find("ходив") << "\n";

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Time taken to find words: " << elapsed1.count() << " seconds\n";


    auto start2 = std::chrono::high_resolution_clock::now();
    // find lemma for each word
    std::vector<std::string> input = {
        "теплому",     // adjective: masc, dat
        "ящірки",      // noun: gen sg
        "синього",     // adjective: masc, gen
        "українська",  // adjective: fem nom
        "ходив",       // verb: masc past

        // More noun forms
        "двері",       // noun: nom pl
        "чоловіка",    // noun: gen sg
        "жінці",       // noun: dat sg
        "вікном",      // noun: ins sg
        "містах",      // noun: loc pl

        // Verb forms
        "читала",      // verb: fem past
        "пишемо",      // verb: 1pl pres
        "розмовляєш",  // verb: 2sg pres
        "поїхав",      // verb: masc past
        "буду",        // verb: 1sg fut

        // Adjective/participle/etc.
        "старіший",    // comparative
        "найбільший",  // superlative
        "відомому",    // adjective: masc, loc
        "знайдену",     // participle/adjective

        // Random test cases
        "невідоме",    // neuter adjective
        "новини",      // noun: pl nom/acc
        "книжками",    // noun: ins pl
        "допомагаючи", // gerund
        "бігатимеш"    // verb: 2sg fut
    };
    for (const auto& word : input) {
        int i = dict.Find(word.c_str());
        if (i == -1) {
            std::cout << "Word not found: " << word << "\n";
            continue;
        }
        const char* lemma = h_dict_vals.data() + i;
        // Find the first null terminator in the lemma
        const char* end = lemma;
        while (*end != '\0' && end - lemma < MAX_WORD_LEN) {
            ++end;
        }
        std::string lemma_str(lemma, end - lemma);
        // Convert to lowercase
        // Print the lemma
        std::cout << "Lemma for " << word << ": " << lemma_str << "\n";
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Time taken to find lemmas: " << elapsed2.count() * 1000 << " milliseconds\n";

    return 0;
}

int main_gpu() {
        // Build from CSV
    std::vector<GpuState> h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char> h_lemmas;

    // build_gpu_trie_from_csv("ukr_morph_dict.csv", h_states, h_transitions, h_lemmas);
    // save_bin_vector("gpu_states.bin", h_states);
    // save_bin_vector("gpu_transitions.bin", h_transitions);
    // save_bin_vector("gpu_lemmas.bin", h_lemmas);
    auto start_load = std::chrono::high_resolution_clock::now();
    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_load = end_load - start_load;
    std::cout << "Time taken to load GPU data: " << elapsed_load.count() << " seconds\n";

    // Input words
    std::vector<std::string> input_words = {
        "теплому",     // adjective: masc, dat
        "ящірки",      // noun: gen sg
        "синього",     // adjective: masc, gen
        "українська",  // adjective: fem nom
        "ходив",       // verb: masc past

        // More noun forms
        "двері",       // noun: nom pl
        "чоловіка",    // noun: gen sg
        "жінці",       // noun: dat sg
        "вікном",      // noun: ins sg
        "містах",      // noun: loc pl

        // Verb forms
        "читала",      // verb: fem past
        "пишемо",      // verb: 1pl pres
        "розмовляєш",  // verb: 2sg pres
        "поїхав",      // verb: masc past
        "буду",        // verb: 1sg fut

        // Adjective/participle/etc.
        "старіший",    // comparative
        "найбільший",  // superlative
        "відомому",    // adjective: masc, loc
        "знайдену",     // participle/adjective

        // Random test cases
        "невідоме",    // neuter adjective
        "новини",      // noun: pl nom/acc
        "книжками",    // noun: ins pl
        "допомагаючи", // gerund
        "бігатимеш"    // verb: 2sg fut
    };
    int num_words = input_words.size();
    std::vector<char> h_input(num_words * MAX_WORD_LEN, 0);
    std::vector<char> h_output(num_words * MAX_WORD_LEN, 0);

    for (int i = 0; i < num_words; ++i) {
        std::string lower = lowercase_ukr(input_words[i]);
        strncpy(&h_input[i * MAX_WORD_LEN], lower.c_str(), MAX_WORD_LEN);
    }

    auto start1 = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    GpuState* d_states;
    GpuTransition* d_transitions;
    char *d_lemmas, *d_input, *d_output;

    cudaMalloc(&d_states, h_states.size() * sizeof(GpuState));
    cudaMalloc(&d_transitions, h_transitions.size() * sizeof(GpuTransition));
    cudaMalloc(&d_lemmas, h_lemmas.size());
    cudaMalloc(&d_input, h_input.size());
    cudaMalloc(&d_output, h_output.size());

    cudaMemcpy(d_states, h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas, h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), h_input.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lookup_kernel<<<blocks, threads>>>(d_input, num_words, d_states, d_transitions, d_lemmas, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU processing time: " << milliseconds << " ms\n";
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_output.data(), d_output, h_output.size(), cudaMemcpyDeviceToHost);

    // Show results
    std::cout << "GPU-normalized output:\n";
    for (int i = 0; i < num_words; ++i) {
        std::cout << "- " << &h_input[i * MAX_WORD_LEN]
                  << " → " << &h_output[i * MAX_WORD_LEN] << "\n";
    }

    // Cleanup
    cudaFree(d_states);
    cudaFree(d_transitions);
    cudaFree(d_lemmas);
    cudaFree(d_input);
    cudaFree(d_output);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end1 - start1;
    std::cout << "Time taken for GPU processing with copy: " << elapsed.count() << " seconds\n";

    return 0;
}


int main() {
    main_dawg();
    return main_gpu();
}
