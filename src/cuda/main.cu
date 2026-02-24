#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <cudf/column/column_factories.hpp>
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
    // --- 1. Load Trie Data from Disk ---
    std::vector<GpuState> h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char> h_lemmas;

    build_gpu_trie_from_csv("ukr_morph_dict.csv", h_states, h_transitions, h_lemmas);
    save_bin_vector("gpu_states.bin", h_states);
    save_bin_vector("gpu_transitions.bin", h_transitions);
    save_bin_vector("gpu_lemmas.bin", h_lemmas);
    auto start_load = std::chrono::high_resolution_clock::now();
    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);

    auto end_load = std::chrono::high_resolution_clock::now();
    std::cout << "Data loaded in " << std::chrono::duration<double>(end_load - start_load).count() << "s\n";

    // --- 2. Prepare Input via cuDF ---
    std::vector<std::string> input_words = {
        "теплому", "ящірки", "синього", "українська", "ходив",
        "двері", "чоловіка", "жінці", "вікном", "містах",
        "читала", "пишемо", "розмовляєш", "поїхав", "буду",
        "старіший", "найбільший", "відомому", "знайдену",
        "невідоме", "новини", "книжками", "допомагаючи", "бігатимеш"
    };

    std::vector<std::string> lower_input;
    for (const auto& w : input_words) {
        lower_input.push_back(lowercase_ukr(w));
    }

    std::vector<char> h_chars;
    std::vector<int32_t> h_offsets = {0};
    for (const auto& s : lower_input) {
        h_chars.insert(h_chars.end(), s.begin(), s.end());
        h_offsets.push_back(static_cast<int32_t>(h_chars.size()));
    }

    rmm::device_uvector<char> d_chars_raw(h_chars.size(), rmm::cuda_stream_default);
    rmm::device_uvector<int32_t> d_offsets_raw(h_offsets.size(), rmm::cuda_stream_default);

    cudaMemcpy(d_chars_raw.data(), h_chars.data(), h_chars.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets_raw.data(), h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Build the offsets column required by the [3/3] factory
    auto offsets_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, h_offsets.size(), cudf::mask_state::UNALLOCATED
    );
    cudaMemcpy(offsets_column->mutable_view().data<int32_t>(), d_offsets_raw.data(),
               h_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost); // Actually, we copy to the column's device memory

    // Construct the string column using [3/3]
    auto input_col = cudf::make_strings_column(
        lower_input.size(),
        std::move(offsets_column),
        rmm::device_buffer{d_chars_raw.data(), d_chars_raw.size(), rmm::cuda_stream_default},
        0, // null_count
        rmm::device_buffer{} // empty null_mask
    );
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    int num_words = static_cast<int>(input_words.size());

    // --- 3. Allocate Trie & Output Memory via RMM ---
    // rmm::device_uvector is like std::vector for GPU (RAII, no manual cudaFree needed)
    rmm::device_uvector<GpuState> d_states(h_states.size(), rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_transitions(h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char> d_lemmas(h_lemmas.size(), rmm::cuda_stream_default);

    // Output buffer: using your existing fixed-width logic for the result
    rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> d_output(num_words, rmm::cuda_stream_default);
    cudaMemset(d_output.data(), 0, d_output.size());

    // --- 4. Copy Trie Data to Device ---
    cudaMemcpy(d_states.data(), h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions.data(), h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas.data(), h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    // --- 5. Launch Kernel ---
    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;

    auto* d_trans_ptr = d_transitions.data();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto start_proc = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);

    // Pass the de-referenced view object (*d_input_view)
    lookup_kernel<<<blocks, threads>>>(
        *d_input_view,
        num_words,
        d_states.data(),
        d_trans_ptr,
        d_lemmas.data(),
        d_output.data()
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution time: " << milliseconds << " ms\n";

    // --- 6. Results & Cleanup ---
    std::vector<thrust::pair<const char*, cudf::size_type>> h_results(num_words);

    // 2. Копіюємо дані (пам'ятай про sizeof!)
    cudaMemcpy(h_results.data(), d_output.data(),
               num_words * sizeof(thrust::pair<const char*, cudf::size_type>),
               cudaMemcpyDeviceToHost);

    std::cout << "\nGPU-normalized output:\n";
    for (int i = 0; i < num_words; ++i) {
        const char* gpu_ptr = h_results[i].first;
        int length = h_results[i].second;

        if (gpu_ptr == nullptr || length <= 0) {
            std::cout << "- " << lower_input[i] << " → NOT FOUND\n";
            continue;
        }

        // РАХУЄМО ОФСЕТ: де ця лема лежить відносно початку масиву d_lemmas
        // Ми знаємо, що d_lemmas.data() — це початок масиву на GPU
        size_t offset = gpu_ptr - d_lemmas.data();

        // Тепер беремо цей офсет і застосовуємо його до нашого хост-вектора h_lemmas
        // h_lemmas — це той вектор, який ми завантажили з gpu_lemmas.bin на самому початку
        if (offset < h_lemmas.size()) {
            std::string lemma(h_lemmas.data() + offset, length);
            std::cout << "- " << lower_input[i] << " → " << lemma << "\n";
        } else {
            std::cout << "- " << lower_input[i] << " → ERROR: Invalid offset\n";
        }
    }

    auto end_proc = std::chrono::high_resolution_clock::now();
    std::cout << "Total processing (inc. H2D/D2H): "
              << std::chrono::duration<double>(end_proc - start_proc).count() << "s\n";

    return 0;
}


int main() {
    main_dawg();
    return main_gpu();
}
