#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "structs.h"
#include "trie.h"
#include "icu_lowercase.h"

// Forward-declare kernel defined in lemmatizer_kernel.cu
__global__ void lookup_kernel_bsearch(
    const char* d_input,
    int num_words,
    const char* d_keys,
    const char* d_vals,
    int num_entries,
    char* d_output);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }
    const std::string input_path = argv[1];
    const bool has_output = (argc >= 3);
    const std::string output_path = has_output ? argv[2] : "";

    // Load or build bsearch dict
    std::vector<char> h_keys, h_vals;
    int num_entries = 0;

    load_bin_vector("bsearch_keys.bin", h_keys);
    load_bin_vector("bsearch_vals.bin", h_vals);

    if (h_keys.empty() || h_vals.empty()) {
        std::cerr << "bsearch_keys.bin / bsearch_vals.bin not found — building from CSV...\n";
        build_flat_sorted_dict_from_csv("ukr_morph_dict.csv", h_keys, h_vals, num_entries);
        save_bin_vector("bsearch_keys.bin", h_keys);
        save_bin_vector("bsearch_vals.bin", h_vals);
    } else {
        num_entries = static_cast<int>(h_keys.size() / MAX_WORD_LEN);
        std::cerr << "Loaded bsearch dict: " << num_entries << " entries\n";
    }

    if (num_entries == 0) {
        std::cerr << "Empty dictionary — aborting.\n";
        return 1;
    }

    // Read and tokenize input
    std::ifstream fin(input_path);
    if (!fin) {
        std::cerr << "Cannot open input file: " << input_path << "\n";
        return 1;
    }
    std::vector<std::string> lines;
    {
        std::string line;
        while (std::getline(fin, line)) lines.push_back(std::move(line));
    }

    std::vector<int> line_word_count(lines.size(), 0);
    std::vector<std::string> words;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::istringstream ss(lines[i]);
        std::string token;
        while (ss >> token) {
            words.push_back(lowercase_ukr(token));
            ++line_word_count[i];
        }
    }
    const int num_words = static_cast<int>(words.size());

    // Handle empty input
    if (num_words == 0) {
        std::ostream* out_ptr = &std::cout;
        std::ofstream fout;
        if (has_output) { fout.open(output_path); out_ptr = &fout; }
        for (size_t i = 0; i < lines.size(); ++i) *out_ptr << '\n';
        return 0;
    }

    // Pack words into fixed-stride buffer
    const size_t stride_bytes = static_cast<size_t>(num_words) * MAX_WORD_LEN;
    std::vector<char> h_input(stride_bytes, 0);
    int truncated = 0;
    for (int i = 0; i < num_words; ++i) {
        const auto& w = words[i];
        if (w.size() >= static_cast<size_t>(MAX_WORD_LEN)) ++truncated;
        const size_t copy_len = std::min(w.size(), static_cast<size_t>(MAX_WORD_LEN - 1));
        std::memcpy(h_input.data() + i * MAX_WORD_LEN, w.data(), copy_len);
    }
    if (truncated > 0)
        std::cerr << "[warn] " << truncated << " word(s) truncated to "
                  << MAX_WORD_LEN - 1 << " bytes\n";

    // --- TOTAL TIMER START (includes H2D, kernel, D2H) ---
    auto total_start = std::chrono::high_resolution_clock::now();

    // Allocate device buffers
    char* d_input_dev  = nullptr;
    char* d_output_dev = nullptr;
    char* d_keys_dev   = nullptr;
    char* d_vals_dev   = nullptr;

    const size_t dict_bytes = static_cast<size_t>(num_entries) * MAX_WORD_LEN;

    cudaMalloc(&d_input_dev,  stride_bytes);
    cudaMalloc(&d_output_dev, stride_bytes);
    cudaMalloc(&d_keys_dev,   dict_bytes);
    cudaMalloc(&d_vals_dev,   dict_bytes);

    cudaMemcpy(d_input_dev, h_input.data(),  stride_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys_dev,  h_keys.data(),   dict_bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_dev,  h_vals.data(),   dict_bytes,   cudaMemcpyHostToDevice);
    cudaMemset(d_output_dev, 0, stride_bytes);

    // --- KERNEL TIMING ---
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    const int threads = 128;
    const int blocks  = (num_words + threads - 1) / threads;

    cudaEventRecord(ev_start);
    lookup_kernel_bsearch<<<blocks, threads>>>(
        d_input_dev, num_words, d_keys_dev, d_vals_dev, num_entries, d_output_dev);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Copy results back
    std::vector<char> h_output(stride_bytes);
    cudaMemcpy(h_output.data(), d_output_dev, stride_bytes, cudaMemcpyDeviceToHost);

    auto total_end = std::chrono::high_resolution_clock::now();
    // --- TOTAL TIMER END ---

    cudaFree(d_input_dev);
    cudaFree(d_output_dev);
    cudaFree(d_keys_dev);
    cudaFree(d_vals_dev);

    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Decode results
    std::vector<std::string> result_words(num_words);
    for (int i = 0; i < num_words; ++i) {
        const char* p = h_output.data() + i * MAX_WORD_LEN;
        result_words[i] = std::string(p, ::strnlen(p, MAX_WORD_LEN));
        if (result_words[i].empty()) result_words[i] = words[i];
    }

    double throughput = (kernel_ms > 0.f) ? (num_words / (kernel_ms / 1000.0)) : 0.0;
    std::cerr << "Words: " << num_words
              << "  Kernel: " << kernel_ms << " ms"
              << "  Total (incl. H2D/D2H): " << total_ms << " ms"
              << "  Throughput: " << static_cast<long long>(throughput) << " words/sec\n";

    // Write output, preserving line structure
    std::ostream* out_ptr = &std::cout;
    std::ofstream fout;
    if (has_output) {
        fout.open(output_path);
        if (!fout) {
            std::cerr << "Cannot open output file: " << output_path << "\n";
            return 1;
        }
        out_ptr = &fout;
    }

    int word_idx = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        for (int j = 0; j < line_word_count[i]; ++j) {
            if (j > 0) *out_ptr << ' ';
            *out_ptr << result_words[word_idx++];
        }
        *out_ptr << '\n';
    }

    return 0;
}
