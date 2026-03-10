#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>

#include "structs.h"
#include "trie.h"
#include "icu_lowercase.h"
#include "lemmatizer_kernel.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }
    const std::string input_path = argv[1];
    const bool has_output = (argc >= 3);
    const std::string output_path = has_output ? argv[2] : "";

    // Load trie data from disk
    std::vector<GpuState> h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char> h_lemmas;
    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);

    if (h_states.empty()) {
        std::cerr << "Failed to load trie data. Run from cmake-build-debug/.\n";
        return 1;
    }

    // --- PREPROCESS TIMER START (file I/O + tokenize + lowercase_ukr) ---
    auto preprocess_start = std::chrono::high_resolution_clock::now();

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

    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    // --- PREPROCESS TIMER END ---

    // Handle empty input
    if (num_words == 0) {
        std::ostream* out_ptr = &std::cout;
        std::ofstream fout;
        if (has_output) { fout.open(output_path); out_ptr = &fout; }
        for (size_t i = 0; i < lines.size(); ++i) *out_ptr << '\n';
        return 0;
    }

    // --- PACK TIMER START (build flat char/offset arrays) ---
    auto pack_start = std::chrono::high_resolution_clock::now();

    std::vector<char> h_chars;
    std::vector<int32_t> h_offsets = {0};
    h_chars.reserve(static_cast<size_t>(num_words) * 16);
    for (const auto& w : words) {
        h_chars.insert(h_chars.end(), w.begin(), w.end());
        h_offsets.push_back(static_cast<int32_t>(h_chars.size()));
    }

    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - pack_start).count();
    // --- PACK TIMER END ---

    // --- TOTAL TIMER START (includes H2D, kernel, D2H) ---
    auto total_start = std::chrono::high_resolution_clock::now();

    // --- H2D TIMER START ---
    auto h2d_start = std::chrono::high_resolution_clock::now();

    // Upload chars to device
    rmm::device_uvector<char> d_chars_raw(h_chars.size(), rmm::cuda_stream_default);
    cudaMemcpy(d_chars_raw.data(), h_chars.data(), h_chars.size(), cudaMemcpyHostToDevice);

    // Build cuDF strings column (mirrors main_gpu() pattern)
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(h_offsets.size()),
        cudf::mask_state::UNALLOCATED);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
               h_offsets.data(),
               h_offsets.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    auto input_col = cudf::make_strings_column(
        static_cast<cudf::size_type>(num_words),
        std::move(offsets_col),
        rmm::device_buffer{d_chars_raw.data(), h_chars.size(), rmm::cuda_stream_default},
        0,
        rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    // Upload trie to device
    rmm::device_uvector<GpuState> d_states(h_states.size(), rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_transitions(h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char> d_lemmas(h_lemmas.size(), rmm::cuda_stream_default);
    cudaMemcpy(d_states.data(), h_states.data(),
               h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions.data(), h_transitions.data(),
               h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas.data(), h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    // Allocate output
    rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> d_output(
        num_words, rmm::cuda_stream_default);
    cudaMemset(d_output.data(), 0,
               static_cast<size_t>(num_words) * sizeof(thrust::pair<const char*, cudf::size_type>));

    cudaDeviceSynchronize();
    auto h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    // --- H2D TIMER END ---

    // --- KERNEL TIMING ---
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;

    cudaEventRecord(ev_start);
    lookup_kernel<<<blocks, threads>>>(
        *d_input_view,
        num_words,
        d_states.data(),
        d_transitions.data(),
        d_lemmas.data(),
        d_output.data());
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // --- D2H TIMER START ---
    auto d2h_start = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    std::vector<thrust::pair<const char*, cudf::size_type>> h_results(num_words);
    cudaMemcpy(h_results.data(), d_output.data(),
               static_cast<size_t>(num_words) * sizeof(thrust::pair<const char*, cudf::size_type>),
               cudaMemcpyDeviceToHost);

    auto total_end = std::chrono::high_resolution_clock::now();
    // --- TOTAL TIMER END / D2H TIMER END ---

    double d2h_ms = std::chrono::duration<double, std::milli>(total_end - d2h_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Decode results: pointer into d_lemmas → offset into h_lemmas; otherwise fallback
    std::vector<std::string> result_words(num_words);
    for (int i = 0; i < num_words; ++i) {
        const char* gpu_ptr = h_results[i].first;
        int len = static_cast<int>(h_results[i].second);
        if (gpu_ptr != nullptr && len > 0) {
            ptrdiff_t offset = gpu_ptr - d_lemmas.data();
            if (offset >= 0 && static_cast<size_t>(offset) < h_lemmas.size()) {
                result_words[i] = std::string(h_lemmas.data() + offset, len);
                continue;
            }
        }
        result_words[i] = words[i];  // fallback: original lowercased word
    }

    double throughput = (kernel_ms > 0.f) ? (num_words / (kernel_ms / 1000.0)) : 0.0;
    std::cerr << "Words: " << num_words << "\n"
              << "  Preprocess (I/O+tokenize+lowercase):        " << preprocess_ms << " ms\n"
              << "  Pack (build flat char/offset arrays):       " << pack_ms << " ms\n"
              << "  H2D (upload words+trie):                    " << h2d_ms << " ms\n"
              << "  Kernel:                                     " << kernel_ms << " ms"
              << "  (" << static_cast<long long>(throughput) << " words/sec)\n"
              << "  D2H (download results):                     " << d2h_ms << " ms\n"
              << "  GPU total (H2D+kernel+D2H):                 " << total_ms << " ms\n"
              << "  End-to-end (preprocess+pack+GPU):           " << (preprocess_ms + pack_ms + total_ms) << " ms\n";

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
