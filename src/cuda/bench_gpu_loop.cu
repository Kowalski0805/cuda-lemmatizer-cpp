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
        std::cerr << "Usage: " << argv[0] << " <input_file> [duration_seconds]\n";
        return 1;
    }
    const std::string input_path = argv[1];
    double run_duration_s = (argc >= 3) ? atof(argv[2]) : 10.0;

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

    // Read and tokenize input
    std::ifstream fin(input_path);
    if (!fin) {
        std::cerr << "Cannot open input file: " << input_path << "\n";
        return 1;
    }
    std::vector<std::string> words;
    {
        std::string line;
        while (std::getline(fin, line)) {
            std::istringstream ss(line);
            std::string token;
            while (ss >> token)
                words.push_back(lowercase_ukr(token));
        }
    }
    const int num_words = static_cast<int>(words.size());

    if (num_words == 0) {
        std::cerr << "No words found in input.\n";
        return 1;
    }

    // Build flat char arrays for cuDF strings column
    std::vector<char> h_chars;
    std::vector<int32_t> h_offsets = {0};
    h_chars.reserve(static_cast<size_t>(num_words) * 16);
    for (const auto& w : words) {
        h_chars.insert(h_chars.end(), w.begin(), w.end());
        h_offsets.push_back(static_cast<int32_t>(h_chars.size()));
    }

    // Upload chars to device
    rmm::device_uvector<char> d_chars_raw(h_chars.size(), rmm::cuda_stream_default);
    cudaMemcpy(d_chars_raw.data(), h_chars.data(), h_chars.size(), cudaMemcpyHostToDevice);

    // Build cuDF strings column
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

    // Allocate output (zeroed once, reused each iteration)
    rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> d_output(
        num_words, rmm::cuda_stream_default);
    cudaMemset(d_output.data(), 0,
               static_cast<size_t>(num_words) * sizeof(thrust::pair<const char*, cudf::size_type>));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int threads = 128;
    int blocks = (num_words + threads - 1) / threads;
    int num_iters = 0;
    double total_kernel_ms = 0.0, peak_kernel_ms = 0.0;

    std::cerr << "Running for " << run_duration_s << "s  words=" << num_words
              << "  blocks=" << blocks << "  threads=" << threads << "\n";

    auto wall_start = std::chrono::high_resolution_clock::now();
    while (true) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wall_start).count();
        if (elapsed >= run_duration_s) break;

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

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_kernel_ms += ms;
        if (ms > peak_kernel_ms) peak_kernel_ms = ms;
        ++num_iters;

        if (num_iters % 100 == 0) {
            double tp = (double)num_words * num_iters / (total_kernel_ms / 1000.0);
            fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                    num_iters, total_kernel_ms / num_iters, tp / 1e6);
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    double avg_ms = total_kernel_ms / num_iters;
    double tp = (double)num_words * num_iters / (total_kernel_ms / 1000.0);
    fprintf(stderr, "\n=== Final ===\n");
    fprintf(stderr, "  Iters:       %d\n",    num_iters);
    fprintf(stderr, "  Words/iter:  %d\n",    num_words);
    fprintf(stderr, "  Avg kernel:  %.3f ms\n", avg_ms);
    fprintf(stderr, "  Peak kernel: %.3f ms\n", peak_kernel_ms);
    fprintf(stderr, "  Throughput:  %.2fM words/sec\n", tp / 1e6);

    return 0;
}
