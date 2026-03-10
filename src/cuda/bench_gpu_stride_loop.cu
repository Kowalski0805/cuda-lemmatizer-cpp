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

__global__ void lookup_kernel_stride(
    const char* d_input,
    int num_words,
    const GpuState* states,
    const GpuTransition* transitions,
    const char* lemmas,
    char* d_output);

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

    // --- PREPROCESS TIMER START (file I/O + tokenize + lowercase_ukr) ---
    auto preprocess_start = std::chrono::high_resolution_clock::now();

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

    double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - preprocess_start).count();
    // --- PREPROCESS TIMER END ---

    // --- PACK TIMER START (build fixed-stride buffer) ---
    auto pack_start = std::chrono::high_resolution_clock::now();

    // Pack words into fixed-stride buffer: [num_words * MAX_WORD_LEN], zero-padded
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

    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - pack_start).count();
    // --- PACK TIMER END ---

    // Upload inputs to device
    char* d_input_dev   = nullptr;
    char* d_output_dev  = nullptr;
    GpuState*      d_states      = nullptr;
    GpuTransition* d_transitions = nullptr;
    char*          d_lemmas_dev  = nullptr;

    cudaMalloc(&d_input_dev,    stride_bytes);
    cudaMalloc(&d_output_dev,   stride_bytes);
    cudaMalloc(&d_states,       h_states.size()      * sizeof(GpuState));
    cudaMalloc(&d_transitions,  h_transitions.size() * sizeof(GpuTransition));
    cudaMalloc(&d_lemmas_dev,   h_lemmas.size());

    cudaMemcpy(d_input_dev,   h_input.data(),       stride_bytes,                                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_states,      h_states.data(),      h_states.size() * sizeof(GpuState),           cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas_dev,  h_lemmas.data(),      h_lemmas.size(),                              cudaMemcpyHostToDevice);
    cudaMemset(d_output_dev, 0, stride_bytes);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    const int threads = 128;
    const int blocks  = (num_words + threads - 1) / threads;
    int    num_iters       = 0;
    double total_kernel_ms = 0.0, peak_kernel_ms = 0.0;

    std::cerr << "Preprocess (I/O+tokenize+lowercase):        " << preprocess_ms << " ms\n";
    std::cerr << "Pack (build fixed-stride buffer):           " << pack_ms << " ms\n";
    std::cerr << "Running for " << run_duration_s << "s  words=" << num_words
              << "  blocks=" << blocks << "  threads=" << threads << "\n";

    auto wall_start = std::chrono::high_resolution_clock::now();
    while (true) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wall_start).count();
        if (elapsed >= run_duration_s) break;

        cudaEventRecord(ev_start);
        lookup_kernel_stride<<<blocks, threads>>>(
            d_input_dev, num_words, d_states, d_transitions, d_lemmas_dev, d_output_dev);
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

    cudaFree(d_input_dev);
    cudaFree(d_output_dev);
    cudaFree(d_states);
    cudaFree(d_transitions);
    cudaFree(d_lemmas_dev);

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
