#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>

#include "structs.h"
#include "trie.h"
#include "lemmatizer_kernel.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>  (one word per line)\n";
        return 1;
    }

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

    // --- LOAD (file I/O only, one word per line, no tokenization/lowercase) ---
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    std::vector<std::string> words;
    { std::string ln; while (std::getline(fin, ln)) if (!ln.empty()) words.push_back(std::move(ln)); }
    const int N = (int)words.size();

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (N == 0) { std::cerr << "No words.\n"; return 1; }

    // --- PACK (build flat char/offset arrays) ---
    t0 = std::chrono::high_resolution_clock::now();

    std::vector<char> h_chars;
    std::vector<int32_t> h_offsets = {0};
    h_chars.reserve((size_t)N * 16);
    for (const auto& w : words) {
        h_chars.insert(h_chars.end(), w.begin(), w.end());
        h_offsets.push_back((int32_t)h_chars.size());
    }

    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // --- H2D ---
    auto h2d_start = std::chrono::high_resolution_clock::now();

    rmm::device_uvector<char> d_chars_raw(h_chars.size(), rmm::cuda_stream_default);
    cudaMemcpy(d_chars_raw.data(), h_chars.data(), h_chars.size(), cudaMemcpyHostToDevice);

    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        (cudf::size_type)h_offsets.size(), cudf::mask_state::UNALLOCATED);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
               h_offsets.data(), h_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto input_col = cudf::make_strings_column(
        (cudf::size_type)N, std::move(offsets_col),
        rmm::device_buffer{d_chars_raw.data(), h_chars.size(), rmm::cuda_stream_default},
        0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    rmm::device_uvector<GpuState>      d_states(h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas(h_lemmas.size(),       rmm::cuda_stream_default);
    cudaMemcpy(d_states.data(), h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans.data(),  h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas.data(), h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    using ResultPair = thrust::pair<const char*, cudf::size_type>;
    rmm::device_uvector<ResultPair> d_out(N, rmm::cuda_stream_default);
    cudaMemset(d_out.data(), 0, (size_t)N * sizeof(ResultPair));

    cudaDeviceSynchronize();
    double h2d_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - h2d_start).count();

    // --- KERNEL ---
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    int threads = 128, blocks = (N + threads - 1) / threads;
    cudaEventRecord(ev0);
    lookup_kernel<<<blocks, threads>>>(
        *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    // --- D2H (just the output pairs, no decode) ---
    auto d2h_start = std::chrono::high_resolution_clock::now();
    std::vector<ResultPair> h_out(N);
    cudaMemcpy(h_out.data(), d_out.data(), (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost);
    double d2h_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - d2h_start).count();

    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kernel_ms + d2h_ms;
    std::cerr << "Words: " << N << "\n"
              << "  Load (file I/O, one word/line):        " << load_ms   << " ms\n"
              << "  Pack (build flat char/offset arrays):  " << pack_ms   << " ms\n"
              << "  H2D (upload words+trie):               " << h2d_ms    << " ms\n"
              << "  Kernel:                                " << kernel_ms << " ms"
              << "  (" << (long long)tp << " words/sec)\n"
              << "  D2H (download results):                " << d2h_ms    << " ms\n"
              << "  GPU total (H2D+kernel+D2H):            " << gpu_total  << " ms\n"
              << "  End-to-end (load+pack+GPU):            " << (load_ms + pack_ms + gpu_total) << " ms\n";

    return 0;
}
