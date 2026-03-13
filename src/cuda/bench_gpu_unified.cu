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
#include <rmm/mr/device/managed_memory_resource.hpp>

#include "structs.h"
#include "trie.h"
#include "icu_lowercase.h"
#include "lemmatizer_kernel.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    // All RMM/cuDF allocations will use managed (unified) memory
    rmm::mr::managed_memory_resource managed_mr;
    rmm::mr::set_current_device_resource(&managed_mr);

    // Load trie from disk into host vectors
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

    // --- PREPROCESS (I/O + tokenize + lowercase) ---
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    std::vector<std::string> lines;
    { std::string ln; while (std::getline(fin, ln)) lines.push_back(std::move(ln)); }

    std::vector<int> line_wc(lines.size(), 0);
    std::vector<std::string> words;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::istringstream ss(lines[i]);
        std::string tok;
        while (ss >> tok) { words.push_back(lowercase_ukr(tok)); ++line_wc[i]; }
    }
    const int N = (int)words.size();
    double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    if (N == 0) { std::cerr << "No words.\n"; return 1; }

    // --- PACK (write directly into managed memory — no host intermediate, no memcpy) ---
    t0 = std::chrono::high_resolution_clock::now();

    // offsets column: allocated by cuDF using managed MR, written from CPU
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, N + 1, cudf::mask_state::UNALLOCATED);
    int32_t* off = offsets_col->mutable_view().data<int32_t>();

    size_t total_chars = 0;
    for (const auto& w : words) total_chars += w.size();

    // chars buffer: allocated via managed MR, written from CPU
    rmm::device_buffer chars_buf(total_chars, rmm::cuda_stream_default);
    char* chr = static_cast<char*>(chars_buf.data());

    off[0] = 0;
    size_t pos = 0;
    for (int i = 0; i < N; ++i) {
        const auto& w = words[i];
        std::memcpy(chr + pos, w.data(), w.size());
        pos += w.size();
        off[i + 1] = (int32_t)pos;
    }

    // Save raw pointers before ownership is moved into cuDF column
    const char*    raw_chr = chr;
    const int32_t* raw_off = off;

    // Build cuDF strings column — moves managed buffers, no copy
    auto input_col = cudf::make_strings_column(
        N, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    // Trie in managed memory: CPU memcpy into managed device_uvectors
    rmm::device_uvector<GpuState>      d_states(h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas(h_lemmas.size(),       rmm::cuda_stream_default);
    std::memcpy(d_states.data(), h_states.data(), h_states.size() * sizeof(GpuState));
    std::memcpy(d_trans.data(),  h_transitions.data(), h_transitions.size() * sizeof(GpuTransition));
    std::memcpy(d_lemmas.data(), h_lemmas.data(), h_lemmas.size());

    // Output buffer in managed memory
    using ResultPair = thrust::pair<const char*, cudf::size_type>;
    rmm::device_uvector<ResultPair> d_out(N, rmm::cuda_stream_default);
    cudaMemset(d_out.data(), 0, (size_t)N * sizeof(ResultPair));

    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // --- PREFETCH input + trie to GPU (≈ H2D) ---
    int dev; cudaGetDevice(&dev);
    t0 = std::chrono::high_resolution_clock::now();
    cudaMemPrefetchAsync(raw_chr,         total_chars,                                   dev);
    cudaMemPrefetchAsync(raw_off,         (size_t)(N + 1) * sizeof(int32_t),             dev);
    cudaMemPrefetchAsync(d_states.data(), h_states.size()      * sizeof(GpuState),       dev);
    cudaMemPrefetchAsync(d_trans.data(),  h_transitions.size() * sizeof(GpuTransition),  dev);
    cudaMemPrefetchAsync(d_lemmas.data(), h_lemmas.size(),                                dev);
    cudaDeviceSynchronize();
    double prefetch_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // --- KERNEL ---
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    int threads = 128, blocks = (N + threads - 1) / threads;
    cudaEventRecord(ev0);
    lookup_kernel<<<blocks, threads>>>(
        *d_input_view, N,
        d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    // --- PREFETCH result to CPU (≈ D2H) ---
    t0 = std::chrono::high_resolution_clock::now();
    cudaMemPrefetchAsync(d_out.data(), (size_t)N * sizeof(ResultPair), cudaCpuDeviceId);
    cudaDeviceSynchronize();
    double d2h_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = prefetch_ms + kernel_ms + d2h_ms;
    std::cerr << "Words: " << N << "\n"
              << "  Preprocess (I/O+tokenize+lowercase):  " << preprocess_ms << " ms\n"
              << "  Pack (write into managed memory):     " << pack_ms       << " ms\n"
              << "  Prefetch to GPU (≈H2D):               " << prefetch_ms   << " ms\n"
              << "  Kernel:                               " << kernel_ms     << " ms"
              << "  (" << (long long)tp << " words/sec)\n"
              << "  Prefetch to CPU (≈D2H):               " << d2h_ms        << " ms\n"
              << "  GPU total (prefetch+kernel+prefetch): " << gpu_total      << " ms\n"
              << "  End-to-end:                           "
              << (preprocess_ms + pack_ms + gpu_total) << " ms\n";

    return 0;
}
