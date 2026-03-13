// bench_gpu_opt.cu — Optimized GPU benchmark (one word per line input, metrics only)
//
// Optimizations vs bench_gpu_raw:
//   - Pre-warmed CUDA context (eliminates 190ms first-cudaMalloc overhead)
//   - mmap file I/O (replaces getline + std::string per line)
//   - Pinned host memory (cudaHostAlloc) for true async DMA on H2D and D2H
//   - Async trie H2D on stream_trie overlapping with file load + pack
//   - Async data H2D on stream_data after pack (pinned → PCIe DMA)
//   - Event-based sync between streams (no cudaDeviceSynchronize)
//   - Async D2H to pinned output buffer
//   - Stream-level sync only (cudaStreamSynchronize, not DeviceSynchronize)

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>

#include "structs.h"
#include "trie.h"
#include "lemmatizer_kernel.cuh"

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

using ResultPair = thrust::pair<const char*, cudf::size_type>;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>  (one word per line)\n";
        return 1;
    }

    // Pre-warm CUDA context (eliminates first-cudaMalloc 190ms overhead)
    { void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp)); }

    // Load trie from disk into host vectors
    std::vector<GpuState>      h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char>          h_lemmas;
    load_bin_vector("gpu_states.bin",      h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin",      h_lemmas);
    if (h_states.empty()) {
        std::cerr << "Failed to load trie data. Run from cmake-build-debug/.\n";
        return 1;
    }

    // Create two streams: trie upload runs concurrently with CPU file I/O + pack
    cudaStream_t stream_trie, stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_trie));
    CUDA_CHECK(cudaStreamCreate(&stream_data));

    // Allocate device trie buffers (synchronous allocation, no data yet)
    rmm::device_uvector<GpuState>      d_states(h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas(h_lemmas.size(),       rmm::cuda_stream_default);

    // --- START ASYNC TRIE H2D (stream_trie) — overlaps with file I/O + pack below ---
    cudaEvent_t ev_trie_start, ev_trie_ready;
    CUDA_CHECK(cudaEventCreate(&ev_trie_start));
    CUDA_CHECK(cudaEventCreate(&ev_trie_ready));
    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_trie));
    // GPU trie upload is now running in background; CPU continues below

    // --- LOAD: mmap + scan for word spans (no per-word heap allocation) ---
    auto t0 = std::chrono::high_resolution_clock::now();

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    struct stat st;
    fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    const char* mapped = (const char*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);

    std::vector<std::pair<const char*, int>> spans;
    size_t total_chars = 0;
    {
        const char* p = mapped, *end = mapped + file_sz;
        while (p < end) {
            const char* ws = p;
            while (p < end && *p != '\n' && *p != '\r') ++p;
            int len = (int)(p - ws);
            if (len > 0) { spans.push_back({ws, len}); total_chars += len; }
            while (p < end && (*p == '\n' || *p == '\r')) ++p;
        }
    }
    const int N = (int)spans.size();

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    if (N == 0) { std::cerr << "No words.\n"; munmap((void*)mapped, file_sz); return 1; }

    // --- PACK: mmap → pinned host buffers (no intermediate std::vector) ---
    t0 = std::chrono::high_resolution_clock::now();

    char*    h_chars;
    int32_t* h_offsets;
    CUDA_CHECK(cudaHostAlloc(&h_chars,   total_chars,                     cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_offsets, (size_t)(N+1) * sizeof(int32_t), cudaHostAllocDefault));

    h_offsets[0] = 0;
    size_t pos = 0;
    for (int i = 0; i < N; ++i) {
        auto [ws, len] = spans[i];
        std::memcpy(h_chars + pos, ws, len);
        pos += len;
        h_offsets[i+1] = (int32_t)pos;
    }
    munmap((void*)mapped, file_sz);  // done with mmap

    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Allocate device input buffers (synchronous allocation, filled via async H2D below)
    rmm::device_buffer d_chars_buf(total_chars, rmm::cuda_stream_default);
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, N+1, cudf::mask_state::UNALLOCATED);

    // Allocate device output
    rmm::device_uvector<ResultPair> d_out(N, rmm::cuda_stream_default);
    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0, (size_t)N * sizeof(ResultPair), stream_data));

    // --- ASYNC DATA H2D on stream_data (pinned → true DMA, no staging copy) ---
    cudaEvent_t ev_data_h2d_start, ev_data_h2d_done;
    CUDA_CHECK(cudaEventCreate(&ev_data_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_data_h2d_done));
    CUDA_CHECK(cudaEventRecord(ev_data_h2d_start, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_chars_buf.data(), h_chars,
        total_chars, cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), h_offsets,
        (size_t)(N+1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_data_h2d_done, stream_data));

    // Build cuDF strings column (pointer wrapping only, no data access)
    auto input_col = cudf::make_strings_column(
        N, std::move(offsets_col), std::move(d_chars_buf), 0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    // Kernel on stream_data depends on both: data H2D (stream ordering) and trie (event)
    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));

    // --- KERNEL ---
    cudaEvent_t ev_kernel_start, ev_kernel_done;
    CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
    int threads = 128, blocks = (N + threads - 1) / threads;
    CUDA_CHECK(cudaEventRecord(ev_kernel_start, stream_data));
    lookup_kernel<<<blocks, threads, 0, stream_data>>>(
        *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
    CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream_data));

    // --- ASYNC D2H to pinned output (no staging, full PCIe bandwidth) ---
    ResultPair* h_out;
    CUDA_CHECK(cudaHostAlloc(&h_out, (size_t)N * sizeof(ResultPair), cudaHostAllocDefault));
    cudaEvent_t ev_d2h_done;
    CUDA_CHECK(cudaEventCreate(&ev_d2h_done));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
        (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));

    // Wait for completion — stream-level sync, not DeviceSynchronize
    CUDA_CHECK(cudaStreamSynchronize(stream_data));
    CUDA_CHECK(cudaStreamSynchronize(stream_trie));

    // Gather GPU-side timings via events
    float trie_h2d_ms = 0, data_h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_h2d_ms, ev_trie_start,     ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&data_h2d_ms, ev_data_h2d_start, ev_data_h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,   ev_kernel_start,   ev_kernel_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,      ev_kernel_done,    ev_d2h_done));

    double tp        = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = data_h2d_ms + kernel_ms + d2h_ms;
    std::cerr
        << "Words: " << N << "\n"
        << "  Load (mmap + find spans):             " << load_ms      << " ms\n"
        << "  Pack (mmap→pinned, no copy):          " << pack_ms      << " ms\n"
        << "  Trie H2D (async, ran during load+pack): " << trie_h2d_ms << " ms\n"
        << "  Data H2D (pinned async):              " << data_h2d_ms  << " ms\n"
        << "  Kernel:                               " << kernel_ms    << " ms"
        << "  (" << (long long)tp << " words/sec)\n"
        << "  D2H (pinned async):                   " << d2h_ms       << " ms\n"
        << "  GPU total (data_h2d+kernel+D2H):      " << gpu_total     << " ms\n"
        << "  End-to-end (load+pack+gpu_total):     "
        << (load_ms + pack_ms + gpu_total) << " ms\n";

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_chars));
    CUDA_CHECK(cudaFreeHost(h_offsets));
    CUDA_CHECK(cudaFreeHost(h_out));
    cudaEventDestroy(ev_trie_start);    cudaEventDestroy(ev_trie_ready);
    cudaEventDestroy(ev_data_h2d_start); cudaEventDestroy(ev_data_h2d_done);
    cudaEventDestroy(ev_kernel_start);   cudaEventDestroy(ev_kernel_done);
    cudaEventDestroy(ev_d2h_done);
    cudaStreamDestroy(stream_trie);
    cudaStreamDestroy(stream_data);

    return 0;
}
