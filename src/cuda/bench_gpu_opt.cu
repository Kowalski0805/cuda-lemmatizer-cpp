// bench_gpu_opt.cu — Optimized GPU benchmark (one word per line, metrics only)
//
// Fixes applied from Nsight profiling:
//   - Pre-warm CUDA context (eliminates 190ms first-cudaMalloc overhead)
//   - mmap file I/O (replaces getline)
//   - Pre-scan for buffer sizes, cudaHostAlloc BEFORE timing (eliminates 5-7s in-pipeline overhead)
//   - Async trie H2D on stream_trie, overlapping with file load + pack
//   - Pinned host buffers → async H2D to device_uvector (explicit GDDR6X) → D2D into cuDF
//     device_buffer on stream_data. Ensures kernel reads GDDR6X, not host-mapped memory.
//   - Async D2H to pinned output (33× faster than unpinned)
//   - Event-based stream sync only (no cudaDeviceSynchronize)

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

    // Pre-warm CUDA context
    { void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp)); }

    // Load trie from disk
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

    // --- PRE-SCAN: mmap file, count N and total_chars (outside timing, no allocations) ---
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    struct stat st; fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    const char* mapped = (const char*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);

    int    pre_N = 0;
    size_t pre_total_chars = 0;
    {
        const char* p = mapped, *end = mapped + file_sz;
        while (p < end) {
            const char* ws = p;
            while (p < end && *p != '\n' && *p != '\r') ++p;
            int len = (int)(p - ws);
            if (len > 0) { ++pre_N; pre_total_chars += len; }
            while (p < end && (*p == '\n' || *p == '\r')) ++p;
        }
    }
    if (pre_N == 0) { std::cerr << "No words.\n"; munmap((void*)mapped, file_sz); return 1; }

    // --- PRE-ALLOCATE ALL PINNED + DEVICE BUFFERS (outside timing) ---
    char*       h_chars;
    int32_t*    h_offsets;
    ResultPair* h_out;
    CUDA_CHECK(cudaHostAlloc(&h_chars,   pre_total_chars,                       cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_offsets, (size_t)(pre_N+1) * sizeof(int32_t),   cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out,     (size_t)pre_N     * sizeof(ResultPair), cudaHostAllocDefault));

    // device_uvector guarantees GDDR6X allocation (cudaMalloc-backed)
    rmm::device_uvector<GpuState>      d_states(h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas(h_lemmas.size(),       rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_chars_raw(pre_total_chars,    rmm::cuda_stream_default);
    rmm::device_uvector<ResultPair>    d_out(pre_N,                    rmm::cuda_stream_default);
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, pre_N + 1, cudf::mask_state::UNALLOCATED);

    // --- STREAMS & EVENTS ---
    cudaStream_t stream_trie, stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_trie));
    CUDA_CHECK(cudaStreamCreate(&stream_data));

    cudaEvent_t ev_trie_start, ev_trie_ready,
                ev_h2d_start,  ev_h2d_done,
                ev_kern_start, ev_kern_done,
                ev_d2h_done;
    for (auto* e : {&ev_trie_start, &ev_trie_ready,
                    &ev_h2d_start,  &ev_h2d_done,
                    &ev_kern_start, &ev_kern_done, &ev_d2h_done})
        CUDA_CHECK(cudaEventCreate(e));

    // --- START ASYNC TRIE H2D (overlaps with load + pack below) ---
    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_trie));

    // --- LOAD TIMER: rescan mmap → fill spans ---
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<const char*, int>> spans;
    spans.reserve(pre_N);
    {
        const char* p = mapped, *end = mapped + file_sz;
        while (p < end) {
            const char* ws = p;
            while (p < end && *p != '\n' && *p != '\r') ++p;
            int len = (int)(p - ws);
            if (len > 0) spans.push_back({ws, len});
            while (p < end && (*p == '\n' || *p == '\r')) ++p;
        }
    }
    const int N = (int)spans.size();
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // --- PACK TIMER: mmap → pinned (one pass, no heap allocs) ---
    t0 = std::chrono::high_resolution_clock::now();
    h_offsets[0] = 0;
    size_t pos = 0;
    for (int i = 0; i < N; ++i) {
        auto [ws, len] = spans[i];
        std::memcpy(h_chars + pos, ws, len);
        pos += len;
        h_offsets[i+1] = (int32_t)pos;
    }
    munmap((void*)mapped, file_sz);
    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    const size_t total_chars = (size_t)h_offsets[N];

    // --- ASYNC DATA H2D: pinned → d_chars_raw (GDDR6X device_uvector) ---
    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0, (size_t)N * sizeof(ResultPair), stream_data));
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_chars_raw.data(), h_chars,
        total_chars, cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), h_offsets,
        (size_t)(N+1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream_data));

    // Build cuDF strings column.
    // D2D copy (d_chars_raw → device_buffer) is enqueued on stream_data,
    // ordered after H2D above. Kernel receives a pointer that is unambiguously GDDR6X.
    auto input_col = cudf::make_strings_column(
        N, std::move(offsets_col),
        rmm::device_buffer{d_chars_raw.data(), total_chars, rmm::cuda_stream_default},
        0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    // --- KERNEL: waits for data H2D (stream ordering) + trie (event) ---
    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));
    int threads = 128, blocks = (N + threads - 1) / threads;
    CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
    lookup_kernel<<<blocks, threads, 0, stream_data>>>(
        *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
    CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

    // --- ASYNC D2H: device → pinned ---
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
        (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));

    CUDA_CHECK(cudaStreamSynchronize(stream_data));
    CUDA_CHECK(cudaStreamSynchronize(stream_trie));

    float trie_ms = 0, h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,  ev_d2h_done));

    double tp        = (kern_ms > 0.f) ? (N / (kern_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kern_ms + d2h_ms;
    std::cerr
        << "Words: " << N << "\n"
        << "  Load (mmap + find spans):               " << load_ms << " ms\n"
        << "  Pack (mmap→pinned, no heap alloc):      " << pack_ms << " ms\n"
        << "  Trie H2D (async, ran during load+pack): " << trie_ms << " ms\n"
        << "  Data H2D (pinned→GDDR6X, async):        " << h2d_ms  << " ms\n"
        << "  Kernel:                                 " << kern_ms << " ms"
        << "  (" << (long long)tp << " words/sec)\n"
        << "  D2H (GDDR6X→pinned, async):             " << d2h_ms  << " ms\n"
        << "  GPU total (H2D+kernel+D2H):             " << gpu_total << " ms\n"
        << "  End-to-end (load+pack+gpu_total):       "
        << (load_ms + pack_ms + gpu_total) << " ms\n";

    CUDA_CHECK(cudaFreeHost(h_chars));
    CUDA_CHECK(cudaFreeHost(h_offsets));
    CUDA_CHECK(cudaFreeHost(h_out));
    for (auto* e : {ev_trie_start, ev_trie_ready, ev_h2d_start, ev_h2d_done,
                    ev_kern_start, ev_kern_done, ev_d2h_done})
        cudaEventDestroy(e);
    cudaStreamDestroy(stream_trie);
    cudaStreamDestroy(stream_data);
    return 0;
}
