// bench_gpu_opt1_utf8_loop.cu — utf8 trie, pinned/async pipeline, optional kernel loop
//
// Usage: <binary> <input_file> [duration_seconds]
//   No duration → one-shot (same as bench_gpu_opt1_utf8)
//   With duration → loop kernel only for N seconds, report avg/peak/throughput

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
using Clock      = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    auto wall_start = Clock::now();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [duration_seconds]\n";
        return 1;
    }
    const bool loop_mode       = (argc >= 3);
    const double run_duration_s = loop_mode ? atof(argv[2]) : 0.0;

    // Pre-warm CUDA context
    { void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp)); }

    auto t0 = Clock::now();

    // Load trie from disk
    std::vector<GpuState>       h_states;
    std::vector<Utf8Transition> h_transitions;
    std::vector<char>           h_lemmas;
    load_bin_vector("utf8_states.bin",      h_states);
    load_bin_vector("utf8_transitions.bin", h_transitions);
    load_bin_vector("utf8_lemmas.bin",      h_lemmas);
    if (h_states.empty()) {
        std::cerr << "Failed to load trie data. Run from cmake-build-debug/.\n";
        return 1;
    }
    fprintf(stderr, "Trie load:     %lld ms\n",
            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - t0).count());

    // --- PRE-SCAN ---
    t0 = Clock::now();
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
    { const char* p = mapped, *end = mapped + file_sz;
      while (p < end) {
          const char* ws = p;
          while (p < end && *p != '\n' && *p != '\r') ++p;
          int len = (int)(p - ws);
          if (len > 0) { ++pre_N; pre_total_chars += len; }
          while (p < end && (*p == '\n' || *p == '\r')) ++p;
      } }
    if (pre_N == 0) { std::cerr << "No words.\n"; munmap((void*)mapped, file_sz); return 1; }
    fprintf(stderr, "Pre-scan:      %lld ms\n",
            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - t0).count());

    // --- PRE-ALLOCATE ---
    t0 = Clock::now();
    char*       h_chars;
    int32_t*    h_offsets;
    ResultPair* h_out;
    CUDA_CHECK(cudaHostAlloc(&h_chars,   pre_total_chars,                        cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_offsets, (size_t)(pre_N+1) * sizeof(int32_t),    cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out,     (size_t)pre_N     * sizeof(ResultPair), cudaHostAllocDefault));

    rmm::device_uvector<GpuState>       d_states   (h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<Utf8Transition> d_trans    (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>           d_lemmas   (h_lemmas.size(),       rmm::cuda_stream_default);
    rmm::device_uvector<char>           d_chars_raw(pre_total_chars,       rmm::cuda_stream_default);
    rmm::device_uvector<ResultPair>     d_out      (pre_N,                 rmm::cuda_stream_default);
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, pre_N + 1, cudf::mask_state::UNALLOCATED);
    fprintf(stderr, "Pre-alloc:     %lld ms\n",
            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - t0).count());

    // --- STREAMS & EVENTS ---
    cudaStream_t stream_trie, stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_trie));
    CUDA_CHECK(cudaStreamCreate(&stream_data));

    cudaEvent_t ev_trie_start, ev_trie_ready,
                ev_h2d_start,  ev_h2d_done,
                ev_kern_start, ev_kern_done,
                ev_d2h_done;
    for (auto* e : {&ev_trie_start, &ev_trie_ready, &ev_h2d_start, &ev_h2d_done,
                    &ev_kern_start, &ev_kern_done, &ev_d2h_done})
        CUDA_CHECK(cudaEventCreate(e));

    // --- ASYNC TRIE H2D (overlaps pack below) ---
    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(Utf8Transition), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_trie));
    CUDA_CHECK(cudaStreamSynchronize(stream_trie));

    // --- PACK: mmap → pinned ---
    t0 = Clock::now();
    h_offsets[0] = 0;
    size_t pos = 0;
    int N = 0;
    const char* p = mapped, *end_p = mapped + file_sz;
    while (p < end_p) {
        const char* ws = p;
        while (p < end_p && *p != '\n' && *p != '\r') ++p;
        int len = (int)(p - ws);
        if (len > 0) {
            std::memcpy(h_chars + pos, ws, len);
            h_offsets[N+1] = (int32_t)(pos += len);
            ++N;
        }
        while (p < end_p && (*p == '\n' || *p == '\r')) ++p;
    }
    munmap((void*)mapped, file_sz);
    double pack_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    const size_t total_chars = (size_t)h_offsets[N];

    // --- ASYNC DATA H2D ---
    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0, (size_t)N * sizeof(ResultPair), stream_data));
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_chars_raw.data(), h_chars,
        total_chars, cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), h_offsets,
        (size_t)(N+1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream_data));

    auto input_col = cudf::make_strings_column(
        N, std::move(offsets_col),
        rmm::device_buffer{d_chars_raw.data(), total_chars, rmm::cuda_stream_default},
        0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    // Kernel must wait for both H2D streams
    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));
    int threads = 128, blocks = (N + threads - 1) / threads;

    if (loop_mode) {
        // Wait for all H2D to land before looping
        CUDA_CHECK(cudaEventSynchronize(ev_h2d_done));

        float trie_ms_v = 0, h2d_ms_v = 0;
        CUDA_CHECK(cudaEventElapsedTime(&trie_ms_v, ev_trie_start, ev_trie_ready));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms_v,  ev_h2d_start,  ev_h2d_done));
        fprintf(stderr,
                "Words: %d  blocks: %d  threads: %d\n"
                "  Pack (mmap→pinned):                    %.3f ms\n"
                "  Trie H2D (async, overlapped pack):     %.3f ms\n"
                "  Data H2D (pinned→GDDR6X, async):       %.3f ms\n"
                "Running for %.1fs ...\n",
                N, blocks, threads, pack_ms, trie_ms_v, h2d_ms_v, run_duration_s);

        // Reuse two events for the loop (avoids per-iteration alloc)
        cudaEvent_t ev_loop0, ev_loop1;
        CUDA_CHECK(cudaEventCreate(&ev_loop0));
        CUDA_CHECK(cudaEventCreate(&ev_loop1));

        int    num_iters = 0;
        double total_kernel_ms = 0.0, peak_kernel_ms = 0.0;
        auto   wall = Clock::now();

        while (std::chrono::duration<double>(Clock::now() - wall).count() < run_duration_s) {
            CUDA_CHECK(cudaEventRecord(ev_loop0, stream_data));
            utf8_lookup_kernel<<<blocks, threads, 0, stream_data>>>(
                *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
            CUDA_CHECK(cudaEventRecord(ev_loop1, stream_data));
            CUDA_CHECK(cudaEventSynchronize(ev_loop1));

            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_loop0, ev_loop1));
            total_kernel_ms += ms;
            if (ms > peak_kernel_ms) peak_kernel_ms = ms;
            ++num_iters;

            if (num_iters % 100 == 0) {
                double tp = (double)N * num_iters / (total_kernel_ms / 1000.0);
                //fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                //        num_iters, total_kernel_ms / num_iters, tp / 1e6);
            }
        }

        CUDA_CHECK(cudaEventDestroy(ev_loop0));
        CUDA_CHECK(cudaEventDestroy(ev_loop1));

        double avg_ms = total_kernel_ms / num_iters;
        double tp     = (double)N * num_iters / (total_kernel_ms / 1000.0);
        fprintf(stderr, "\n=== Final ===\n"
                "  Iters:       %d\n"
                "  Words/iter:  %d\n"
                "  Avg kernel:  %.3f ms\n"
                "  Peak kernel: %.3f ms\n"
                "  Throughput:  %.2fM words/sec\n",
                num_iters, N, avg_ms, peak_kernel_ms, tp / 1e6);
    } else {
        // --- ONE-SHOT ---
        CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
        utf8_lookup_kernel<<<blocks, threads, 0, stream_data>>>(
            *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
        CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
            (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
        CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));
        CUDA_CHECK(cudaStreamSynchronize(stream_data));

        float trie_ms = 0, h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
        CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_done));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,  ev_d2h_done));

        double tp        = (kern_ms > 0.f) ? (N / (kern_ms / 1000.0)) : 0.0;
        double gpu_total = h2d_ms + kern_ms + d2h_ms;
        fprintf(stderr,
                "Words: %d\n"
                "  Pack (mmap→pinned):                    %.3f ms\n"
                "  Trie H2D (async, overlapped pack):     %.3f ms\n"
                "  Data H2D (pinned→GDDR6X, async):       %.3f ms\n"
                "  Kernel:                                %.3f ms  (%lld words/sec)\n"
                "  D2H (GDDR6X→pinned, async):            %.3f ms\n"
                "  GPU total (H2D+kernel+D2H):            %.3f ms\n"
                "  End-to-end (pack+GPU):                 %.3f ms\n",
                N, pack_ms, trie_ms, h2d_ms, kern_ms, (long long)tp,
                d2h_ms, gpu_total, pack_ms + gpu_total);

        fprintf(stderr, "  Wall time since start:                 %.3f ms\n",
                std::chrono::duration<double, std::milli>(
                    Clock::now() - wall_start).count());
    }

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
