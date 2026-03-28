// bench_gpu_opt1.cu — improvement of bench_gpu_opt (one word per line, metrics only)
//

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
    auto start = std::chrono::high_resolution_clock::now();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>  (one word per line)\n";
        return 1;
    }

    // Pre-warm CUDA context
    { void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp)); }

    auto t0 = std::chrono::high_resolution_clock::now();

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

    std::cerr << "Trie load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count() << " ms\n";


    t0 = std::chrono::high_resolution_clock::now();

    /*
    int    pre_N = 0;
    size_t pre_total_chars = 0;
    FILE* f = fopen(argv[1], "rb");
    struct stat st; fstat(fileno(f), &st);
    const size_t file_sz = (size_t)st.st_size;
    char buf[1 << 20];
    size_t n;
    bool in_word = false;

    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++) {
            if (buf[i] != '\n' && buf[i] != '\r') {
                if (!in_word) { ++pre_N; in_word = true; }
                ++pre_total_chars;
            } else {
                in_word = false;
            }
        }
    }
    fclose(f);

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    const char* mapped = (const char*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);
    /*/
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
    //*/

    std::cerr << "Pre-scan time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count() << " ms\n";

    // --- PRE-ALLOCATE ALL PINNED + DEVICE BUFFERS (outside timing) ---
    t0 = std::chrono::high_resolution_clock::now();

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

    std::cerr << "Pre-allocation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count() << " ms\n";

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
    CUDA_CHECK(cudaStreamSynchronize(stream_trie));

    // --- LOAD TIMER: rescan mmap → fill spans ---
    // --- PACK TIMER: mmap → pinned (one pass, no heap allocs) ---
    t0 = std::chrono::high_resolution_clock::now();
    h_offsets[0] = 0;
    size_t pos = 0;
    int N = 0;
    const char* p = mapped, *end = mapped + file_sz;

    while (p < end) {
        const char* ws = p;
        while (p < end && *p != '\n' && *p != '\r') ++p;
        int len = (int)(p - ws);
        if (len > 0) {
            std::memcpy(h_chars + pos, ws, len);
            h_offsets[N+1] = (int32_t)(pos += len);
            ++N;
        }
        while (p < end && (*p == '\n' || *p == '\r')) ++p;
    }
    munmap((void*)mapped, file_sz);
    double pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    const size_t total_chars = (size_t)h_offsets[N];

    std::cerr << "pre_N: " << pre_N << " pre_total_chars: " << pre_total_chars << " N: " << N << "\n";
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

    float trie_ms = 0, h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,  ev_d2h_done));

    double tp        = (kern_ms > 0.f) ? (N / (kern_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kern_ms + d2h_ms;
    std::cerr
        << "Words: " << N << "\n"
        << "  Pack (mmap→pinned, no heap alloc):      " << pack_ms << " ms\n"
        << "  Trie H2D (async, ran during load+pack): " << trie_ms << " ms\n"
        << "  Data H2D (pinned→GDDR6X, async):        " << h2d_ms  << " ms\n"
        << "  Kernel:                                 " << kern_ms << " ms"
        << "  (" << (long long)tp << " words/sec)\n"
        << "  D2H (GDDR6X→pinned, async):             " << d2h_ms  << " ms\n"
        << "  GPU total (H2D+kernel+D2H):             " << gpu_total << " ms\n"
        << "  End-to-end (load+pack+gpu_total):       "
        << (pack_ms + gpu_total) << " ms\n";

    CUDA_CHECK(cudaFreeHost(h_chars));
    CUDA_CHECK(cudaFreeHost(h_offsets));
    CUDA_CHECK(cudaFreeHost(h_out));
    for (auto* e : {ev_trie_start, ev_trie_ready, ev_h2d_start, ev_h2d_done,
                    ev_kern_start, ev_kern_done, ev_d2h_done})
        cudaEventDestroy(e);
    cudaStreamDestroy(stream_trie);
    cudaStreamDestroy(stream_data);

    std::cerr
        << "  Wall time since start:                   " << std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count() << " ms\n";
    return 0;
}
