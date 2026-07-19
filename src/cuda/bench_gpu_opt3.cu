// bench_gpu_opt3.cu
// Eliminates pre-scan, pack pass, cudaHostAlloc.
// Chunked GPU newline scan — chunk_sz tunable, fits any VRAM budget.

#include <chrono>
#include <iostream>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>

#include "structs.h"
#include "trie.h"
#include "lemmatizer_kernel.cuh"
#include "newline_scan.cuh"

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUDA_MEM_CHECK(label) do { \
    size_t _free, _total; \
    cudaMemGetInfo(&_free, &_total); \
    std::cerr << "[MEM] " << label \
    << "  free=" << _free/1024/1024 << "MB" \
    << "  used=" << (_total-_free)/1024/1024 << "MB" \
    << "  total=" << _total/1024/1024 << "MB\n"; \
} while(0)

using ResultPair = thrust::pair<const char*, cudf::size_type>;

// 512 MB chunks — uses ~6 GB peak VRAM, leaves room for trie + output
static constexpr size_t CHUNK_SZ = 512ULL << 20;
static constexpr int    THREADS  = 256;

// Helper: run CUB ExclusiveSum on a chunk, reusing a scratch buffer
template<typename InT, typename OutT>
static void exclusive_sum(
    void* d_tmp, size_t tmp_sz,
    InT* in, OutT* out, int n, cudaStream_t stream)
{
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_sz, in, out, n, stream);
}

int main(int argc, char* argv[]) {
    auto t_wall = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    { void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp)); }

    // ── Trie load ─────────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<GpuState>      h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char>          h_lemmas;
    load_bin_vector("gpu_states.bin",      h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin",      h_lemmas);
    if (h_states.empty()) {
        std::cerr << "Failed to load trie.\n"; return 1;
    }

    rmm::device_uvector<GpuState>      d_states(h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas_dev(h_lemmas.size(),   rmm::cuda_stream_default);

    std::cerr << "Trie load:        "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - t0).count()
              << " ms\n";

    // ── mmap + pin ────────────────────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();

    CUDA_MEM_CHECK("Before mmap");
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }
    struct stat st; fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    char* mapped = (char*)mmap(nullptr, file_sz, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    madvise(mapped, file_sz, MADV_SEQUENTIAL);
    CUDA_MEM_CHECK("After mmap");
    // std::cerr << "file_sz: " << file_sz << "\n";
    // CUDA_CHECK(cudaHostRegister(mapped, file_sz, cudaHostRegisterDefault));

    std::vector<size_t> chunk_starts;
    chunk_starts.push_back(0);
    size_t pos = 0;
    while (pos < file_sz) {
        pos = std::min(pos + CHUNK_SZ, file_sz);
        if (pos < file_sz) {
            // walk back to previous newline
            while (pos > chunk_starts.back() && mapped[pos] != '\n' && mapped[pos] != '\r')
                --pos;
            ++pos; // start after the newline
        }
        chunk_starts.push_back(pos);
    }
    const size_t n_chunks = chunk_starts.size() - 1;

    std::cerr << "mmap+register:    "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - t0).count()
              << " ms\n";

    // ── Streams ───────────────────────────────────────────────────────────────
    cudaStream_t stream_trie, stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_trie));
    CUDA_CHECK(cudaStreamCreate(&stream_data));

    // ── Async trie H2D (runs while scan chunks execute) ───────────────────────
    cudaEvent_t ev_trie_start, ev_trie_ready;
    CUDA_CHECK(cudaEventCreate(&ev_trie_start));
    CUDA_CHECK(cudaEventCreate(&ev_trie_ready));

    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas_dev.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_trie));

    // ── Per-chunk device buffers (allocated once, reused) ─────────────────────
    rmm::device_uvector<char>    d_raw   (CHUNK_SZ,     stream_data);
    rmm::device_uvector<uint8_t> d_flags (CHUNK_SZ,     stream_data);
    rmm::device_uvector<uint32_t>d_wlen  (CHUNK_SZ,     stream_data);
    rmm::device_uvector<uint32_t> d_widx  (CHUNK_SZ,     stream_data);
    rmm::device_uvector<uint32_t> d_coff  (CHUNK_SZ,     stream_data);

    // CUB scratch — size for CHUNK_SZ elements
    size_t cub_tmp_sz = 0;
    {
        // query scratch size (use widx/coff as dummy in/out)
        cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_sz,
            d_flags.data(), d_widx.data(), (int)CHUNK_SZ, stream_data);
    }
    rmm::device_uvector<char> d_cub_tmp(cub_tmp_sz, stream_data);

    // ── Scan pass: accumulate N and total_chars per chunk ─────────────────────
    t0 = std::chrono::high_resolution_clock::now();

    std::vector<int32_t> chunk_N(n_chunks);        // words per chunk
    std::vector<uint32_t> chunk_chars(n_chunks);    // chars per chunk

    // device scalars for reduce results
    rmm::device_uvector<int32_t> d_scalar(1, stream_data);
    rmm::device_uvector<uint32_t> d_scalar_u(1, stream_data);

    char prev_byte = -1; // sentinel: treat start-of-file as separator

    for (size_t c = 0; c < n_chunks; ++c) {
        const size_t off = chunk_starts[c];
        const size_t csz = chunk_starts[c + 1] - chunk_starts[c];
        const int    n   = (int)csz;
        const int    blk = (n + THREADS - 1) / THREADS;

        CUDA_CHECK(cudaMemcpyAsync(d_raw.data(), mapped + off, csz,
            cudaMemcpyHostToDevice, stream_data));

        scan_mark_words<<<blk, THREADS, 0, stream_data>>>(
            d_raw.data(), csz, prev_byte, d_flags.data(), d_wlen.data());

        // word count = sum(flags)
        {
            size_t tmp = cub_tmp_sz;
            cub::DeviceReduce::Sum(d_cub_tmp.data(), tmp,
                d_flags.data(), d_scalar.data(), n, stream_data);
            CUDA_CHECK(cudaStreamSynchronize(stream_data));
            CUDA_CHECK(cudaMemcpy(&chunk_N[c], d_scalar.data(),
                sizeof(int32_t), cudaMemcpyDeviceToHost));
        }

        // total chars = sum(wlen)
        {
            size_t tmp = cub_tmp_sz;
            cub::DeviceReduce::Sum(d_cub_tmp.data(), tmp,
            d_wlen.data(), d_scalar_u.data(), n, stream_data);
            CUDA_CHECK(cudaStreamSynchronize(stream_data));
            CUDA_CHECK(cudaMemcpy(&chunk_chars[c], d_scalar_u.data(),
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }

        prev_byte = '\n';
    }

    d_scalar   = rmm::device_uvector<int32_t> (0, stream_data);
    d_scalar_u = rmm::device_uvector<uint32_t>(0, stream_data);

    double scan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // prefix sums over chunks → base offsets for compact pass
    std::vector<uint32_t> chunk_word_base(n_chunks + 1, 0);
    std::vector<uint32_t> chunk_char_base(n_chunks + 1, 0);
    for (size_t c = 0; c < n_chunks; ++c) {
        chunk_word_base[c + 1] = chunk_word_base[c] + chunk_N[c];
        chunk_char_base[c + 1] = chunk_char_base[c] + chunk_chars[c];
    }

    const uint32_t total_N     = chunk_word_base[n_chunks];
    const uint32_t total_chars = chunk_char_base[n_chunks];

    std::cerr << "Scan pass:        " << scan_ms << " ms"
              << "  N=" << total_N << "  chars=" << total_chars << "\n";

    // ── Allocate final output buffers ─────────────────────────────────────────
    rmm::device_buffer    d_chars  (total_chars,     stream_data);
    rmm::device_uvector<uint32_t> d_off_buf(total_N + 1,     stream_data);

    // ── Compact pass: fill d_chars + d_off_buf ────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    prev_byte = -1;

    for (size_t c = 0; c < n_chunks; ++c) {
        const size_t off = chunk_starts[c];
        const size_t csz = chunk_starts[c + 1] - chunk_starts[c];
        const int    n   = (int)csz;
        const int    blk = (n + THREADS - 1) / THREADS;

        CUDA_CHECK(cudaMemcpyAsync(d_raw.data(), mapped + off, csz,
            cudaMemcpyHostToDevice, stream_data));

        scan_mark_words<<<blk, THREADS, 0, stream_data>>>(
            d_raw.data(), csz, prev_byte, d_flags.data(), d_wlen.data());

        // ExclusiveSum(flags) → d_widx
        {
            size_t tmp = cub_tmp_sz;
            cub::DeviceScan::ExclusiveSum(d_cub_tmp.data(), tmp,
                d_flags.data(), d_widx.data(), n, stream_data);
        }
        // ExclusiveSum(wlen) → d_coff
        {
            size_t tmp = cub_tmp_sz;
            cub::DeviceScan::ExclusiveSum(d_cub_tmp.data(), tmp,
                d_wlen.data(), d_coff.data(), n, stream_data);
        }

        scan_compact<<<blk, THREADS, 0, stream_data>>>(
            d_raw.data(), csz,
            d_flags.data(), d_wlen.data(),
            d_widx.data(), d_coff.data(),
            chunk_word_base[c], chunk_char_base[c],
            (char*)d_chars.data(), d_off_buf.data());

        prev_byte = '\n';
    }

    scan_write_sentinel<<<1, 1, 0, stream_data>>>(
        d_off_buf.data(), total_N, total_chars);

    double compact_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Free chunk buffers
    d_raw    = rmm::device_uvector<char>    (0, stream_data);
    d_flags  = rmm::device_uvector<uint8_t> (0, stream_data);
    d_wlen   = rmm::device_uvector<uint32_t>(0, stream_data);
    d_widx   = rmm::device_uvector<uint32_t>(0, stream_data);
    d_coff   = rmm::device_uvector<uint32_t>(0, stream_data);
    d_cub_tmp= rmm::device_uvector<char>    (0, stream_data);

    CUDA_MEM_CHECK("after freeing chunk buffers");

    // Now allocate output
    rmm::device_uvector<ResultPair> d_out(total_N, stream_data);
    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0,
        (size_t)total_N * sizeof(ResultPair), stream_data));

    CUDA_CHECK(cudaStreamSynchronize(stream_data));
    std::cerr << "Compact pass:     " << compact_ms << " ms\n";

    // ── Build cuDF strings column (all on device, no host involvement) ────────
    // auto offsets_col = cudf::make_numeric_column(
    //     cudf::data_type{cudf::type_id::INT32}, total_N + 1,
    //     cudf::mask_state::UNALLOCATED);
    // CUDA_CHECK(cudaMemcpyAsync(
    //     offsets_col->mutable_view().data<int32_t>(),
    //     d_off_buf.data(),
    //     (size_t)(total_N + 1) * sizeof(uint32_t),
    //     cudaMemcpyDeviceToDevice, stream_data));
    // CUDA_CHECK(cudaStreamSynchronize(stream_data));


    // d_off_buf = rmm::device_uvector<uint32_t>(0, stream_data);  // free 664MB
    // auto input_col = cudf::make_strings_column(
    //     total_N, std::move(offsets_col),
    //     std::move(d_chars),
    //     0, rmm::device_buffer{});
    //
    // auto d_input_view = cudf::column_device_view::create(input_col->view());

    // ── Lookup kernel ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));

    cudaEvent_t ev_kern_start, ev_kern_done, ev_d2h_done;
    CUDA_CHECK(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK(cudaEventCreate(&ev_kern_done));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_done));

    int kthreads = 128, kblocks = (total_N + kthreads - 1) / kthreads;
    CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
    CUDA_MEM_CHECK("before lookup kernel");
    lookup_kernel_raw<<<kblocks, kthreads, 0, stream_data>>>(
        (const char*)d_chars.data(),
        d_off_buf.data(),
        total_N,
        d_states.data(), d_trans.data(), d_lemmas_dev.data(),
        d_out.data());
    CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

    // ── D2H ───────────────────────────────────────────────────────────────────
    ResultPair* h_out = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_out,
        (size_t)total_N * sizeof(ResultPair), cudaHostAllocDefault));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
        (size_t)total_N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));
    CUDA_CHECK(cudaStreamSynchronize(stream_data));

    // ── Timings ───────────────────────────────────────────────────────────────
    float trie_ms = 0, kern_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start,  ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start,  ev_kern_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,   ev_d2h_done));

    double tp = (kern_ms > 0) ? total_N / (kern_ms / 1000.0) : 0;

    std::cerr
        << "Words:            " << total_N << "\n"
        << "  Trie H2D (async):   " << trie_ms    << " ms\n"
        << "  Scan pass:          " << scan_ms     << " ms\n"
        << "  Compact pass:       " << compact_ms  << " ms\n"
        << "  Kernel:             " << kern_ms     << " ms"
        << "  (" << (long long)tp << " words/sec)\n"
        << "  D2H:                " << d2h_ms      << " ms\n"
        << "  Wall time:          "
        << std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t_wall).count()
        << " ms\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFreeHost(h_out));
    // CUDA_CHECK(cudaHostUnregister(mapped));
    munmap(mapped, file_sz);
    for (auto e : {ev_trie_start, ev_trie_ready,
                   ev_kern_start, ev_kern_done, ev_d2h_done})
        cudaEventDestroy(e);
    cudaStreamDestroy(stream_trie);
    cudaStreamDestroy(stream_data);
    return 0;
}