#include <chrono>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cudf/io/csv.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <rmm/device_uvector.hpp>

#include "structs.h"
#include "trie.h"
#include "lemmatizer_kernel.cuh"

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

__global__ void noop_kernel() {}


using ResultPair = thrust::pair<const char*, int32_t>;

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

    cudaStream_t stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_data));

    cudaEvent_t ev_trie_start, ev_trie_ready;
    CUDA_CHECK(cudaEventCreate(&ev_trie_start));
    CUDA_CHECK(cudaEventCreate(&ev_trie_ready));

    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas_dev.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_data));

    std::cerr << "Trie load:        "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - t0).count()
              << " ms\n";

    // ── Load file via cuDF ────────────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();

    auto source  = cudf::io::source_info(argv[1]);
    auto options = cudf::io::csv_reader_options::builder(source)
                    .header(-1)
                    .quoting(cudf::io::quote_style::NONE)  // disable quote handling
                    .doublequote(false)  // disable double quote handling
                    .build();
    auto result  = cudf::io::read_csv(options);
    CUDA_CHECK(cudaDeviceSynchronize()); // let cuDF finish all internal async work
    auto& col    = result.tbl->get_column(0);
    const int N  = col.size();

    std::cerr << "cuDF read_csv:    "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - t0).count()
              << " ms  N=" << N << "\n";

    // ── Lookup kernel ─────────────────────────────────────────────────────────

    auto d_input_view = cudf::column_device_view::create(col.view());
    rmm::device_uvector<ResultPair> d_out(N, stream_data);
    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0,
        (size_t)N * sizeof(ResultPair), stream_data));

    cudaEvent_t ev_kern_start, ev_kern_done, ev_d2h_done;
    CUDA_CHECK(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK(cudaEventCreate(&ev_kern_done));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_done));

    int threads = 128, blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaDeviceSynchronize()); // let cuDF finish all internal async work

    // auto const& offsets_col = col.view().child(0);
    // auto const& chars_col   = col.view().child(1);

    // std::cerr << "offsets size: " << offsets_col.size() << "\n"; // expect N+1
    // std::cerr << "chars size:   " << chars_col.size()   << "\n"; // expect 2061743986

    cudf::strings_column_view scv(col.view());
    const char*    chars_ptr   = scv.chars_begin(stream_data);
    const uint32_t* offsets_ptr = scv.offsets().begin<uint32_t>();

    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));

    CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
    lookup_kernel_raw<<<blocks, threads, 0, stream_data>>>(
        chars_ptr, (const uint32_t*)offsets_ptr,  // int32 cast — safe since all positive
        N,
        d_states.data(), d_trans.data(), d_lemmas_dev.data(),
        d_out.data());

    // lookup_kernel<<<blocks, threads, 0, stream_data>>>(
    //     *d_input_view, N,
    //     d_states.data(), d_trans.data(), d_lemmas_dev.data(),
    //     d_out.data());
    CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

    // ── D2H ───────────────────────────────────────────────────────────────────
    ResultPair* h_out = (ResultPair*)malloc((size_t)N * sizeof(ResultPair));
    // use cudaMemcpy instead of cudaMemcpyAsync since h_out is unpinned
    CUDA_CHECK(cudaMemcpy(h_out, d_out.data(),
        (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost));
    // ResultPair* h_out = nullptr;
    // CUDA_CHECK(cudaHostAlloc(&h_out,
    //     (size_t)N * sizeof(ResultPair), cudaHostAllocDefault));
    // CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
    //     (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));
    CUDA_CHECK(cudaStreamSynchronize(stream_data));

    // ── Timings ───────────────────────────────────────────────────────────────
    float trie_ms = 0, kern_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start,  ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start,  ev_kern_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,   ev_d2h_done));

    double tp = (kern_ms > 0) ? N / (kern_ms / 1000.0) : 0;

    std::cerr
        << "Words:            " << N << "\n"
        << "  Trie H2D:           " << trie_ms << " ms\n"
        << "  Kernel:             " << kern_ms  << " ms"
        << "  (" << (long long)tp << " words/sec)\n"
        << "  D2H:                " << d2h_ms   << " ms\n"
        << "  Wall time:          "
        << std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t_wall).count()
        << " ms\n";

    // copy results to host
    std::vector<ResultPair> h_results(N);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_out.data(),
        (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost));

    // print first/last 20
    int to_print = 20;
    fprintf(stderr, "First %d words:\n", to_print);
    for (int i = 0; i < to_print; ++i) {
        auto [ptr, len] = h_results[i];
        // ptr points to device memory — need to copy chars too
        std::string word(len, '\0');
        CUDA_CHECK(cudaMemcpy(word.data(), ptr, len, cudaMemcpyDeviceToHost));
        fprintf(stderr, "  [%s]\n", word.c_str());
    }
    if ((size_t)to_print < d_out.size()) {
        fprintf(stderr, "Last %d words:\n", to_print);
        for (int i = (int)d_out.size() - to_print; i < (int)d_out.size(); ++i) {
            auto [ptr, len] = h_results[i];
            // ptr points to device memory — need to copy chars too
            std::string word(len, '\0');
            CUDA_CHECK(cudaMemcpy(word.data(), ptr, len, cudaMemcpyDeviceToHost));
            fprintf(stderr, "  [%s]\n", word.c_str());
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    // CUDA_CHECK(cudaFreeHost(h_out));
    free(h_out);

    for (auto e : {ev_trie_start, ev_trie_ready,
                   ev_kern_start, ev_kern_done, ev_d2h_done})
        cudaEventDestroy(e);
    cudaStreamDestroy(stream_data);
    return 0;
}