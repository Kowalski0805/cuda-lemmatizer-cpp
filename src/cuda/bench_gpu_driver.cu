// bench_gpu_driver.cu — unified GPU benchmark driver
//
// Replaces: bench_gpu.cu, bench_gpu_raw.cu, bench_gpu_unified.cu,
//           bench_gpu_stride.cu, bench_gpu_stride_loop.cu, bench_gpu_loop.cu,
//           bench_gpu_unified_loop.cu, bench_gpu_opt.cu, bench_gpu_opt1.cu,
//           bench_gpu_col.cu, bench_gpu_bsearch.cu
//
// Usage: bench_gpu_driver <input_file>
//          [--kernel  packed|stride|col|bsearch]   default: packed
//          [--memory  device|unified|pinned]        default: device
//          [--input   multiline|raw]                default: multiline
//          [--loop]   [--duration N]                loop kernel N seconds (default 10)
//          [--output  <file>]                       write lemmatized output
//          [--warm]                                 pre-warm CUDA context
//          [--verbose]                              extra phase timings

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>

#include "icu_lowercase.h"
#include "lemmatizer_kernel.cuh"
#include "structs.h"
#include "trie.h"

// Kernels not declared in lemmatizer_kernel.cuh
__global__ void lookup_kernel_stride(
    const char* d_input, int num_words,
    const GpuState* states, const GpuTransition* transitions,
    const char* lemmas, char* d_output);

__global__ void lookup_kernel_bsearch(
    const char* d_input, int num_words,
    const char* d_keys, const char* d_vals, int num_entries,
    char* d_output);

__global__ void noop_kernel() {}

// Col kernel: returns one int32 lemma offset per word (or -1).
// Originally defined inline in bench_gpu_col.cu.
__global__ void lookup_kernel_index(
    const char* d_chars, const int32_t* d_offsets, int num_words,
    const GpuState* states, const GpuTransition* transitions,
    int32_t* d_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    int start = d_offsets[idx];
    int end   = d_offsets[idx + 1];
    int state = 0;

    for (int i = start; i < end; ++i) {
        char ch = d_chars[i];
        const GpuState& s = states[state];
        bool found = false;
        for (int j = 0; j < static_cast<int>(s.num_transitions); ++j) {
            const GpuTransition& t = transitions[s.transition_start_idx + j];
            if (t.c == ch) {
                state = t.next_state;
                found = true;
                break;
            }
        }
        if (!found) {
            d_out[idx] = -1;
            return;
        }
    }
    d_out[idx] = states[state].lemma_offset;
}

// ============================================================
// Utilities
// ============================================================

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

using ResultPair = thrust::pair<const char*, cudf::size_type>;
using Clock      = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ============================================================
// Config
// ============================================================

enum class Kernel { packed, stride, col, bsearch };
enum class Memory { device_, unified_, pinned_ };
enum class Input  { multiline, raw, multiline_mmap };

struct Config {
    std::string input_path, output_path;
    Kernel  kernel   = Kernel::packed;
    Memory  memory   = Memory::device_;
    Input   input    = Input::multiline;
    bool    loop     = false;
    double  duration = 10.0;
    bool    warm     = false;
    bool    verbose  = false;
    int     print    = 0;
    int     max_words = 0;
    bool    noop     = false;
};

static Config parse_args(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <input_file>\n"
            "  [--kernel  packed|stride|col|bsearch]    (default: packed)\n"
            "  [--memory  device|unified|pinned]        (default: device)\n"
            "  [--input   multiline|raw|multiline_mmap] (default: multiline)\n"
            "  [--loop]   [--duration N]                (default duration: 10s)\n"
            "  [--output  <file>]\n"
            "  [--warm]   [--verbose]\n", argv[0]);
        exit(1);
    }
    Config cfg;
    cfg.input_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires argument\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "--kernel")   { auto v = next();
            if      (v == "packed")  cfg.kernel = Kernel::packed;
            else if (v == "stride")  cfg.kernel = Kernel::stride;
            else if (v == "col")     cfg.kernel = Kernel::col;
            else if (v == "bsearch") cfg.kernel = Kernel::bsearch;
            else { fprintf(stderr, "Unknown kernel: %s\n", v.c_str()); exit(1); } }
        else if (a == "--memory")   { auto v = next();
            if      (v == "device")  cfg.memory = Memory::device_;
            else if (v == "unified") cfg.memory = Memory::unified_;
            else if (v == "pinned")  cfg.memory = Memory::pinned_;
            else { fprintf(stderr, "Unknown memory: %s\n", v.c_str()); exit(1); } }
        else if (a == "--input")    { auto v = next();
            if      (v == "multiline")      cfg.input = Input::multiline;
            else if (v == "raw")            cfg.input = Input::raw;
            else if (v == "multiline_mmap") cfg.input = Input::multiline_mmap;
            else { fprintf(stderr, "Unknown input: %s\n", v.c_str()); exit(1); } }
        else if (a == "--output")   { cfg.output_path = next(); }
        else if (a == "--duration") { cfg.duration = atof(next().c_str()); }
        else if (a == "--loop")     { cfg.loop = true; }
        else if (a == "--warm")     { cfg.warm = true; }
        else if (a == "--verbose")  { cfg.verbose = true; }
        else if (a == "--noop")     { cfg.noop = true; }
        else if (a == "--print")    { cfg.print = (int)strtol(next().c_str(), nullptr, 10); }
        else if (a == "--words")    { cfg.max_words = (int)strtol(next().c_str(), nullptr, 10); }
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); exit(1); }
    }
    if (cfg.memory == Memory::pinned_) cfg.input = Input::raw;  // mmap path requires one-word-per-line
    return cfg;
}

// ============================================================
// Shared helpers
// ============================================================

static void load_trie(
    std::vector<GpuState>&      h_states,
    std::vector<GpuTransition>& h_transitions,
    std::vector<char>&          h_lemmas)
{
    load_bin_vector("gpu_states.bin",      h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin",      h_lemmas);
    if (h_states.empty()) {
        fprintf(stderr, "Failed to load trie data. Run from cmake-build-debug/.\n"); exit(1);
    }
}

static void lowercase_ukr_fast(char* p, size_t len) {
    for (size_t i = 0; i + 1 < len; ) {
        uint8_t b0 = (uint8_t)p[i];
        uint8_t b1 = (uint8_t)p[i + 1];

        if (b0 == 0xD0) {
            if (b1 >= 0x90 && b1 <= 0x9F) {
                // А-П → а-п: second byte += 0x20, first byte stays 0xD0
                p[i + 1] = b1 + 0x20;
                i += 2; continue;
            }
            if (b1 >= 0xA0 && b1 <= 0xAF) {
                // Р-Я → р-я: first byte 0xD0→0xD1, second byte -= 0x20
                p[i]     = 0xD1;
                p[i + 1] = b1 - 0x20;
                i += 2; continue;
            }
            if (b1 == 0x84) { p[i] = 0xD1; p[i+1] = 0x94; i += 2; continue; } // Є→є
            if (b1 == 0x86) { p[i] = 0xD1; p[i+1] = 0x96; i += 2; continue; } // І→і
            if (b1 == 0x87) { p[i] = 0xD1; p[i+1] = 0x97; i += 2; continue; } // Ї→ї
        } else if (b0 == 0xD2) {
            if (b1 == 0x90) { p[i+1] = 0x91; i += 2; continue; } // Ґ→ґ
        }

        // skip by UTF-8 sequence length
        if      (b0 < 0x80) i += 1;
        else if (b0 < 0xE0) i += 2;
        else if (b0 < 0xF0) i += 3;
        else                 i += 4;
    }
}

static size_t load_multiline(
    const std::string &path,
    char *&h_chars,
    uint32_t *&h_offsets,
    Memory mem,
    int max_words = 0
) {
    std::vector<char> v_chars;
    std::vector<uint32_t> v_offsets;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    v_chars.reserve(file_sz);
    v_offsets.reserve(file_sz / 8);  // rough estimate, ~8 bytes per word average
    v_offsets.push_back(0);
    std::ifstream fin(path);

    if (!fin) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    std::string ln;
    while (std::getline(fin, ln)) {
        std::istringstream ss(ln);
        std::string tok;
        while (ss >> tok) {
            v_chars.insert(v_chars.end(), tok.begin(), tok.end());
            lowercase_ukr_fast(v_chars.data() + v_chars.size() - tok.size(), tok.size());
            v_offsets.push_back((uint32_t)v_chars.size());
            if (max_words > 0 && v_offsets.size() - 1 >= (size_t)max_words) goto done;
        }
    }

    done:
    if (mem == Memory::unified_) {
        cudaMallocManaged(&h_chars, v_chars.size() * sizeof(char));
        cudaMallocManaged(&h_offsets, v_offsets.size() * sizeof(uint32_t));
    } else {
        cudaHostAlloc(&h_chars, v_chars.size(), cudaHostAllocDefault);
        cudaHostAlloc(&h_offsets, v_offsets.size() * sizeof(uint32_t), cudaHostAllocDefault);
    }
    std::memcpy(h_chars, v_chars.data(), v_chars.size());
    std::memcpy(h_offsets, v_offsets.data(), v_offsets.size() * sizeof(uint32_t));
    return v_offsets.size();
}

static std::size_t load_multiline_mmap(
    const std::string &path,
    char *&h_chars,
    uint32_t *&h_offsets,
    Memory mem,
    int max_words = 0)
{
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const auto file_sz = (size_t)st.st_size;
    const char* mapped = static_cast<const char *>(mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0));
    close(fd);
    if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);

    std::vector<char>     v_chars;
    std::vector<uint32_t> v_offsets;
    v_chars.reserve(file_sz);
    v_offsets.reserve(file_sz / 8);  // rough estimate, ~8 bytes per word average
    v_offsets.push_back(0);

    const char* p = mapped;
    const char* end = mapped + file_sz;

    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;  // skip spaces, not newlines
        if (p >= end || *p == '\n' || *p == '\r') {        // end of line
            while (p < end && (*p == '\n' || *p == '\r')) ++p;
            continue;
        }
        const char* tok = p;
        while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
        size_t len = p - tok;
        if (len > 0) {
            v_chars.insert(v_chars.end(), tok, tok + len);
            lowercase_ukr_fast(v_chars.data() + v_chars.size() - len, len);
            v_offsets.push_back((uint32_t)v_chars.size());
            if (max_words > 0 && v_offsets.size() - 1 >= (size_t)max_words) goto done;
        }
    }
    done:

    munmap((void*)mapped, file_sz);

    if (mem == Memory::unified_) {
        cudaMallocManaged(&h_chars,   v_chars.size());
        cudaMallocManaged(&h_offsets, v_offsets.size() * sizeof(uint32_t));
    } else {
        cudaHostAlloc(&h_chars,   v_chars.size(),                        cudaHostAllocDefault);
        cudaHostAlloc(&h_offsets, v_offsets.size() * sizeof(uint32_t),   cudaHostAllocDefault);
    }
    std::memcpy(h_chars,   v_chars.data(),   v_chars.size());
    std::memcpy(h_offsets, v_offsets.data(), v_offsets.size() * sizeof(uint32_t));
    return v_offsets.size();
}

static void load_single(
    const std::string& path,
    char*& h_chars,
    uint32_t*& h_offsets,
    Memory mem,
    int max_words = 0)
{
    // mmap file
    // cudaHostAlloc h_chars with file_sz (tight upper bound)
    // vector<uint32_t> v_offsets grows dynamically
    // single pass: tokenize + lowercase + pack into h_chars
    // then cudaHostAlloc h_offsets exactly from v_offsets.size()
    // memcpy v_offsets → h_offsets
}

static void load_two_pass(const std::string& path,
    char*& h_chars,
    uint32_t*& h_offsets, int max_words = 0) {



}

static std::size_t load_raw(
    const std::string &path,
    char *&h_chars,
    uint32_t *&h_offsets,
    Memory mem,
    int max_words = 0
) {
    auto t_load = Clock::now();

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    const char* mapped = (const char*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);

    // alloc pinned at file_sz upfront — tight upper bound for raw (one word per line)
    if (mem == Memory::unified_)
        cudaMallocManaged(&h_chars, file_sz);
    else
        cudaHostAlloc(&h_chars, file_sz, cudaHostAllocDefault);

    std::vector<uint32_t> v_offsets;
    v_offsets.reserve(file_sz / 10);
    v_offsets.push_back(0);

    const char* p = mapped;
    const char* end = mapped + file_sz;
    size_t pos = 0;

    while (p < end) {
        const char* tok = p;
        while (p < end && *p != '\n' && *p != '\r') ++p;
        size_t len = p - tok;
        if (len > 0) {
            std::memcpy(h_chars + pos, tok, len);
            pos += len;
            v_offsets.push_back((uint32_t)pos);
        }
        while (p < end && (*p == '\n' || *p == '\r')) ++p;
        if (max_words > 0 && v_offsets.size() - 1 >= (size_t)max_words) break;
    }

    munmap((void*)mapped, file_sz);
    fprintf(stderr, "mmap loop: %.0f ms\n", ms_since(t_load));

    auto t_alloc = Clock::now();
    if (mem == Memory::unified_)
        cudaMallocManaged(&h_offsets, v_offsets.size() * sizeof(uint32_t));
    else
        cudaHostAlloc(&h_offsets, v_offsets.size() * sizeof(uint32_t), cudaHostAllocDefault);
    std::memcpy(h_offsets, v_offsets.data(), v_offsets.size() * sizeof(uint32_t));
    fprintf(stderr, "cudaHostAlloc+memcpy: %.0f ms\n", ms_since(t_alloc));

    return v_offsets.size();
}

static void pack_stride(const std::vector<std::string>& words, std::vector<char>& h_input) {
    const int N = (int)words.size();
    h_input.assign((size_t)N * MAX_WORD_LEN, 0);
    int truncated = 0;
    for (int i = 0; i < N; ++i) {
        const auto& w = words[i];
        if (w.size() >= (size_t)MAX_WORD_LEN) ++truncated;
        std::memcpy(h_input.data() + i * MAX_WORD_LEN, w.data(),
                    std::min(w.size(), (size_t)(MAX_WORD_LEN - 1)));
    }
    if (truncated)
        fprintf(stderr, "[warn] %d word(s) truncated to %d bytes\n", truncated, MAX_WORD_LEN - 1);
}

// Duration-based kernel loop. kernel_fn() records its own events and returns elapsed ms.
static void run_loop(const std::function<float()> &kernel_fn, size_t num_words, double duration_s,
                     int threads, int blocks) {
    int    num_iters = 0;
    double total_ms = 0.0, peak_ms = 0.0;
    fprintf(stderr, "Running for %.1fs  words=%lu  blocks=%d  threads=%d\n",
            duration_s, num_words, blocks, threads);
    auto wall = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - wall).count() < duration_s) {
        float ms = kernel_fn();
        total_ms += ms;
        if (ms > peak_ms) peak_ms = ms;
        ++num_iters;
        // if (++num_iters % 100 == 0) {
        //     double tp = (double)num_words * num_iters / (total_ms / 1000.0);
        //     fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
        //             num_iters, total_ms / num_iters, tp / 1e6);
        // }
    }
    double tp = (double)num_words * num_iters / (total_ms / 1000.0);
    fprintf(stderr, "\n=== Final ===\n"
            "  Iters:       %d\n  Words/iter:  %lu\n"
            "  Avg kernel:  %.3f ms\n  Peak kernel: %.3f ms\n"
            "  Throughput:  %.2fM words/sec\n",
            num_iters, num_words, total_ms / num_iters, peak_ms, tp / 1e6);
}

// One-shot kernel timer using CUDA events.
static float time_kernel_once(const std::function<void()>& fn) {
    noop_kernel<<<1, 1>>>();

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    fn();
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return ms;
}

// ============================================================
// run_packed — cuDF column kernel, device or unified memory
// ============================================================

static int run_packed(
    const Config&                cfg,
    const std::vector<GpuState>&       h_states,
    const std::vector<GpuTransition>&  h_transitions,
    const std::vector<char>&           h_lemmas,
    char*&                       h_chars,
    uint32_t*&                   h_offsets,
    const size_t                 N,
    double preprocess_ms)
{
    int threads = 128, blocks = (N + threads - 1) / threads;

    // ---- H2D ----
    std::string h2d_label = (cfg.memory == Memory::unified_) ? "Prefetch to GPU" : "H2D";
    auto t0 = Clock::now();

    GpuState*      d_states;
    GpuTransition* d_trans;
    char*          d_lemmas;
    ResultPair*    d_out;
    char*          d_chars;
    uint32_t*      d_offsets;

    cudaMalloc(&d_states, h_states.size()      * sizeof(GpuState));
    cudaMalloc(&d_trans,  h_transitions.size() * sizeof(GpuTransition));
    cudaMalloc(&d_lemmas, h_lemmas.size());

    cudaMemcpy(d_states, h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans,  h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas, h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    if (cfg.memory == Memory::unified_) {
        int dev; cudaGetDevice(&dev);
        cudaMemPrefetchAsync(h_chars, N, dev);
        cudaMemPrefetchAsync(h_offsets,(N + 1) * sizeof(int32_t), dev);
        d_chars   = h_chars;    // same pointer — unified
        d_offsets = h_offsets;
        cudaMallocManaged(&d_out, N * sizeof(ResultPair));
    } else {
        cudaMalloc(&d_chars,   h_offsets[N]);
        cudaMalloc(&d_offsets, (N + 1) * sizeof(uint32_t));
        cudaMalloc(&d_out,    N * sizeof(ResultPair));
        cudaMemcpy(d_chars,   h_chars,   h_offsets[N],                  cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, h_offsets, (N + 1) * sizeof(uint32_t),    cudaMemcpyHostToDevice);
    }

    cudaMemset(d_out, 0, N * sizeof(ResultPair));

    double h2d_ms = ms_since(t0);

    noop_kernel<<<1,1>>>();


    // ---- KERNEL DISPATCH ----
    auto launch = [&]() {
        cfg.noop ? noop_kernel<<<blocks, threads>>>() :
        lookup_kernel_raw<<<blocks, threads>>>(
            d_chars, d_offsets, N, d_states, d_trans, d_lemmas, d_out);
    };

    auto timed_launch = [&]() -> float {
        return time_kernel_once(launch);
    };

    if (cfg.loop) {
        // For unified loop: pages are already resident after the one-time prefetch above.
        fprintf(stderr, "Preprocess: %.3f ms %s: %.3f ms\n",
                preprocess_ms, h2d_label.c_str(), h2d_ms);
        run_loop(timed_launch, N, cfg.duration, threads, blocks);
        return 0;
    }

    float kernel_ms = timed_launch();

    // ---- D2H ----
    ResultPair* h_results;
    t0 = Clock::now();

    if (cfg.memory == Memory::unified_) {
        cudaMemPrefetchAsync(d_out, (size_t)N * sizeof(ResultPair), cudaCpuDeviceId);
        cudaDeviceSynchronize();
        h_results = d_out;  // same pointer, no copy needed
    } else {
        cudaHostAlloc(&h_results, (size_t)N * sizeof(ResultPair), cudaHostAllocDefault);
        cudaMemcpy(h_results, d_out, (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost);
    }

    double d2h_ms = ms_since(t0);

    // ---- DECODE ----
    t0 = Clock::now();
    std::vector<char> out_chars;
    std::vector<int32_t> out_offsets(N + 1);
    out_chars.reserve(h_offsets[N]);
    out_offsets[0] = 0;

    for (int i = 0; i < N; ++i) {
        ResultPair rp = h_results[i];
        const char* gpu_ptr = rp.first;
        int len = rp.second;

        const char* src;
        size_t src_len;
        if (gpu_ptr && len > 0) {
            ptrdiff_t off = gpu_ptr - d_lemmas;
            if (off >= 0 && static_cast<size_t>(off) < h_lemmas.size()) {
                src = h_lemmas.data() + off;
                src_len = len;
            } else {
                src = h_chars + h_offsets[i];
                src_len = h_offsets[i + 1] - h_offsets[i];
            }
        } else {
            src = h_chars + h_offsets[i];
            src_len = h_offsets[i + 1] - h_offsets[i];
        }
        out_chars.insert(out_chars.end(), src, src + src_len);
        out_offsets[i + 1] = out_chars.size();
    }
    double decode_ms = ms_since(t0);

    // ---- REPORT ----
    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kernel_ms + d2h_ms;
    fprintf(stderr, "Words: %lu\n"
            "  Preprocess:  %.3f ms\n"
            "  %-13s%.3f ms\n"
            "  Kernel:      %.3f ms  (%lld words/sec)\n"
            "  D2H:         %.3f ms\n"
            "  Decode:      %.3f ms\n"
            "  GPU total:   %.3f ms\n"
            "  End-to-end:  %.3f ms\n",
            N, preprocess_ms,
            (h2d_label + ":").c_str(), h2d_ms,
            kernel_ms, (long long)tp,
            d2h_ms, decode_ms, gpu_total, preprocess_ms + gpu_total);

    // if --print, print X words at start and end
    if (cfg.print > 0) {
        int to_print = std::min(cfg.print, (int)N);
        fprintf(stderr, "First %d words:\n", to_print);
        for (int i = 0; i < to_print; ++i) {
            uint32_t start = out_offsets[i];
            uint32_t len   = out_offsets[i + 1] - out_offsets[i];
            fprintf(stderr, "  %.*s\n", (int)len, out_chars.data() + start);
        }
        if (N > to_print) {
            fprintf(stderr, "Last %d words:\n", to_print);
            for (int i = N - to_print; i < N; ++i) {
                uint32_t start = out_offsets[i];
                uint32_t len   = out_offsets[i + 1] - out_offsets[i];
                fprintf(stderr, "  %.*s\n", (int)len, out_chars.data() + start);
            }
        }
    }

    return 0;
}

// ============================================================
// run_stride — fixed-stride kernel, cudaMalloc, device only
// ============================================================

static int run_stride(
    const Config&               cfg,
    std::vector<GpuState>&      h_states,
    std::vector<GpuTransition>& h_transitions,
    std::vector<char>&          h_lemmas,
    std::vector<std::string>&   words,
    std::vector<int>&           line_counts,
    double preprocess_ms)
{
    const int N = (int)words.size();
    int threads = 128, blocks = (N + threads - 1) / threads;
    const size_t stride_bytes = (size_t)N * MAX_WORD_LEN;

    // ---- PACK ----
    auto t0 = Clock::now();
    std::vector<char> h_input;
    pack_stride(words, h_input);
    double pack_ms = ms_since(t0);

    // ---- ALLOC + H2D ----
    t0 = Clock::now();
    char *d_in = nullptr, *d_out = nullptr;
    GpuState*      d_states = nullptr;
    GpuTransition* d_trans  = nullptr;
    char*          d_lemmas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,     stride_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,    stride_bytes));
    CUDA_CHECK(cudaMalloc(&d_states, h_states.size()      * sizeof(GpuState)));
    CUDA_CHECK(cudaMalloc(&d_trans,  h_transitions.size() * sizeof(GpuTransition)));
    CUDA_CHECK(cudaMalloc(&d_lemmas, h_lemmas.size()));

    cudaMemcpy(d_in,     h_input.data(),       stride_bytes,                                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states.data(),       h_states.size() * sizeof(GpuState),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans,  h_transitions.data(),  h_transitions.size() * sizeof(GpuTransition),cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas, h_lemmas.data(),        h_lemmas.size(),                             cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, stride_bytes);
    cudaDeviceSynchronize();
    double h2d_ms = ms_since(t0);

    // ---- KERNEL DISPATCH ----
    auto launch = [&]() {
        cfg.noop ? noop_kernel<<<blocks, threads>>>() :
        lookup_kernel_stride<<<blocks, threads>>>(d_in, N, d_states, d_trans, d_lemmas, d_out);
    };
    auto timed_launch = [&]() -> float { return time_kernel_once(launch); };

    if (cfg.loop) {
        fprintf(stderr, "Preprocess: %.3f ms  Pack: %.3f ms  H2D: %.3f ms\n",
                preprocess_ms, pack_ms, h2d_ms);
        run_loop(timed_launch, N, cfg.duration, threads, blocks);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_states); cudaFree(d_trans); cudaFree(d_lemmas);
        return 0;
    }

    float kernel_ms = timed_launch();

    // ---- D2H ----
    t0 = Clock::now();
    std::vector<char> h_output(stride_bytes);
    cudaMemcpy(h_output.data(), d_out, stride_bytes, cudaMemcpyDeviceToHost);
    double d2h_ms = ms_since(t0);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_states); cudaFree(d_trans); cudaFree(d_lemmas);

    // ---- DECODE ----
    std::vector<std::string> result_words(N);
    for (int i = 0; i < N; ++i) {
        const char* p = h_output.data() + i * MAX_WORD_LEN;
        result_words[i] = std::string(p, ::strnlen(p, MAX_WORD_LEN));
        if (result_words[i].empty()) result_words[i] = words[i];
    }

    // ---- REPORT ----
    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kernel_ms + d2h_ms;
    fprintf(stderr, "Words: %d\n"
            "  Preprocess:  %.3f ms\n  Pack:        %.3f ms\n  H2D:         %.3f ms\n"
            "  Kernel:      %.3f ms  (%lld words/sec)\n"
            "  D2H:         %.3f ms\n"
            "  GPU total:   %.3f ms\n"
            "  End-to-end:  %.3f ms\n",
            N, preprocess_ms, pack_ms, h2d_ms,
            kernel_ms, (long long)tp, d2h_ms, gpu_total, preprocess_ms + pack_ms + gpu_total);

    return 0;
}

// ============================================================
// run_col — packed-column kernel, int32 index output (4B/word)
// ============================================================

static int run_col(
    const Config&               cfg,
    std::vector<GpuState>&      h_states,
    std::vector<GpuTransition>& h_transitions,
    std::vector<char>&          h_lemmas,
    std::vector<std::string>&   words,
    std::vector<int>&           line_counts,
    double preprocess_ms)
{
    const int N = (int)words.size();
    int threads = 128, blocks = (N + threads - 1) / threads;

    // ---- PACK ----
    auto t0 = Clock::now();
    std::vector<char>    h_chars;
    std::vector<int32_t> h_offsets;
    double pack_ms = ms_since(t0);

    // ---- H2D ----
    t0 = Clock::now();
    rmm::device_uvector<char>          d_chars   (h_chars.size(),       rmm::cuda_stream_default);
    rmm::device_uvector<int32_t>       d_offsets (h_offsets.size(),     rmm::cuda_stream_default);
    rmm::device_uvector<GpuState>      d_states  (h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans   (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas  (h_lemmas.size(),      rmm::cuda_stream_default);
    // rmm::device_uvector<int32_t>       d_indices (N,                    rmm::cuda_stream_default);
    int32_t* d_indices_raw;
    CUDA_CHECK(cudaMalloc(&d_indices_raw, N * sizeof(int32_t)));

    cudaMemcpy(d_chars.data(),   h_chars.data(),        h_chars.size(),                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets.data(), h_offsets.data(),      h_offsets.size() * sizeof(int32_t),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_states.data(),  h_states.data(),       h_states.size()      * sizeof(GpuState),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans.data(),   h_transitions.data(),  h_transitions.size() * sizeof(GpuTransition),cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas.data(),  h_lemmas.data(),        h_lemmas.size(),                             cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    double h2d_ms = ms_since(t0);

    fprintf(stderr, "First 5 offsets: ");
    for (int i = 0; i < 5; ++i) fprintf(stderr, "%d ", h_offsets[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "First 20 chars: ");
    for (int i = 0; i < 20; ++i) fprintf(stderr, "%c", h_chars[i] ? h_chars[i] : '?');
    fprintf(stderr, "\n");

    if (cfg.loop) {
        fprintf(stderr, "Preprocess: %.3f ms  Pack: %.3f ms  H2D: %.3f ms\n",
                preprocess_ms, pack_ms, h2d_ms);
        run_loop([&]() -> float {
            return time_kernel_once([&]() {
                cfg.noop ? noop_kernel<<<blocks, threads>>>() :
                lookup_kernel_index<<<blocks, threads>>>(
                    d_chars.data(), d_offsets.data(), N,
                    d_states.data(), d_trans.data(), d_indices_raw);
            });
        }, N, cfg.duration, threads, blocks);
        cudaFree(d_indices_raw);
        return 0;
    }

    // ---- KERNEL ----
    float kernel_ms = time_kernel_once([&]() {
        cfg.noop ? noop_kernel<<<blocks, threads>>>() :
        lookup_kernel_index<<<blocks, threads>>>(
            d_chars.data(), d_offsets.data(), N,
            d_states.data(), d_trans.data(), d_indices_raw);
    });

    t0 = Clock::now();
    int32_t* h_indices;
    CUDA_CHECK(cudaHostAlloc(&h_indices, N * sizeof(int32_t), cudaHostAllocDefault));
    double h_alloc_ms = ms_since(t0);
    // ---- D2H (4 bytes/word) ----
    t0 = Clock::now();
    cudaMemcpy(h_indices, d_indices_raw, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    auto d2h_ms = ms_since(t0);

    t0 = Clock::now();
    // ---- DECODE ----
    std::vector<std::string> result_words(N);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < N; ++i) {
        int32_t idx = h_indices[i];
        result_words[i] = (idx >= 0 && idx < h_lemmas.size())
            ? std::string(h_lemmas.data() + idx)
            : words[i];
    }
    double decode_ms = ms_since(t0);

    // ---- REPORT ----
    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kernel_ms + d2h_ms;
    fprintf(stderr, "Words: %d\n"
            "  Preprocess:  %.3f ms\n  Pack:        %.3f ms\n  H2D:         %.3f ms\n"
            "  Kernel:      %.3f ms  (%lld words/sec)\n"
            "  Host alloc:     %.3f ms\n"
            "  D2H (4B/word): %.3f ms\n"
            "  Decode:       %.3f ms\n"
            "  GPU total:   %.3f ms\n"
            "  End-to-end:  %.3f ms\n",
            N, preprocess_ms, pack_ms, h2d_ms,
            kernel_ms, (long long)tp, h_alloc_ms, d2h_ms, decode_ms, gpu_total, preprocess_ms + pack_ms + gpu_total + decode_ms);

    cudaFreeHost(h_indices);
    cudaFree(d_indices_raw);

    return 0;
}

// ============================================================
// run_bsearch — binary-search dict kernel
// ============================================================

static int run_bsearch(
    const Config&               cfg,
    std::vector<std::string>&   words,
    std::vector<int>&           line_counts,
    double preprocess_ms)
{
    // ---- LOAD DICT ----
    std::vector<char> h_keys, h_vals;
    int num_entries = 0;
    load_bin_vector("bsearch_keys.bin", h_keys);
    load_bin_vector("bsearch_vals.bin", h_vals);
    if (h_keys.empty() || h_vals.empty()) {
        fprintf(stderr, "bsearch_keys.bin / bsearch_vals.bin not found — building from CSV...\n");
        build_flat_sorted_dict_from_csv("ukr_morph_dict.csv", h_keys, h_vals, num_entries);
        save_bin_vector("bsearch_keys.bin", h_keys);
        save_bin_vector("bsearch_vals.bin", h_vals);
    } else {
        num_entries = (int)(h_keys.size() / MAX_WORD_LEN);
        fprintf(stderr, "Loaded bsearch dict: %d entries\n", num_entries);
    }
    if (num_entries == 0) { fprintf(stderr, "Empty dictionary.\n"); return 1; }

    const int N = (int)words.size();
    int threads = 128, blocks = (N + threads - 1) / threads;
    const size_t stride_bytes = (size_t)N * MAX_WORD_LEN;
    const size_t dict_bytes   = (size_t)num_entries * MAX_WORD_LEN;

    // ---- PACK ----
    std::vector<char> h_input;
    pack_stride(words, h_input);

    // ---- ALLOC + H2D ----
    auto t0 = Clock::now();
    char *d_in = nullptr, *d_out = nullptr, *d_keys = nullptr, *d_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,   stride_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,  stride_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys, dict_bytes));
    CUDA_CHECK(cudaMalloc(&d_vals, dict_bytes));

    cudaMemcpy(d_in,   h_input.data(), stride_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, h_keys.data(),  dict_bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(),  dict_bytes,   cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, stride_bytes);

    // ---- KERNEL ----
    float kernel_ms = time_kernel_once([&]() {
        cfg.noop ? noop_kernel<<<blocks, threads>>>() :
        lookup_kernel_bsearch<<<blocks, threads>>>(d_in, N, d_keys, d_vals, num_entries, d_out);
    });

    // ---- D2H ----
    std::vector<char> h_output(stride_bytes);
    cudaMemcpy(h_output.data(), d_out, stride_bytes, cudaMemcpyDeviceToHost);
    double total_ms = ms_since(t0);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_keys); cudaFree(d_vals);

    // ---- DECODE ----
    std::vector<std::string> result_words(N);
    for (int i = 0; i < N; ++i) {
        const char* p = h_output.data() + i * MAX_WORD_LEN;
        result_words[i] = std::string(p, ::strnlen(p, MAX_WORD_LEN));
        if (result_words[i].empty()) result_words[i] = words[i];
    }

    double tp = (kernel_ms > 0.f) ? (N / (kernel_ms / 1000.0)) : 0.0;
    fprintf(stderr, "Words: %d  Kernel: %.3f ms  Total (H2D+kernel+D2H): %.3f ms  "
            "Throughput: %lld words/sec\n", N, kernel_ms, total_ms, (long long)tp);

    return 0;
}

// ============================================================
// run_pinned — async streams, mmap, pinned memory (opt path)
// ============================================================

static int run_pinned(
    const Config&               cfg,
    std::vector<GpuState>&      h_states,
    std::vector<GpuTransition>& h_transitions,
    std::vector<char>&          h_lemmas)
{
    auto wall_start = Clock::now();

    // ---- PRE-SCAN: mmap, count words ----
    int fd = open(cfg.input_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", cfg.input_path.c_str()); return 1; }
    struct stat st; fstat(fd, &st);
    const size_t file_sz = (size_t)st.st_size;
    const char* mapped = (const char*)mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return 1; }
    madvise((void*)mapped, file_sz, MADV_SEQUENTIAL);

    int pre_N = 0; size_t pre_total_chars = 0;
    { const char* p = mapped, *end = mapped + file_sz;
      while (p < end) {
          const char* ws = p;
          while (p < end && *p != '\n' && *p != '\r') ++p;
          int len = (int)(p - ws);
          if (len > 0) {
              ++pre_N; pre_total_chars += len;
              if (cfg.max_words > 0 && pre_N >= cfg.max_words) break;
          }
          while (p < end && (*p == '\n' || *p == '\r')) ++p;
      } }
    if (pre_N == 0) { fprintf(stderr, "No words.\n"); munmap((void*)mapped, file_sz); return 1; }

    // ---- PRE-ALLOCATE (outside timing) ----
    auto t0 = Clock::now();
    char*       h_chars;
    int32_t*    h_offsets;
    ResultPair* h_out;
    CUDA_CHECK(cudaHostAlloc(&h_chars,   pre_total_chars,                       cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_offsets, (size_t)(pre_N+1) * sizeof(int32_t),   cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out,     (size_t)pre_N     * sizeof(ResultPair),cudaHostAllocDefault));

    rmm::device_uvector<GpuState>      d_states   (h_states.size(),      rmm::cuda_stream_default);
    rmm::device_uvector<GpuTransition> d_trans    (h_transitions.size(), rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_lemmas   (h_lemmas.size(),       rmm::cuda_stream_default);
    rmm::device_uvector<char>          d_chars_raw(pre_total_chars,       rmm::cuda_stream_default);
    // in PRE-ALLOCATE section, alongside other device allocations
    int32_t* d_indices_raw = nullptr;
    if (cfg.kernel == Kernel::col)
        CUDA_CHECK(cudaMalloc(&d_indices_raw, (size_t)pre_N * sizeof(int32_t)));
    int32_t* h_indices = nullptr;
    if (cfg.kernel == Kernel::col)
        CUDA_CHECK(cudaHostAlloc(&h_indices, (size_t)pre_N * sizeof(int32_t), cudaHostAllocDefault));
    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, pre_N + 1, cudf::mask_state::UNALLOCATED);
    if (cfg.verbose)
        fprintf(stderr, "Pre-alloc: %.0f ms\n", ms_since(t0));

    // ---- STREAMS + EVENTS ----
    cudaStream_t stream_trie, stream_data;
    CUDA_CHECK(cudaStreamCreate(&stream_trie));
    CUDA_CHECK(cudaStreamCreate(&stream_data));
    cudaEvent_t ev_trie_start, ev_trie_ready, ev_h2d_start, ev_h2d_done,
                ev_kern_start, ev_kern_done, ev_d2h_done;
    for (auto* e : {&ev_trie_start, &ev_trie_ready, &ev_h2d_start, &ev_h2d_done,
                    &ev_kern_start, &ev_kern_done, &ev_d2h_done})
        CUDA_CHECK(cudaEventCreate(e));

    // ---- ASYNC TRIE H2D (overlaps load+pack below) ----
    CUDA_CHECK(cudaEventRecord(ev_trie_start, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_states.data(), h_states.data(),
        h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_trans.data(), h_transitions.data(),
        h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaMemcpyAsync(d_lemmas.data(), h_lemmas.data(),
        h_lemmas.size(), cudaMemcpyHostToDevice, stream_trie));
    CUDA_CHECK(cudaEventRecord(ev_trie_ready, stream_trie));
    CUDA_CHECK(cudaStreamSynchronize(stream_trie));

    // ---- LOAD + PACK: mmap → pinned (one pass) ----
    t0 = Clock::now();
    h_offsets[0] = 0;
    size_t pos = 0; int N = 0;
    const char* p = mapped, *end = mapped + file_sz;
    while (p < end) {
        const char* ws = p;
        while (p < end && *p != '\n' && *p != '\r') ++p;
        int len = (int)(p - ws);
        if (len > 0) {
            std::memcpy(h_chars + pos, ws, len);
            h_offsets[N + 1] = (int32_t)(pos += len);
            ++N;
            if (cfg.max_words > 0 && N >= cfg.max_words) break;
        }
        while (p < end && (*p == '\n' || *p == '\r')) ++p;
    }
    munmap((void*)mapped, file_sz);
    double pack_ms = ms_since(t0);
    const size_t total_chars = (size_t)h_offsets[N];

    fprintf(stderr, "First 5 offsets: ");
    for (int i = 0; i < 5; ++i) fprintf(stderr, "%d ", h_offsets[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "First 20 chars: ");
    for (int i = 0; i < 20; ++i) fprintf(stderr, "%c", h_chars[i] ? h_chars[i] : '?');
    fprintf(stderr, "\n");

    // ---- ASYNC DATA H2D ----
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(d_chars_raw.data(), h_chars,
        total_chars, cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), h_offsets,
        (size_t)(N + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream_data));

    int threads = 128, blocks = (N + threads - 1) / threads;

    if (cfg.kernel == Kernel::col) {
        // rmm::device_uvector<int32_t> d_indices(N, rmm::cuda_stream_default);
        fprintf(stderr, "total_chars: %zu N: %d\n", total_chars, N);
        CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));
        CUDA_CHECK(cudaStreamSynchronize(stream_data));

        if (cfg.loop) {
            float trie_ms = 0, h2d_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
            CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
            fprintf(stderr, "Pack: %.3f ms  Trie H2D: %.3f ms  Data H2D: %.3f ms\n",
                    pack_ms, trie_ms, h2d_ms);
            run_loop([&]() -> float {
                return time_kernel_once([&]() {
                    cfg.noop ? noop_kernel<<<blocks, threads>>>() :
                    lookup_kernel_index<<<blocks, threads>>>(
                        d_chars_raw.data(),
                        offsets_col->mutable_view().data<int32_t>(),
                        N, d_states.data(), d_trans.data(), d_indices_raw);
                });
            }, N, cfg.duration, threads, blocks);
            // cleanup
            CUDA_CHECK(cudaFree(d_indices_raw));
            CUDA_CHECK(cudaFreeHost(h_indices));
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

        noop_kernel<<<1,1>>>();

        CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
        cfg.noop ? noop_kernel<<<blocks, threads>>>() :
        lookup_kernel_index<<<blocks, threads, 0, stream_data>>>(
            d_chars_raw.data(),
            offsets_col->mutable_view().data<int32_t>(),
            N, d_states.data(), d_trans.data(), d_indices_raw);
        CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

        CUDA_CHECK(cudaMemcpyAsync(h_indices, d_indices_raw,
            (size_t)N * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_data));
        CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));
        CUDA_CHECK(cudaStreamSynchronize(stream_data));

        float trie_ms = 0, h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
        CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_done));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,  ev_d2h_done));

        double tp = (kern_ms > 0.f) ? (N / (kern_ms / 1000.0)) : 0.0;
        double gpu_total = h2d_ms + kern_ms + d2h_ms;
        fprintf(stderr, "Words: %d\n"
                "  Pack (mmap→pinned):                    %.3f ms\n"
                "  Trie H2D (async, overlapped pack):     %.3f ms\n"
                "  Data H2D (pinned→GDDR6X, async):       %.3f ms\n"
                "  Kernel:                                %.3f ms  (%lld words/sec)\n"
                "  D2H (4B/word, GDDR6X→host):            %.3f ms\n"
                "  GPU total (H2D+kernel+D2H):            %.3f ms\n"
                "  End-to-end (pack+GPU):                 %.3f ms\n",
                N, pack_ms, trie_ms, h2d_ms, kern_ms, (long long)tp,
                d2h_ms, gpu_total, pack_ms + gpu_total);

        // if --print, print X words at start and end
        if (cfg.print > 0) {
            int to_print = std::min(cfg.print, N);
            fprintf(stderr, "First %d words:\n", to_print);
            for (int i = 0; i < to_print; ++i) {
                int32_t idx = h_indices[i];
                const char* lemma = (idx >= 0) ? (h_lemmas.data() + idx) : "(no match)";
                fprintf(stderr, "  %s\n", lemma);
            }
            if ((size_t)to_print < N) {
                fprintf(stderr, "Last %d words:\n", to_print);
                for (int i = N - to_print; i < N; ++i) {
                    int32_t idx = h_indices[i];
                    const char* lemma = (idx >= 0) ? (h_lemmas.data() + idx) : "(no match)";
                    fprintf(stderr, "  %s\n", lemma);
                }
            }
        }

        int oov_count = 0;
        long oov_bytes = 0, vocab_bytes = 0;

        for (int i = 0; i < N; i++) {
            int word_len = h_offsets[i+1] - h_offsets[i]; // byte length of word i
            if (h_indices[i] == -1) {
                oov_count++;
                oov_bytes += word_len;
            } else {
                vocab_bytes += word_len;
            }
        }

        int vocab_count = N - oov_count;
        printf("OOV:   %d / %d (%.1f%%), avg bytes: %.2f\n",
               oov_count, N, 100.0 * oov_count / N,
               oov_count > 0 ? (double)oov_bytes / oov_count : 0.0);
        printf("Vocab: %d / %d (%.1f%%), avg bytes: %.2f\n",
               vocab_count, N, 100.0 * vocab_count / N,
               vocab_count > 0 ? (double)vocab_bytes / vocab_count : 0.0);

        // cleanup and return before packed path runs
        CUDA_CHECK(cudaFree(d_indices_raw));
        CUDA_CHECK(cudaFreeHost(h_indices));
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
    rmm::device_uvector<ResultPair>    d_out      (pre_N,                 rmm::cuda_stream_default);

    CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0, (size_t)N * sizeof(ResultPair), stream_data));

    auto input_col = cudf::make_strings_column(
        N, std::move(offsets_col),
        rmm::device_buffer{d_chars_raw.data(), total_chars, rmm::cuda_stream_default},
        0, rmm::device_buffer{});
    auto d_input_view = cudf::column_device_view::create(input_col->view());

    if (cfg.loop) {
        float trie_ms = 0, h2d_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
        fprintf(stderr, "Pack: %.3f ms  Trie H2D: %.3f ms  Data H2D: %.3f ms\n",
                pack_ms, trie_ms, h2d_ms);
        run_loop([&]() -> float {
            return time_kernel_once([&]() {
                cfg.noop ? noop_kernel<<<blocks, threads>>>() :
                lookup_kernel<<<blocks, threads, 0, stream_data>>>(
        *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
            });
        }, N, cfg.duration, threads, blocks);
        // cleanup
        CUDA_CHECK(cudaFree(d_indices_raw));
        CUDA_CHECK(cudaFreeHost(h_indices));
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

    noop_kernel<<<1,1>>>();

    // ---- KERNEL ----
    CUDA_CHECK(cudaStreamWaitEvent(stream_data, ev_trie_ready));
    CUDA_CHECK(cudaEventRecord(ev_kern_start, stream_data));
    cfg.noop ? noop_kernel<<<blocks, threads>>>() :
    lookup_kernel<<<blocks, threads, 0, stream_data>>>(
        *d_input_view, N, d_states.data(), d_trans.data(), d_lemmas.data(), d_out.data());
    CUDA_CHECK(cudaEventRecord(ev_kern_done, stream_data));

    // ---- ASYNC D2H ----
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.data(),
        (size_t)N * sizeof(ResultPair), cudaMemcpyDeviceToHost, stream_data));
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream_data));
    CUDA_CHECK(cudaStreamSynchronize(stream_data));

    float trie_ms = 0, h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&trie_ms, ev_trie_start, ev_trie_ready));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  ev_h2d_start,  ev_h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_done));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ev_kern_done,  ev_d2h_done));

    double tp = (kern_ms > 0.f) ? (N / (kern_ms / 1000.0)) : 0.0;
    double gpu_total = h2d_ms + kern_ms + d2h_ms;
    fprintf(stderr, "Words: %d\n"
            "  Pack (mmap→pinned):                    %.3f ms\n"
            "  Trie H2D (async, overlapped pack):     %.3f ms\n"
            "  Data H2D (pinned→GDDR6X, async):       %.3f ms\n"
            "  Kernel:                                %.3f ms  (%lld words/sec)\n"
            "  D2H (GDDR6X→pinned, async):            %.3f ms\n"
            "  GPU total (H2D+kernel+D2H):            %.3f ms\n"
            "  End-to-end (pack+GPU):                 %.3f ms\n",
            N, pack_ms, trie_ms, h2d_ms, kern_ms, (long long)tp,
            d2h_ms, gpu_total, pack_ms + gpu_total);
    if (cfg.verbose)
        fprintf(stderr, "  Wall time since start:                 %.3f ms\n", ms_since(wall_start));

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

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.warm) {
        // void* tmp; CUDA_CHECK(cudaMalloc(&tmp, 1)); CUDA_CHECK(cudaFree(tmp));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // Load trie (all kernels except bsearch)
    std::vector<GpuState>      h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char>          h_lemmas;
    if (cfg.kernel != Kernel::bsearch)
        load_trie(h_states, h_transitions, h_lemmas);

    // Pinned path: no standard preprocess step (mmap handles everything)
    if (cfg.memory == Memory::pinned_)
        return run_pinned(cfg, h_states, h_transitions, h_lemmas);

    // Load + preprocess input
    auto t0 = Clock::now();
    char* h_chars;
    uint32_t* h_offsets;
    size_t N;
    if (cfg.input == Input::multiline)
        N = load_multiline(cfg.input_path, h_chars, h_offsets, cfg.memory, cfg.max_words);
    else if (cfg.input == Input::multiline_mmap)
        N = load_multiline_mmap(cfg.input_path, h_chars, h_offsets, cfg.memory, cfg.max_words);
    else
        N = load_raw(cfg.input_path, h_chars, h_offsets, cfg.memory, cfg.max_words);
    double preprocess_ms = ms_since(t0);
    N = N - 1;

    if (N <= 0) { fprintf(stderr, "No words.\n"); return 1; }

    switch (cfg.kernel) {
        case Kernel::packed:
            run_packed(cfg, h_states, h_transitions, h_lemmas,
                              h_chars, h_offsets, N, preprocess_ms);
            break;
        // case Kernel::stride:
        //     run_stride(cfg, h_states, h_transitions, h_lemmas,
        //                       h_chars, h_offsets, preprocess_ms);
        //     break;
        // case Kernel::col:
        //     run_col(cfg, h_states, h_transitions, h_lemmas,
        //                    h_chars, h_offsets, preprocess_ms);
        //     break;
        // case Kernel::bsearch:
        //     run_bsearch(cfg, h_chars, h_offsets, preprocess_ms);
        //     break;
    }

    // if --print, print X words at start and end
    // if (cfg.print > 0) {
    //     int to_print = std::min(cfg.print, (int)words.size());
    //     fprintf(stderr, "First %d words:\n", to_print);
    //     for (int i = 0; i < to_print; ++i)
    //         fprintf(stderr, "  %s\n", words[i].c_str());
    //     if ((size_t)to_print < words.size()) {
    //         fprintf(stderr, "Last %d words:\n", to_print);
    //         for (int i = (int)words.size() - to_print; i < (int)words.size(); ++i)
    //             fprintf(stderr, "  %s\n", words[i].c_str());
    //     }
    // }

    return 0;
}
