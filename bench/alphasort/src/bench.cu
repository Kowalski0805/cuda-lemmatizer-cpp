// bench.cu — input-ordering strategies for GPU trie lemmatization.
//
// Variants:
//   0  baseline        no reordering
//   1  cpu-sort        std::sort on host by 64-bit prefix key (+ full tie-break)
//   2  gpu-sort        CUB radix sort, full 64-bit key (first 8 bytes)
//   2b gpu-full        exact full-depth sort: initial radix pass + iterative
//                      segmented refinement of tie runs (8 bytes per pass)
//   3  gpu-prefix      CUB radix sort restricted to the top X bytes of the key
//   4  gpu-partition   single-pass bucket partition (histogram+scan+scatter),
//                      i.e. exactly one radix pass — O(n) approximate ordering
//   5  streaming       batched pipeline: copy/partition batch k+1 while worker
//                      streams run lookups for batch k; a fixed prefix range is
//                      pinned to each stream so the same stream keeps touching
//                      the same subtries batch after batch
//
// All variants preserve output order: results are scattered to out[originalIdx]
// through a permutation array, so no re-materialization of a sorted corpus is
// ever needed.

#include "../include/common.cuh"
#ifdef USE_REAL_TRIE
#include "../include/trie_real.cuh"  // production trie from gpu_*.bin
#else
#include "../include/trie.cuh"  // self-contained stand-in
#endif

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- data model
struct WordsHost {
  std::vector<uint8_t> bytes;   // concatenated UTF-8 words (no separators)
  std::vector<uint32_t> offs;   // n+1 offsets (CSR)
  uint32_t n() const { return (uint32_t)offs.size() - 1; }
};

struct WordsDev {
  uint8_t* bytes = nullptr;
  uint32_t* offs = nullptr;
  uint32_t n = 0;

  void upload(const WordsHost& h) {
    n = h.n();
    CUDA_CHECK(cudaMalloc(&bytes, h.bytes.size()));
    CUDA_CHECK(cudaMalloc(&offs, h.offs.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(bytes, h.bytes.data(), h.bytes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offs, h.offs.data(), h.offs.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
  }
  void free() {
    cudaFree(bytes);
    cudaFree(offs);
  }
};

// ------------------------------------------------------------------- kernels
__global__ void makeKeysKernel(const uint8_t* bytes, const uint32_t* offs,
                               uint32_t n, uint32_t base, int keyMode,
                               uint64_t* keys, uint32_t* idx) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t o = offs[i] - base;
  uint32_t len = offs[i + 1] - offs[i];
  keys[i] = makeSortKey(bytes + o, len, keyMode);
  if (idx) idx[i] = i;
}

// perm == nullptr -> natural order. Results always land at out[word's index],
// so downstream consumers never see the permutation.
__global__ void lookupKernel(TrieDev t, const uint8_t* bytes,
                             const uint32_t* offs, uint32_t n, uint32_t base,
                             const uint32_t* perm, int32_t* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t w = perm ? perm[i] : i;
  uint32_t o = offs[w] - base;
  uint32_t len = offs[w + 1] - offs[w];
  out[w] = trieLookup(t, bytes + o, len);
}

// ---- variant 6: sorted order with the word bytes compacted to match -------
// The profiler shows that sorting wins warp uniformity but loses input-stream
// coalescing: reading words through a permutation costs ~2 sectors per word
// instead of one shared line per warp. Materialising the words in sorted order
// once removes that loss, so the two effects can be had together. The compact
// copy is a transient device buffer — the corpus on disk and the output order
// are untouched, results still scatter through perm to out[originalIdx].
__global__ void gatherLenKernel(const uint32_t* offs, uint32_t n,
                                const uint32_t* perm, uint32_t* lens) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t w = perm[i];
  lens[i] = offs[w + 1] - offs[w];
}

__global__ void gatherBytesKernel(const uint8_t* bytes, const uint32_t* offs,
                                  uint32_t n, const uint32_t* perm,
                                  const uint32_t* cOffs, uint8_t* cBytes) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t w = perm[i];
  const uint32_t src = offs[w], len = offs[w + 1] - offs[w];
  uint32_t dst = cOffs[i];
  for (uint32_t k = 0; k < len; ++k) cBytes[dst + k] = bytes[src + k];
}

// Word i is contiguous at cBytes[cOffs[i]]; the result still lands at the
// token's original position, so downstream order is preserved exactly as in
// every other variant.
__global__ void lookupCompactKernel(TrieDev t, const uint8_t* cBytes,
                                    const uint32_t* cOffs, uint32_t n,
                                    const uint32_t* perm, int32_t* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t o = cOffs[i];
  const uint32_t len = cOffs[i + 1] - o;
  out[perm[i]] = trieLookup(t, cBytes + o, len);
}

// One radix pass by hand: histogram over the top `binBits` of the key.
__global__ void histKernel(const uint64_t* keys, uint32_t n, uint32_t* hist,
                           int binBits) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t b = (uint32_t)(keys[i] >> (64 - binBits));
  atomicAdd(&hist[b], 1u);
}

__global__ void scatterKernel(const uint64_t* keys, uint32_t n,
                              uint32_t* cursor, int binBits, uint32_t* perm) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t b = (uint32_t)(keys[i] >> (64 - binBits));
  uint32_t pos = atomicAdd(&cursor[b], 1u);
  perm[pos] = i;
}

// ---- kernels for variant 2b: full sort via segmented tie refinement --------

// Key covering bytes [byteOff, byteOff+8) of a word; 0 once the word is
// exhausted (NUL bytes cannot occur in newline-delimited UTF-8 words, so
// key == 0  <=>  exhausted).
__host__ __device__ inline uint64_t packKeyAt(const uint8_t* w, uint32_t len,
                                              uint32_t byteOff) {
  if (byteOff >= len) return 0;
  return packKey8(w + byteOff, len - byteOff);
}

// Depth-d keys, read through the current permutation.
__global__ void makeKeysAtKernel(const uint8_t* bytes, const uint32_t* offs,
                                 uint32_t n, uint32_t base,
                                 const uint32_t* perm, uint32_t byteOff,
                                 uint64_t* keys) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t w = perm[i];
  uint32_t o = offs[w] - base;
  uint32_t len = offs[w + 1] - offs[w];
  keys[i] = packKeyAt(bytes + o, len, byteOff);
}

// Segment boundaries accumulate: once set, a head flag stays set, and each
// depth adds boundaries where the just-sorted keys differ.
__global__ void headUpdateKernel(const uint64_t* keys, uint32_t n,
                                 uint32_t* head) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i == 0)
    head[0] = 1;
  else if (keys[i] != keys[i - 1])
    head[i] = 1;
}

// A tie run whose depth-d keys are all zero consists of identical words
// (all exhausted) — no further pass can refine it, so it must not keep the
// loop alive. Mark runs containing at least one non-zero key as active.
__global__ void markActiveKernel(const uint64_t* keys, const uint32_t* segId,
                                 uint32_t n, uint8_t* runActive) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (keys[i] != 0) runActive[segId[i] - 1] = 1;  // benign race, same value
}

// Compact runs of length > 1 that are still active into begin/end pairs for
// cub::DeviceSegmentedSort.
__global__ void buildTieSegsKernel(const uint32_t* starts, uint32_t numRuns,
                                   uint32_t n, const uint8_t* runActive,
                                   uint32_t* segBegin, uint32_t* segEnd,
                                   uint32_t* ctr) {
  uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= numRuns) return;
  uint32_t b = starts[r];
  uint32_t e = (r + 1 < numRuns) ? starts[r + 1] : n;
  if (e - b > 1 && runActive[r]) {
    uint32_t p = atomicAdd(ctr, 1u);
    segBegin[p] = b;
    segEnd[p] = e;
  }
}

// Extract the S+1 segment boundaries (in word positions) for the worker
// streams from the scanned histogram. segBounds[s] = binStart[s * binsPerSeg].
__global__ void segBoundsKernel(const uint32_t* binStart, uint32_t n,
                                int numBins, int numSegs, uint32_t* segBounds) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s > numSegs) return;
  if (s == numSegs)
    segBounds[s] = n;
  else
    segBounds[s] = binStart[(uint64_t)s * numBins / numSegs];
}

// Lookup restricted to one stream's segment of the partitioned permutation.
__global__ void lookupSegmentKernel(TrieDev t, const uint8_t* bytes,
                                    const uint32_t* offs, uint32_t base,
                                    const uint32_t* perm,
                                    const uint32_t* segBounds, int seg,
                                    int32_t* out) {
  uint32_t lo = segBounds[seg], hi = segBounds[seg + 1];
  uint32_t i = lo + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hi) return;
  uint32_t w = perm[i];
  uint32_t o = offs[w] - base;
  uint32_t len = offs[w + 1] - offs[w];
  out[w] = trieLookup(t, bytes + o, len);
}

// -------------------------------------------------------------- data loading
static void appendWord(WordsHost& W, const uint8_t* p, size_t len) {
  W.bytes.insert(W.bytes.end(), p, p + len);
  W.offs.push_back((uint32_t)W.bytes.size());
}

// In-place Ukrainian lowercasing, restricted to the cases that occur in the
// corpora: ASCII A-Z and the 2-byte Cyrillic block. Every mapped codepoint
// stays 2-byte in UTF-8 (А-Я -> а-я, Ё/Є/І/Ї -> ё/є/і/ї, Ґ -> ґ), so byte
// lengths are preserved and offsets need no fixing up. The production pipeline
// applies ICU's full Ukrainian lowercasing before lookup; this covers the same
// inputs without pulling ICU into the benchmark.
static void lowercaseUk(std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const uint8_t c = (uint8_t)s[i];
    if (c < 0x80) {
      if (c >= 'A' && c <= 'Z') s[i] = (char)(c + 32);
      ++i;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
      const uint32_t cp = ((c & 0x1Fu) << 6) | ((uint8_t)s[i + 1] & 0x3Fu);
      uint32_t lo = cp;
      if (cp >= 0x410 && cp <= 0x42F) lo = cp + 0x20;        // А-Я
      else if (cp >= 0x400 && cp <= 0x40F) lo = cp + 0x50;   // Ё, Є, І, Ї, ...
      else if (cp == 0x490) lo = 0x491;                      // Ґ
      if (lo != cp) {
        s[i] = (char)(0xC0 | (lo >> 6));
        s[i + 1] = (char)(0x80 | (lo & 0x3F));
      }
      i += 2;
    } else {
      i += (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    }
  }
}

static bool loadWordsFile(const std::string& path, uint32_t maxN, bool lower,
                          WordsHost& W, std::vector<std::string>& vocab) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  W.offs = {0};
  std::unordered_set<std::string> seen;
  std::string line;
  while (std::getline(in, line) && (maxN == 0 || W.n() < maxN)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (line.empty() || line.size() > 250) continue;
    if (lower) lowercaseUk(line);
    appendWord(W, (const uint8_t*)line.data(), line.size());
    if (seen.insert(line).second) vocab.push_back(line);
  }
  return W.n() > 0;
}

// Synthetic fallback: pseudo-Ukrainian words (2-byte Cyrillic UTF-8) with a
// Zipf-ish frequency draw, so the benchmark runs without a corpus.
static void genWords(uint32_t nSamples, uint32_t vocabSize, uint64_t seed,
                     WordsHost& W, std::vector<std::string>& vocab) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> lenD(3, 12);  // letters
  std::uniform_int_distribution<int> chD(0, 31);   // а..я
  std::unordered_set<std::string> seen;
  vocab.reserve(vocabSize);
  while (vocab.size() < vocabSize) {
    int L = lenD(rng);
    std::string w;
    w.reserve(2 * L);
    for (int i = 0; i < L; ++i) {
      int r = chD(rng);
      if (r < 16) {
        w.push_back((char)0xD0);
        w.push_back((char)(0xB0 + r));
      } else {
        w.push_back((char)0xD1);
        w.push_back((char)(0x80 + (r - 16)));
      }
    }
    if (seen.insert(w).second) vocab.push_back(w);
  }
  std::vector<double> wts(vocabSize);
  for (uint32_t r = 0; r < vocabSize; ++r) wts[r] = 1.0 / (r + 1.0);
  std::discrete_distribution<uint32_t> zipf(wts.begin(), wts.end());
  W.offs = {0};
  W.bytes.reserve((size_t)nSamples * 12);
  for (uint32_t i = 0; i < nSamples; ++i) {
    const std::string& w = vocab[zipf(rng)];
    appendWord(W, (const uint8_t*)w.data(), w.size());
  }
}

// ------------------------------------------------------------------- helpers
static constexpr int TPB = 256;
static inline uint32_t blocksFor(uint32_t n) { return (n + TPB - 1) / TPB; }

struct Result {
  std::string name;
  double prepMs = 0;    // sorting / partitioning cost
  double lookupMs = 0;  // trie traversal kernel time
  double totalMs = 0;
};

static void printResult(const Result& r, uint32_t n) {
  double kernelThr = n / (r.lookupMs * 1e3);   // Mwords/s
  double e2eThr = n / (r.totalMs * 1e3);
  printf("%-22s prep %8.3f ms   lookup %8.3f ms   total %8.3f ms   "
         "kernel %8.1f Mw/s   e2e %8.1f Mw/s\n",
         r.name.c_str(), r.prepMs, r.lookupMs, r.totalMs, kernelThr, e2eThr);
}

// An ordering, decoupled from its measurement: `d == nullptr` means identity
// (baseline). Building every permutation first and timing the lookup kernels
// afterwards is not a stylistic choice — see `measureAll` below.
struct Perm {
  std::string name;
  double prepMs = 0;
  uint32_t* d = nullptr;
  int passes = -1;  // refinement passes, variant 2b only
  // Variant 6 only: words materialised in permuted order, so the lookup reads
  // them sequentially instead of gathering through `d`.
  uint8_t* cBytes = nullptr;
  uint32_t* cOffs = nullptr;
  std::vector<double> times;  // per-round lookup times (ms)

  double median() const {
    if (times.empty()) return 0;
    std::vector<double> v = times;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
  }
  double best() const {
    return times.empty() ? 0
                         : *std::min_element(times.begin(), times.end());
  }
  double spread() const {  // max/min, a drift/noise indicator
    if (times.empty()) return 0;
    auto mm = std::minmax_element(times.begin(), times.end());
    return *mm.first > 0 ? *mm.second / *mm.first : 0;
  }
};

static void checksum(const char* tag, const std::vector<int32_t>& out) {
  uint64_t h = 1469598103934665603ull;
  for (int32_t v : out) h = (h ^ (uint32_t)v) * 1099511628211ull;
  printf("  %-20s checksum %016llx\n", tag, (unsigned long long)h);
}

static std::vector<int32_t> download(const int32_t* d, uint32_t n) {
  std::vector<int32_t> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  return h;
}

// ------------------------------------------------------------------ variants
static Perm permCpuSort(const WordsHost& H, int keyMode, bool fullTieBreak) {
  Perm r{"1 cpu-sort"};
  const uint32_t n = H.n();

  auto t0 = Clock::now();
  std::vector<uint64_t> keys(n);
  for (uint32_t i = 0; i < n; ++i)
    keys[i] = makeSortKey(H.bytes.data() + H.offs[i], H.offs[i + 1] - H.offs[i],
                          keyMode);
  std::vector<uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  std::sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
    if (keys[a] != keys[b]) return keys[a] < keys[b];
    if (!fullTieBreak) return a < b;  // stable-ish, prefix-key only
    uint32_t la = H.offs[a + 1] - H.offs[a], lb = H.offs[b + 1] - H.offs[b];
    int c = memcmp(H.bytes.data() + H.offs[a], H.bytes.data() + H.offs[b],
                   la < lb ? la : lb);
    if (c) return c < 0;
    return la < lb;
  });
  r.prepMs = msSince(t0);

  // The perm upload is part of what this strategy costs on a real pipeline.
  CUDA_CHECK(cudaMalloc(&r.d, n * sizeof(uint32_t)));
  auto t1 = Clock::now();
  CUDA_CHECK(cudaMemcpy(r.d, perm.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  r.prepMs += msSince(t1);
  return r;
}

// Shared by variants 2 and 3: CUB radix sort over [beginBit, 64).
static Perm permGpuSort(const WordsDev& D, int keyMode, int prefixBytes,
                        const char* name) {
  Perm r{name};
  const uint32_t n = D.n;
  const int beginBit = (8 - prefixBytes) * 8;  // sort only top prefixBytes

  uint64_t *dKeysIn = nullptr, *dKeysOut = nullptr;
  uint32_t *dIdxIn = nullptr, *dPerm = nullptr;
  CUDA_CHECK(cudaMalloc(&dKeysIn, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dKeysOut, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dIdxIn, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dPerm, n * sizeof(uint32_t)));
  r.d = dPerm;

  void* dTemp = nullptr;
  size_t tempBytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, dKeysIn,
                                             dKeysOut, dIdxIn, dPerm, n,
                                             beginBit, 64));
  CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

  GpuTimer tp;
  tp.start();
  makeKeysKernel<<<blocksFor(n), TPB>>>(D.bytes, D.offs, n, 0, keyMode, dKeysIn,
                                        dIdxIn);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, dKeysIn,
                                             dKeysOut, dIdxIn, dPerm, n,
                                             beginBit, 64));
  r.prepMs = tp.stopMs();

  cudaFree(dTemp);
  cudaFree(dKeysIn);
  cudaFree(dKeysOut);
  cudaFree(dIdxIn);
  return r;  // r.d == dPerm, kept for the measurement phase
}

// Variant 2b: exact full-depth GPU sort. One full radix pass on the initial
// 64-bit key, then iterative refinement: detect runs of equal keys ("ties"),
// compute the next 8 bytes for those positions, and segmented-sort each run.
// Ties beyond the first key are rare in real text, so later passes touch a
// vanishing fraction of the array; runs of *identical* tokens (frequent under
// Zipf) are excluded via the runActive filter so they never keep the loop
// alive or get pointlessly re-sorted.
//
// Refinement byte offset per depth d >= 1, chosen so the final order is exact
// for every key mode:
//   alpha:    8*d          (initial key covered bytes 0..7)
//   len:      8*(d-1)      (initial key covered length only -> len-then-alpha)
//   lenalpha: 7 + 8*(d-1)  (initial key covered len + bytes 0..6)
static inline uint32_t refineOffset(int keyMode, int d) {
  switch (keyMode) {
    case KEY_LEN: return 8u * (uint32_t)(d - 1);
    case KEY_LEN_ALPHA: return 7u + 8u * (uint32_t)(d - 1);
    default: return 8u * (uint32_t)d;
  }
}

struct TempBuf {
  void* p = nullptr;
  size_t cap = 0;
  void ensure(size_t bytes) {
    if (bytes > cap) {
      if (p) cudaFree(p);
      CUDA_CHECK(cudaMalloc(&p, bytes));
      cap = bytes;
    }
  }
  ~TempBuf() { cudaFree(p); }
};

static Perm permGpuFullSort(const WordsDev& D, int keyMode) {
  Perm r{"2b gpu-full"};
  const uint32_t n = D.n;

  uint64_t *dKeys = nullptr, *dKeysB = nullptr;
  uint32_t *dPerm = nullptr, *dPermB = nullptr;
  uint32_t *dHead = nullptr, *dSegId = nullptr, *dStarts = nullptr;
  uint32_t *dSegBegin = nullptr, *dSegEnd = nullptr;
  uint32_t *dCtr = nullptr, *dNumRuns = nullptr;
  uint8_t* dRunActive = nullptr;
  CUDA_CHECK(cudaMalloc(&dKeys, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dKeysB, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dPerm, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dPermB, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dHead, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dSegId, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dStarts, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dSegBegin, (n / 2 + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dSegEnd, (n / 2 + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dCtr, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dNumRuns, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dRunActive, n * sizeof(uint8_t)));
  TempBuf temp;

  GpuTimer tp;
  tp.start();

  // Depth 0: full radix sort on the initial key (identical to variant 2).
  makeKeysKernel<<<blocksFor(n), TPB>>>(D.bytes, D.offs, n, 0, keyMode, dKeys,
                                        dPerm);  // dPerm <- identity
  {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, dKeys, dKeysB,
                                               dPerm, dPermB, n, 0, 64));
    temp.ensure(bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.p, bytes, dKeys, dKeysB,
                                               dPerm, dPermB, n, 0, 64));
    std::swap(dKeys, dKeysB);  // dKeys/dPerm now hold the sorted state
    std::swap(dPerm, dPermB);
  }
  CUDA_CHECK(cudaMemset(dHead, 0, n * sizeof(uint32_t)));

  int passes = 0;
  for (int d = 1; d <= 32; ++d) {
    // Boundaries from the keys just sorted at depth d-1.
    headUpdateKernel<<<blocksFor(n), TPB>>>(dKeys, n, dHead);
    {
      size_t bytes = 0;
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, dHead, dSegId,
                                               n));
      temp.ensure(bytes);
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp.p, bytes, dHead, dSegId,
                                               n));
    }
    // Depth-d keys for every position (cheap; only tie runs get sorted).
    makeKeysAtKernel<<<blocksFor(n), TPB>>>(D.bytes, D.offs, n, 0, dPerm,
                                            refineOffset(keyMode, d), dKeys);
    CUDA_CHECK(cudaMemset(dRunActive, 0, n * sizeof(uint8_t)));
    markActiveKernel<<<blocksFor(n), TPB>>>(dKeys, dSegId, n, dRunActive);

    // Run starts = positions with head flag set.
    {
      cub::CountingInputIterator<uint32_t> cnt(0);
      size_t bytes = 0;
      CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, cnt, dHead,
                                            dStarts, dNumRuns, n));
      temp.ensure(bytes);
      CUDA_CHECK(cub::DeviceSelect::Flagged(temp.p, bytes, cnt, dHead,
                                            dStarts, dNumRuns, n));
    }
    uint32_t numRuns = 0;
    CUDA_CHECK(cudaMemcpy(&numRuns, dNumRuns, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dCtr, 0, sizeof(uint32_t)));
    buildTieSegsKernel<<<blocksFor(numRuns), TPB>>>(dStarts, numRuns, n,
                                                    dRunActive, dSegBegin,
                                                    dSegEnd, dCtr);
    uint32_t numSegs = 0;
    CUDA_CHECK(cudaMemcpy(&numSegs, dCtr, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    if (numSegs == 0) break;  // fully ordered (identical-token runs excluded)
    ++passes;

    // Segmented sort touches only covered ranges; pre-copy so uncovered
    // positions carry over into the output buffers.
    CUDA_CHECK(cudaMemcpy(dKeysB, dKeys, n * sizeof(uint64_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dPermB, dPerm, n * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    {
      size_t bytes = 0;
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
          nullptr, bytes, dKeys, dKeysB, dPerm, dPermB, n, numSegs, dSegBegin,
          dSegEnd));
      temp.ensure(bytes);
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
          temp.p, bytes, dKeys, dKeysB, dPerm, dPermB, n, numSegs, dSegBegin,
          dSegEnd));
    }
    std::swap(dKeys, dKeysB);
    std::swap(dPerm, dPermB);
  }
  r.prepMs = tp.stopMs();
  r.passes = passes;
  r.d = dPerm;  // after the final swap, dPerm holds the sorted permutation

  cudaFree(dKeys);
  cudaFree(dKeysB);
  cudaFree(dPermB);
  cudaFree(dHead);
  cudaFree(dSegId);
  cudaFree(dStarts);
  cudaFree(dSegBegin);
  cudaFree(dSegEnd);
  cudaFree(dCtr);
  cudaFree(dNumRuns);
  cudaFree(dRunActive);
  return r;
}

// Variant 4: one hand-rolled radix pass — histogram + exclusive scan + scatter.
// O(n), two data-touching kernels, gives bin-level (approximate) ordering.
static Perm permGpuPartition(const WordsDev& D, int keyMode, int binBits) {
  Perm r{"4 gpu-partition"};
  const uint32_t n = D.n;
  const uint32_t numBins = 1u << binBits;

  uint64_t* dKeys = nullptr;
  uint32_t *dHist = nullptr, *dStart = nullptr, *dCursor = nullptr,
           *dPerm = nullptr;
  CUDA_CHECK(cudaMalloc(&dKeys, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&dHist, numBins * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dStart, numBins * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dCursor, numBins * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&dPerm, n * sizeof(uint32_t)));

  void* dTemp = nullptr;
  size_t tempBytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTemp, tempBytes, dHist, dStart,
                                           numBins));
  CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

  GpuTimer tp;
  tp.start();
  makeKeysKernel<<<blocksFor(n), TPB>>>(D.bytes, D.offs, n, 0, keyMode, dKeys,
                                        nullptr);
  CUDA_CHECK(cudaMemsetAsync(dHist, 0, numBins * sizeof(uint32_t)));
  histKernel<<<blocksFor(n), TPB>>>(dKeys, n, dHist, binBits);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTemp, tempBytes, dHist, dStart,
                                           numBins));
  CUDA_CHECK(cudaMemcpyAsync(dCursor, dStart, numBins * sizeof(uint32_t),
                             cudaMemcpyDeviceToDevice));
  scatterKernel<<<blocksFor(n), TPB>>>(dKeys, n, dCursor, binBits, dPerm);
  r.prepMs = tp.stopMs();
  r.d = dPerm;

  cudaFree(dTemp);
  cudaFree(dKeys);
  cudaFree(dHist);
  cudaFree(dStart);
  cudaFree(dCursor);
  return r;
}

// Variant 6: sort as in variant 2/3, then materialise the words in that order.
// Prep therefore carries both the sort and one gather pass over the corpus.
static Perm permSortCompact(const WordsDev& D, int keyMode, int prefixBytes,
                            const char* name) {
  Perm r = permGpuSort(D, keyMode, prefixBytes, name);
  const uint32_t n = D.n;

  uint32_t* dLens = nullptr;
  CUDA_CHECK(cudaMalloc(&dLens, (n + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&r.cOffs, (n + 1) * sizeof(uint32_t)));

  void* dTemp = nullptr;
  size_t tempBytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTemp, tempBytes, dLens, r.cOffs,
                                           n + 1));
  CUDA_CHECK(cudaMalloc(&dTemp, tempBytes));

  GpuTimer tp;
  tp.start();
  gatherLenKernel<<<blocksFor(n), TPB>>>(D.offs, n, r.d, dLens);
  CUDA_CHECK(cudaMemset(dLens + n, 0, sizeof(uint32_t)));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(dTemp, tempBytes, dLens, r.cOffs,
                                           n + 1));
  uint32_t totalBytes = 0;
  CUDA_CHECK(cudaMemcpy(&totalBytes, r.cOffs + n, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMalloc(&r.cBytes, totalBytes));
  gatherBytesKernel<<<blocksFor(n), TPB>>>(D.bytes, D.offs, n, r.d, r.cOffs,
                                           r.cBytes);
  r.prepMs += tp.stopMs();

  cudaFree(dLens);
  cudaFree(dTemp);
  return r;
}

// Prep must be measured with the same discipline as lookup. Timed once, a
// single CUB tile-config change or a clock blip lands directly on the
// break-even estimate — which is the number the paper's recommendation turns
// on. Build k times, keep the median, and hold only one copy live at a time so
// repetition costs time rather than memory.
static Perm buildRepeated(const std::function<Perm()>& f, int k) {
  std::vector<double> t;
  Perm keep;
  for (int i = 0; i < k; ++i) {
    Perm p = f();
    t.push_back(p.prepMs);
    if (i + 1 < k) {
      cudaFree(p.d);
      cudaFree(p.cBytes);
      cudaFree(p.cOffs);
    } else {
      keep = p;
    }
  }
  std::sort(t.begin(), t.end());
  keep.prepMs = t[t.size() / 2];
  return keep;
}

// ------------------------------------------------- measurement discipline
// On a consumer GPU without root, clocks cannot be pinned (`nvidia-smi -lgc`
// needs privileges), and an idle GPU drops to ~210 MHz core / 405 MHz memory
// against a 3105 / 11501 MHz maximum. Any host-side work between timed kernels
// — variant 1's multi-second std::sort above all — therefore parks the clocks,
// and the variants that happen to run next measure a ramping GPU rather than
// their own locality. Measuring in declaration order once per variant produced
// a monotone "improvement" that tracked execution order exactly.
//
// Two mechanisms remove that confound:
//   (a) all permutations are built first, so no host work sits between timed
//       kernels;
//   (b) the timed kernels run round-robin over R rounds after a busy warm-up,
//       so any residual drift is spread evenly across variants instead of
//       accumulating in whichever ran last. Per-variant medians are reported,
//       and max/min per variant is printed as a drift indicator.
static void launchLookup(TrieDev trie, const WordsDev& D, const Perm& p,
                         int32_t* dOut) {
  if (p.cBytes)
    lookupCompactKernel<<<blocksFor(D.n), TPB>>>(trie, p.cBytes, p.cOffs, D.n,
                                                 p.d, dOut);
  else
    lookupKernel<<<blocksFor(D.n), TPB>>>(trie, D.bytes, D.offs, D.n, 0, p.d,
                                          dOut);
}

static void warmUp(TrieDev trie, const WordsDev& D, int32_t* dOut,
                   double budgetMs) {
  auto t0 = Clock::now();
  while (msSince(t0) < budgetMs) {
    for (int k = 0; k < 8; ++k)
      lookupKernel<<<blocksFor(D.n), TPB>>>(trie, D.bytes, D.offs, D.n, 0,
                                            nullptr, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

static void measureAll(TrieDev trie, const WordsDev& D, int32_t* dOut,
                       std::vector<Perm>& perms, int rounds, int reps) {
  for (int r = 0; r < rounds; ++r) {
    for (Perm& p : perms) {
      GpuTimer t;
      t.start();
      for (int k = 0; k < reps; ++k) launchLookup(trie, D, p, dOut);
      p.times.push_back(t.stopMs() / reps);
    }
  }
}

// ----------------------------------------------------- variant 5: streaming
// Double-buffered batches. For batch k: (copy stream) H2D + keys + partition,
// then each of S worker streams runs lookups on its fixed slice of the prefix
// space. Bin ranges -> streams is a *static* mapping, so worker stream s sees
// the same first-letter range in every batch ("feed the same kernel similar
// data"), while the O(n) partition avoids ever rebuilding a sorted corpus.
struct StreamSlot {
  uint8_t* bytes = nullptr;
  uint32_t* offs = nullptr;
  uint64_t* keys = nullptr;
  uint32_t *hist = nullptr, *start = nullptr, *cursor = nullptr,
           *perm = nullptr, *segBounds = nullptr;
  void* scanTemp = nullptr;
  size_t scanTempBytes = 0;
  cudaEvent_t ready{};
  std::vector<cudaEvent_t> done;  // one per worker stream
};

static Result runStreaming(TrieDev trie, const WordsHost& H, int keyMode,
                           int binBits, uint32_t batchWords, int numStreams,
                           bool partitionOn, int32_t* dOut) {
  Result r{partitionOn ? "5 streaming+bins" : "5 streaming-nobins"};
  const uint32_t n = H.n();
  const uint32_t numBins = 1u << binBits;
  const uint32_t numBatches = (n + batchWords - 1) / batchWords;

  // Pinned host staging for async copies.
  uint8_t* hBytes = nullptr;
  uint32_t* hOffs = nullptr;
  CUDA_CHECK(cudaHostAlloc(&hBytes, H.bytes.size(), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&hOffs, H.offs.size() * sizeof(uint32_t),
                           cudaHostAllocDefault));
  memcpy(hBytes, H.bytes.data(), H.bytes.size());
  memcpy(hOffs, H.offs.data(), H.offs.size() * sizeof(uint32_t));

  // Max batch byte size for buffer allocation.
  size_t maxBatchBytes = 0;
  for (uint32_t b = 0; b < numBatches; ++b) {
    uint32_t w0 = b * batchWords;
    uint32_t w1 = std::min(n, w0 + batchWords);
    maxBatchBytes = std::max(maxBatchBytes, (size_t)(H.offs[w1] - H.offs[w0]));
  }

  const int SLOTS = 2;
  StreamSlot slot[SLOTS];
  for (int s = 0; s < SLOTS; ++s) {
    CUDA_CHECK(cudaMalloc(&slot[s].bytes, maxBatchBytes));
    CUDA_CHECK(cudaMalloc(&slot[s].offs, (batchWords + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&slot[s].keys, batchWords * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&slot[s].hist, numBins * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&slot[s].start, numBins * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&slot[s].cursor, numBins * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&slot[s].perm, batchWords * sizeof(uint32_t)));
    CUDA_CHECK(
        cudaMalloc(&slot[s].segBounds, (numStreams + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(slot[s].scanTemp,
                                             slot[s].scanTempBytes,
                                             slot[s].hist, slot[s].start,
                                             numBins));
    CUDA_CHECK(cudaMalloc(&slot[s].scanTemp, slot[s].scanTempBytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&slot[s].ready,
                                        cudaEventDisableTiming));
    slot[s].done.resize(numStreams);
    for (auto& ev : slot[s].done)
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  }

  cudaStream_t copyStream;
  CUDA_CHECK(cudaStreamCreate(&copyStream));
  std::vector<cudaStream_t> work(numStreams);
  for (auto& ws : work) CUDA_CHECK(cudaStreamCreate(&ws));

  auto t0 = Clock::now();
  for (uint32_t b = 0; b < numBatches; ++b) {
    StreamSlot& S = slot[b % SLOTS];
    uint32_t w0 = b * batchWords;
    uint32_t w1 = std::min(n, w0 + batchWords);
    uint32_t bn = w1 - w0;
    uint32_t base = H.offs[w0];
    size_t nbytes = H.offs[w1] - base;

    // Before reusing this slot (batch b - SLOTS), make the copy stream wait
    // device-side until that batch's workers finished reading the buffers.
    if (b >= (uint32_t)SLOTS)
      for (auto& ev : S.done)
        CUDA_CHECK(cudaStreamWaitEvent(copyStream, ev, 0));

    CUDA_CHECK(cudaMemcpyAsync(S.bytes, hBytes + base, nbytes,
                               cudaMemcpyHostToDevice, copyStream));
    CUDA_CHECK(cudaMemcpyAsync(S.offs, hOffs + w0,
                               (bn + 1) * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, copyStream));
    if (partitionOn) {
      makeKeysKernel<<<blocksFor(bn), TPB, 0, copyStream>>>(
          S.bytes, S.offs, bn, base, keyMode, S.keys, nullptr);
      CUDA_CHECK(cudaMemsetAsync(S.hist, 0, numBins * sizeof(uint32_t),
                                 copyStream));
      histKernel<<<blocksFor(bn), TPB, 0, copyStream>>>(S.keys, bn, S.hist,
                                                        binBits);
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(S.scanTemp, S.scanTempBytes,
                                               S.hist, S.start, numBins,
                                               copyStream));
      CUDA_CHECK(cudaMemcpyAsync(S.cursor, S.start,
                                 numBins * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice, copyStream));
      scatterKernel<<<blocksFor(bn), TPB, 0, copyStream>>>(S.keys, bn, S.cursor,
                                                           binBits, S.perm);
      segBoundsKernel<<<1, numStreams + 1, 0, copyStream>>>(
          S.start, bn, numBins, numStreams, S.segBounds);
    }
    CUDA_CHECK(cudaEventRecord(S.ready, copyStream));

    if (partitionOn) {
      for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamWaitEvent(work[s], S.ready, 0));
        // Grid sized for the whole batch; threads outside the segment exit.
        lookupSegmentKernel<<<blocksFor(bn), TPB, 0, work[s]>>>(
            trie, S.bytes, S.offs, base, S.perm, S.segBounds, s, dOut + w0);
      }
    } else {
      CUDA_CHECK(cudaStreamWaitEvent(work[b % numStreams], S.ready, 0));
      lookupKernel<<<blocksFor(bn), TPB, 0, work[b % numStreams]>>>(
          trie, S.bytes, S.offs, bn, base, nullptr, dOut + w0);
    }
    for (int s = 0; s < numStreams; ++s)
      CUDA_CHECK(cudaEventRecord(S.done[s], work[s]));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  r.totalMs = msSince(t0);
  r.lookupMs = r.totalMs;  // fully overlapped pipeline: report e2e only

  for (auto& ws : work) cudaStreamDestroy(ws);
  cudaStreamDestroy(copyStream);
  for (int s = 0; s < SLOTS; ++s) {
    cudaFree(slot[s].bytes);
    cudaFree(slot[s].offs);
    cudaFree(slot[s].keys);
    cudaFree(slot[s].hist);
    cudaFree(slot[s].start);
    cudaFree(slot[s].cursor);
    cudaFree(slot[s].perm);
    cudaFree(slot[s].segBounds);
    cudaFree(slot[s].scanTemp);
    cudaEventDestroy(slot[s].ready);
    for (auto& ev : slot[s].done) cudaEventDestroy(ev);
  }
  cudaFreeHost(hBytes);
  cudaFreeHost(hOffs);
  return r;
}

// ----------------------------------------------------------------------- main
static int argInt(int argc, char** argv, const char* flag, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], flag)) return atoi(argv[i + 1]);
  return def;
}
static const char* argStr(int argc, char** argv, const char* flag,
                          const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], flag)) return argv[i + 1];
  return def;
}

int main(int argc, char** argv) {
  const char* file = argStr(argc, argv, "--words", "");
  const uint32_t nGen = (uint32_t)argInt(argc, argv, "--n", 20'000'000);
  const uint32_t vocabSize = (uint32_t)argInt(argc, argv, "--vocab", 300'000);
  const int prefixBytes = argInt(argc, argv, "--prefix", 4);
  const int binBits = argInt(argc, argv, "--binbits", 16);
  const uint32_t batchWords =
      (uint32_t)argInt(argc, argv, "--batch", 2'000'000);
  const int numStreams = argInt(argc, argv, "--streams", 4);
  const int reps = argInt(argc, argv, "--reps", 5);
  const int rounds = argInt(argc, argv, "--rounds", 7);
  const int warmMs = argInt(argc, argv, "--warm", 2000);
  const char* keyStr = argStr(argc, argv, "--key", "alpha");
  const int keyMode = !strcmp(keyStr, "len")        ? KEY_LEN
                      : !strcmp(keyStr, "lenalpha") ? KEY_LEN_ALPHA
                                                    : KEY_ALPHA;
  const bool lower = argInt(argc, argv, "--lower", 1) != 0;
  const int touchLemma = argInt(argc, argv, "--lemma", 0);
  const char* trieDir = argStr(argc, argv, "--trie", ".");

  WordsHost H;
  std::vector<std::string> vocab;
  if (file[0] && loadWordsFile(file, nGen, lower, H, vocab)) {
    printf("loaded %u words (%zu unique, type/token %.4f) from %s%s\n", H.n(),
           vocab.size(), (double)vocab.size() / H.n(), file,
           lower ? " [lowercased]" : "");
  } else {
    if (file[0]) fprintf(stderr, "could not read %s, using synthetic data\n",
                         file);
    genWords(nGen, vocabSize, 42, H, vocab);
    printf("generated %u synthetic words, vocab %zu\n", H.n(), vocab.size());
  }

  TrieDevStorage trieStore;
#ifdef USE_REAL_TRIE
  if (!trieStore.load(trieDir, touchLemma)) return 1;
  TrieDev trie = trieStore.dev();
#else
  (void)trieDir;
  (void)touchLemma;
  TrieBuilder tb;
  for (size_t i = 0; i < vocab.size(); ++i)
    tb.insert((const uint8_t*)vocab[i].data(), (uint32_t)vocab[i].size(),
              (int32_t)i);
  TrieFlatHost flat = flattenTrie(tb);
  printf("trie: %zu states, %zu edges\n", flat.firstEdge.size(),
         flat.edgeLabel.size());
  TrieDev trie = trieStore.upload(flat);
#endif

  WordsDev D;
  D.upload(H);
  int32_t* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, D.n * sizeof(int32_t)));

  printf("\nkey mode: %s | prefix bytes: %d | bin bits: %d | batch: %u | "
         "streams: %d | reps: %d\n\n",
         keyStr, prefixBytes, binBits, batchWords, numStreams, reps);

  // ---- phase A: build every ordering (host work happens only here) --------
  std::vector<Perm> perms;
  perms.push_back(Perm{"0 baseline"});  // identity: d == nullptr
  // Variant 1 costs tens of seconds of host time at 50 M tokens and its
  // ordering is reproduced exactly by 2b, so sweeps can skip it.
  if (argInt(argc, argv, "--nocpu", 0) == 0)
    perms.push_back(permCpuSort(H, keyMode, /*fullTieBreak=*/true));
  const int pr = argInt(argc, argv, "--prepreps", 5);  // median-of-k prep
  perms.push_back(buildRepeated(
      [&] { return permGpuSort(D, keyMode, 8, "2 gpu-sort-8B"); }, pr));
  perms.push_back(
      buildRepeated([&] { return permGpuFullSort(D, keyMode); }, pr));
  perms.push_back(buildRepeated(
      [&] { return permGpuSort(D, keyMode, prefixBytes, "3 gpu-prefix"); },
      pr));
  perms.push_back(
      buildRepeated([&] { return permGpuPartition(D, keyMode, binBits); }, pr));
  perms.push_back(buildRepeated(
      [&] { return permSortCompact(D, keyMode, 8, "6 sort+compact"); }, pr));
  perms.push_back(buildRepeated(
      [&] { return permSortCompact(D, keyMode, 2, "6b sort-2B+compact"); },
      pr));

  for (const Perm& p : perms)
    if (p.passes >= 0)
      printf("  (2b: %d refinement pass%s beyond the initial sort)\n",
             p.passes, p.passes == 1 ? "" : "es");

  // ---- correctness: every ordering must reproduce the baseline result -----
  std::vector<int32_t> ref;
  for (const Perm& p : perms) {
    launchLookup(trie, D, p, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got = download(dOut, D.n);
    if (ref.empty()) {
      ref = got;
      size_t hits = 0;
      for (int32_t v : ref) hits += (v >= 0);
      printf("\nhit rate: %zu / %u tokens recognised (%.2f%%)\n", hits, D.n,
             100.0 * hits / D.n);
      checksum("baseline", ref);
    } else if (got != ref) {
      printf("  !! %s MISMATCH vs baseline\n", p.name.c_str());
      checksum(p.name.c_str(), got);
    }
  }
  printf("all %zu orderings reproduce the baseline result\n", perms.size());

  // ---- phase B: warm the clocks, then time the kernels round-robin --------
  printf("\nwarming up (%d ms), then %d interleaved rounds x %d reps...\n\n",
         warmMs, rounds, reps);
  warmUp(trie, D, dOut, warmMs);
  measureAll(trie, D, dOut, perms, rounds, reps);

  const double base = perms[0].median();
  printf("%-22s %8s %10s %10s %8s %8s %12s\n", "variant", "prep ms",
         "lookup ms", "best ms", "max/min", "speedup", "kernel Mw/s");
  for (const Perm& p : perms) {
    const double lk = p.median();
    printf("%-22s %8.3f %10.3f %10.3f %8.3f %8.2fx %12.1f\n", p.name.c_str(),
           p.prepMs, lk, p.best(), p.spread(), base / lk, D.n / (lk * 1e3));
  }

  // Amortisation: how many times a permutation must be reused before its
  // build cost is repaid by the per-batch lookup saving.
  printf("\nbreak-even reuse count (prep / lookup saving vs baseline):\n");
  for (size_t i = 1; i < perms.size(); ++i) {
    const double gain = base - perms[i].median();
    if (gain > 0)
      printf("  %-22s %8.1f reuses\n", perms[i].name.c_str(),
             perms[i].prepMs / gain);
    else
      printf("  %-22s never (no lookup gain)\n", perms[i].name.c_str());
  }

  // ---- variant 5: streaming (its own timing basis — includes H2D) ---------
  printf("\nstreaming pipeline (includes H2D; compare the two rows to each "
         "other, not to the table above):\n");
  Result r5a = runStreaming(trie, H, keyMode, binBits, batchWords, numStreams,
                            /*partitionOn=*/false, dOut);
  printResult(r5a, D.n);
  checksum("stream-nobins", download(dOut, D.n));

  Result r5b = runStreaming(trie, H, keyMode, binBits, batchWords, numStreams,
                            /*partitionOn=*/true, dOut);
  printResult(r5b, D.n);
  checksum("stream+bins", download(dOut, D.n));

  for (Perm& p : perms) {
    cudaFree(p.d);
    cudaFree(p.cBytes);
    cudaFree(p.cOffs);
  }
  cudaFree(dOut);
  D.free();
  trieStore.free();
  return 0;
}
