// common.cuh — shared helpers for the sorting-strategy benchmark
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  %s\n", #call, __FILE__,    \
              __LINE__, cudaGetErrorString(_e));                                \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

struct GpuTimer {
  cudaEvent_t evStart{}, evStop{};
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
  }
  ~GpuTimer() {
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(evStart, s)); }
  float stopMs(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(evStop, s));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    return ms;
  }
};

// Pack the first up to 8 bytes of a word into a big-endian uint64 so that
// numeric ordering of keys == bytewise lexicographic ordering of words.
// NOTE: for Ukrainian UTF-8 every Cyrillic letter is 2 bytes, so 8 bytes
// covers the first 4 letters of a word.
__host__ __device__ inline uint64_t packKey8(const uint8_t* w, uint32_t len) {
  uint64_t k = 0;
  uint32_t n = len < 8u ? len : 8u;
  for (uint32_t i = 0; i < n; ++i)
    k |= (uint64_t)w[i] << (56 - 8 * i);
  return k;
}

// Key modes for the alpha/len sorting comparison.
enum KeyMode : int {
  KEY_ALPHA = 0,      // key = first 8 bytes (4 Cyrillic letters)
  KEY_LEN = 1,        // key = byte length only (uniform loop trip counts per warp)
  KEY_LEN_ALPHA = 2,  // key = (len << 56) | first 7 bytes (length-major, alpha-minor)
};

__host__ __device__ inline uint64_t makeSortKey(const uint8_t* w, uint32_t len,
                                                int mode) {
  switch (mode) {
    case KEY_LEN:
      return (uint64_t)(len & 0xFFu) << 56;
    case KEY_LEN_ALPHA:
      return ((uint64_t)(len & 0xFFu) << 56) | (packKey8(w, len) >> 8);
    default:
      return packKey8(w, len);
  }
}
