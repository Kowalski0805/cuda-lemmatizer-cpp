#pragma once
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cstdint>

// ── Pass 1: mark word starts within a chunk ───────────────────────────────────
// chunk_start: byte offset of this chunk in the full file (for boundary check)
// prev_byte:   last byte of the previous chunk (-1 if first chunk)
__global__ void scan_mark_words(
    const char*    __restrict__ raw,       // chunk data
    size_t                      chunk_sz,
    char                        prev_byte, // '\n' or '\r' or -1 means "treat as sep"
    uint8_t*       __restrict__ flags,     // 1 = word start
    uint32_t*      __restrict__ wlen)      // word length at start, else 0
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_sz) return;

    const char c   = raw[i];
    const bool sep = (c == '\n' || c == '\r');

    char prev;
    if (i == 0)
        prev = prev_byte; // crosses chunk boundary
    else
        prev = raw[i - 1];
    const bool prev_sep = (prev == '\n' || prev == '\r' || prev == -1);
    const bool is_start = !sep && prev_sep;

    flags[i] = is_start ? 1 : 0;

    if (is_start) {
        uint32_t len = 0;
        while (i + len < chunk_sz) {
            if (raw[i + len] == '\n' || raw[i + len] == '\r') break;
            ++len;
        }
        wlen[i] = len;
    } else {
        wlen[i] = 0;
    }
}

// ── Pass 2: compact words into output buffers ─────────────────────────────────
// word_idx_base: word index offset accumulated from previous chunks
// char_off_base: char offset accumulated from previous chunks
__global__ void scan_compact(
    const char*     __restrict__ raw,
    size_t                       chunk_sz,
    const uint8_t*  __restrict__ flags,
    const uint32_t* __restrict__ wlen,
    const uint32_t*  __restrict__ word_idx,   // ExclusiveSum(flags) within chunk
    const uint32_t*  __restrict__ char_off,   // ExclusiveSum(wlen)  within chunk
    uint32_t                      word_idx_base,
    uint32_t                      char_off_base,
    char*           __restrict__ d_chars,
    uint32_t*        __restrict__ d_offsets)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_sz || !flags[i]) return;

    const uint32_t wi  = word_idx_base + word_idx[i];
    const uint32_t off = char_off_base + char_off[i];
    const uint32_t len = wlen[i];

    d_offsets[wi] = off;

    for (uint32_t k = 0; k < len; ++k)
        d_chars[off + k] = raw[i + k];
}

// ── Write sentinel d_offsets[N] = total_chars ─────────────────────────────────
__global__ void scan_write_sentinel(uint32_t* d_offsets, uint32_t N, uint32_t total_chars)
{
    d_offsets[N] = total_chars;
}