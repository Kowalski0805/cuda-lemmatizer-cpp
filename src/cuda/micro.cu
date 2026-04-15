#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Ada Lovelace (RTX 4080 Super)
// L1 per SM: 128KB
// L2:        64MB
// Cache line: 128 bytes = 16 x uint64

#define STRIDE     16ULL   // elements between hops (one cache line)
#define ITERS      4000

// Array sizes — in number of cache lines
// L1:   use 512 lines = 64KB  (fits in L1 128KB)
// L2:   use 256K lines = 32MB (exceeds L1, fits in L2 64MB)
// DRAM: use 2M lines  = 256MB (exceeds L2 64MB)
#define L1_LINES    512ULL
#define L2_LINES    (256ULL * 1024)
#define DRAM_LINES  (2ULL * 1024 * 1024)

// ----------------------------------------------------------------
// Build randomized pointer chain entirely on device.
// Uses a simple LCG to shuffle — no host involvement.
// Each element arr[order[i] * STRIDE] = &arr[order[i+1] * STRIDE]
// ----------------------------------------------------------------
__global__ void build_chain(uint64_t* arr, uint64_t n_lines, uint64_t seed) {
    // Generate a random permutation via LCG inside a single thread.
    // For large arrays this is slow but only runs once.
    // LCG: x = (a*x + c) % n_lines  with full-period constants
    uint64_t a = 6364136223846793005ULL;
    uint64_t c = 1442695040888963407ULL;

    uint64_t* prev = arr;  // start of chain
    uint64_t x = seed;

    for (uint64_t i = 0; i < n_lines - 1; i++) {
        x = a * x + c;
        uint64_t next_line = x % n_lines;
        // store device pointer to next element
        *prev = (uint64_t)(arr + next_line * STRIDE);
        prev = arr + next_line * STRIDE;
    }
    // close the loop
    *prev = (uint64_t)arr;
}

// ----------------------------------------------------------------
// Thrash L2: write to a 128MB buffer sequentially to evict
// everything from L2 before DRAM measurement.
// ----------------------------------------------------------------
__global__ void thrash_l2(uint64_t* buf, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        buf[i] = i;
}

// ----------------------------------------------------------------
// Chase kernels — single thread, dependent loads
// ----------------------------------------------------------------
__global__ void chase_l1(uint64_t* arr, uint64_t* result) {
    uint64_t* ptr = arr;

    // Warmup: populate L1
    #pragma unroll 1
    for (int i = 0; i < 200; i++)
        asm volatile("ld.global.ca.u64 %0, [%0];" : "+l"(*(uint64_t*)&ptr) :: "memory");

    ptr = arr;
    uint64_t start = clock64();
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("ld.global.ca.u64 %0, [%0];" : "+l"(*(uint64_t*)&ptr) :: "memory");
    uint64_t end = clock64();

    *result = (end - start) / ITERS;
    if (ptr == 0) *result = 0;
}

__global__ void chase_l2(uint64_t* arr, uint64_t* result) {
    uint64_t* ptr = arr;
    uint64_t start = clock64();
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(*(uint64_t*)&ptr) :: "memory");
    uint64_t end = clock64();

    *result = (end - start) / ITERS;
    if (ptr == 0) *result = 0;
}

// For DRAM: thrash L2 before each run, then use cg load.
// Since the array (256MB) >> L2 (64MB), after thrashing the
// chain data won't be in L2 and first-touch loads go to DRAM.
__global__ void chase_dram(uint64_t* arr, uint64_t* result) {
    uint64_t* ptr = arr;
    uint64_t start = clock64();
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(*(uint64_t*)&ptr) :: "memory");
    uint64_t end = clock64();

    *result = (end - start) / ITERS;
    if (ptr == 0) *result = 0;
}

// ----------------------------------------------------------------

void measure(const char* name, uint64_t n_lines, bool thrash,
             uint64_t* d_thrash_buf, uint64_t thrash_elems) {

    uint64_t n_elems = n_lines * STRIDE;
    size_t   bytes   = n_elems * sizeof(uint64_t);

    uint64_t *d_arr, *d_result, h_result;
    cudaMalloc(&d_arr,   bytes);
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemset(d_arr, 0, bytes);

    // Build chain on device with a single thread (avoids host pointer math)
    build_chain<<<1,1>>>(d_arr, n_lines, 12345ULL);
    cudaDeviceSynchronize();

    // Warmup run
    if (thrash) {
        thrash_l2<<<256,256>>>(d_thrash_buf, thrash_elems);
        cudaDeviceSynchronize();
        chase_dram<<<1,1>>>(d_arr, d_result);
    } else if (strcmp(name, "L1") == 0) {
        chase_l1<<<1,1>>>(d_arr, d_result);
    } else {
        chase_l2<<<1,1>>>(d_arr, d_result);
    }
    cudaDeviceSynchronize();

    // Measure
    uint64_t total = 0;
    int runs = 10;
    for (int r = 0; r < runs; r++) {
        if (thrash) {
            thrash_l2<<<256,256>>>(d_thrash_buf, thrash_elems);
            cudaDeviceSynchronize();
        }
        if (thrash)
            chase_dram<<<1,1>>>(d_arr, d_result);
        else if (strcmp(name, "L1") == 0)
            chase_l1<<<1,1>>>(d_arr, d_result);
        else
            chase_l2<<<1,1>>>(d_arr, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        total += h_result;
    }

    int clock_khz;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    double cycles = (double)(total / runs);
    double ns     = cycles / (clock_khz / 1e6);

    printf("%-6s | %6llu KB | %7.1f cycles | %6.1f ns\n",
           name, (unsigned long long)(bytes / 1024), cycles, ns);

    cudaFree(d_arr);
    cudaFree(d_result);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s  |  SM clock: %d MHz  |  L2: %d MB\n\n",
           prop.name, prop.clockRate / 1000, prop.l2CacheSize / (1024*1024));
    printf("%-6s | %8s | %15s | %s\n", "Level", "Array", "Latency(cycles)", "Latency(ns)");
    printf("----------------------------------------------------\n");

    // Thrash buffer: 128MB, used to evict L2 before DRAM measurement
    uint64_t thrash_elems = (128ULL * 1024 * 1024) / sizeof(uint64_t);
    uint64_t* d_thrash;
    cudaMalloc(&d_thrash, thrash_elems * sizeof(uint64_t));

    measure("L1",   L1_LINES,   false, d_thrash, thrash_elems);
    measure("L2",   L2_LINES,   false, d_thrash, thrash_elems);
    measure("DRAM", DRAM_LINES, true,  d_thrash, thrash_elems);

    cudaFree(d_thrash);
    return 0;
}
