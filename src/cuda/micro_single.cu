#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define ITERS 10000

__global__ void measure(uint32_t* arr, uint64_t* result) {
    uint64_t t1, t2, t3;
    uint32_t v1, v2;

    // First load: cold miss → goes to target cache level (or DRAM)
    // Second load: from same cache block → hits L1 if ca, L2 if cg
    t1 = clock64();
    v1 = arr[0];   // cold load
    t2 = clock64();
    v2 = arr[1];   // same cache line, should hit L1
    t3 = clock64();

    result[0] = t2 - t1;  // cold miss latency
    result[1] = t3 - t2;  // cache hit latency

    // prevent dead-code elimination
    if (v1 == 0 && v2 == 0) result[0] = 0;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clock_khz;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("Device: %s  |  SM clock: %d MHz\n\n", prop.name, clock_khz / 1000);

    // Allocate a small array — we only need two elements
    // but make it large enough to avoid any caching side effects
    uint32_t* d_arr;
    uint64_t* d_result;
    cudaMalloc(&d_arr,    64 * sizeof(uint32_t));
    cudaMalloc(&d_result, 2  * sizeof(uint64_t));
    cudaMemset(&d_arr, 1, 64 * sizeof(uint32_t));

    // Warmup
    measure<<<1,1>>>(d_arr, d_result);
    cudaDeviceSynchronize();

    // Measure: average over many runs
    uint64_t total_cold = 0, total_hit = 0;
    uint64_t h_result[2];
    int runs = 1000;
    for (int r = 0; r < runs; r++) {
        // Flush d_arr from cache by overwriting it between runs
        cudaMemset(d_arr, 1, 64 * sizeof(uint32_t));
        cudaDeviceSynchronize();
        measure<<<1,1>>>(d_arr, d_result);
        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        total_cold += h_result[0];
        total_hit  += h_result[1];
    }

    double cold_cycles = (double)(total_cold / runs);
    double hit_cycles  = (double)(total_hit  / runs);
    double cold_ns     = cold_cycles / (clock_khz / 1e6);
    double hit_ns      = hit_cycles  / (clock_khz / 1e6);

    printf("Cold miss:  %.1f cycles  |  %.1f ns\n", cold_cycles, cold_ns);
    printf("Cache hit:  %.1f cycles  |  %.1f ns\n", hit_cycles,  hit_ns);

    cudaFree(d_arr);
    cudaFree(d_result);
    return 0;
}
