//
// Created by Illya on 19.04.2025.
//

#include "lemmatizer.h"
#include "structs.h"
#include "trie.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/wrappers/durations.hpp>
#include <rmm/exec_policy.hpp>
#include "lemmatizer_kernel.cuh"
#include "lemmatizer_kernel.h"

struct GpuState;
struct GpuTransition;
static bool is_initialized = false;
static std::vector<GpuState> h_states;
static std::vector<GpuTransition> h_transitions;
static std::vector<char> h_lemmas;

static GpuState* d_states = nullptr;
static GpuTransition* d_transitions = nullptr;
static char* d_lemmas = nullptr;

void init_trie_data() {
    if (is_initialized) return;

    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);

    cudaMalloc(&d_states, h_states.size() * sizeof(GpuState));
    cudaMalloc(&d_transitions, h_transitions.size() * sizeof(GpuTransition));
    cudaMalloc(&d_lemmas, h_lemmas.size());

    cudaMemcpy(d_states, h_states.data(), h_states.size() * sizeof(GpuState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transitions, h_transitions.data(), h_transitions.size() * sizeof(GpuTransition), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lemmas, h_lemmas.data(), h_lemmas.size(), cudaMemcpyHostToDevice);

    is_initialized = true;
}

std::unique_ptr<cudf::column> lemmatize_batch(cudf::column_view const& strs) {
    init_trie_data();

    auto strings_count = strs.size();
    if (strings_count == 0) {
        return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }

    auto stream = rmm::cuda_stream_default;
    rmm::device_uvector<thrust::pair<char const*, cudf::size_type>> d_output(strings_count, stream);

    auto strs_device_view = cudf::column_device_view::create(strs, stream);
    auto d_strs_view = *strs_device_view;

    // 🕒 CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.value());

    // 🔥 запуск ядра один раз
    launch_lookup_kernel(d_strs_view, strings_count, d_states, d_transitions, d_lemmas, d_output.data());

    cudaEventRecord(stop, stream.value());
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ✅ формування стовпця з готових результатів
    return cudf::make_strings_column(
        cudf::device_span<thrust::pair<char const*, cudf::size_type>>(d_output.data(), strings_count),
        stream,
        rmm::mr::get_current_device_resource()
    );
}
