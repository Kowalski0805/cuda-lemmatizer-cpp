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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/wrappers/durations.hpp>
#include <rmm/exec_policy.hpp>

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

// std::unique_ptr<cudf::column> lemmatize_batch(cudf::column_view const& strs) {
//     init_trie_data();
//
//     auto strings_count = strs.size();
//     if (strings_count == 0) {
//         return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
//     }
//
//     auto stream = rmm::cuda_stream_default;
//     rmm::device_uvector<thrust::pair<char const*, cudf::size_type>> d_output(strings_count, stream);
//
//     auto strs_device_view = cudf::column_device_view::create(strs, stream);
//     auto d_strs_view = *strs_device_view;
//
//     // 🔥 запуск ядра один раз
//     launch_lookup_kernel(d_strs_view, strings_count, d_states, d_transitions, d_lemmas, d_output.data());
//
//     // ✅ формування стовпця з готових результатів
//     return cudf::make_strings_column(
//         cudf::device_span<thrust::pair<char const*, cudf::size_type>>(d_output.data(), strings_count),
//         stream,
//         rmm::mr::get_current_device_resource()
//     );
// }

std::unique_ptr<cudf::column> lemmatize_batch(cudf::column_view const& strs) {
    init_trie_data();

    auto strings_count = strs.size();
    if (strings_count == 0) {
        return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }

    rmm::device_buffer null_mask = cudf::copy_bitmask(strs);

    auto stream = rmm::cuda_stream_default;
    rmm::device_uvector<const char*> d_ptrs(strings_count, stream);
    rmm::device_uvector<int32_t> d_lengths(strings_count, stream);

    auto strs_device_view = cudf::column_device_view::create(strs, stream);
    auto d_strs_view = *strs_device_view;
    auto policy = rmm::exec_policy(stream);

    thrust::for_each_n(
        d_ptrs.begin(),
        d_ptrs.size(),
        [d_strs_view, ptrs = d_ptrs.data(), lens = d_lengths.data()] __device__ (size_t idx) {
            auto const word = d_strs_view.element<cudf::string_view>(idx);
            int state = 0;
            bool fail = false;

            for (int i = 0; i < word.size_bytes(); ++i) {
                char ch = word.data()[i];
                const GpuState& s = d_states[state];
                bool found = false;

                for (int j = 0; j < s.num_transitions; ++j) {
                    const GpuTransition& t = d_transitions[s.transition_start_idx + j];
                    if (t.c == ch) {
                        state = t.next_state;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    fail = true;
                    break;
                }
            }

            if (!fail) {
                const GpuState& final_state = d_states[state];
                if (final_state.lemma_offset >= 0) {
                    for (int i = 0; i < MAX_WORD_LEN; ++i) {
                        if (char c = d_lemmas[final_state.lemma_offset + i]; c == '\0') {
                            ptrs[idx] = d_lemmas + final_state.lemma_offset;
                            lens[idx] = i;
                            return;
                        }
                    }
                }
            }

            // Fallback if lemma not found
            ptrs[idx] = word.data();
            lens[idx] = word.size_bytes();
        }
    );

    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        strings_count + 1,
        d_lengths.release(),
        rmm::device_buffer{0, stream}, // no null mask
        0
    );

    auto result_column = cudf::make_strings_column(
        strings_count,
        std::move(offsets_col),
        d_ptrs.release(),
        0,
        rmm::device_buffer{0, stream}
    );

    return result_column;
}
