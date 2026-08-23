// Fused sentence-to-sentence lemmatization.
//
// The staged pipeline spends 21.5 % of its time in the trie traversal and the
// rest in dataframe string plumbing: split_record materializes a list column of
// words, explode flattens it, groupby regroups the lemmas, join_list_elements
// rebuilds the sentences. All four exist only to get tokens into and out of the
// traversal, and none of them is necessary: a token is a span of the input
// sentence, and a lemma is a span of the trie's lemma buffer, so nothing needs
// to be materialized as a cuDF column in between.
//
// This takes a column of sentence strings and returns a column of lemmatized
// sentence strings, with the same splitting semantics as
// split_record(" ") + join_list_elements(" ") -- including empty fields
// produced by consecutive separators, which are preserved verbatim.
//
//   K1  count fields per sentence                    (one thread per sentence)
//   scan -> per-sentence token offsets, N tokens
//   K2  write each token as (pointer, length)        (one thread per sentence)
//   K3  trie traversal, one thread per token         <- the production kernel
//   scan over (lemma length + 1) -> per-token output positions
//   K4  per-sentence output sizes                    (one thread per sentence)
//   scan -> output offsets column
//   K5  copy lemma bytes and separators              (one thread per token)
//
// Only K3 touches the trie. The four surrounding kernels are byte scans and
// copies, and the two large scans are bandwidth-bound.
#include <cuda_runtime.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/scan.h>

#include "structs.h"

namespace {

constexpr int kBlock = 256;

__device__ inline int field_count(char const* p, int n) {
    // split(" ") yields (number of separators + 1) fields, empty ones included.
    int k = 1;
    for (int i = 0; i < n; ++i) k += (p[i] == ' ');
    return k;
}

__global__ void count_fields_kernel(cudf::column_device_view in,
                                    cudf::size_type num_rows,
                                    cudf::size_type* counts) {
    auto const s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_rows) return;
    if (in.is_null(s)) { counts[s] = 0; return; }
    auto const row = in.element<cudf::string_view>(s);
    counts[s] = field_count(row.data(), row.size_bytes());
}

__global__ void emit_fields_kernel(cudf::column_device_view in,
                                   cudf::size_type num_rows,
                                   cudf::size_type const* tok_off,
                                   char const** tok_ptr,
                                   cudf::size_type* tok_len,
                                   cudf::size_type* tok_sent) {
    auto const s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_rows) return;
    if (in.is_null(s)) return;
    auto const row = in.element<cudf::string_view>(s);
    char const* p = row.data();
    int const n = row.size_bytes();

    auto slot = tok_off[s];
    int start = 0;
    for (int i = 0; i <= n; ++i) {
        if (i == n || p[i] == ' ') {
            tok_ptr[slot]  = p + start;
            tok_len[slot]  = i - start;
            tok_sent[slot] = s;
            ++slot;
            start = i + 1;
        }
    }
}

// The production traversal, reading a (pointer, length) span instead of a cuDF
// element. Byte per level, linear scan of each state's transitions, fall back
// to the surface form when the walk fails or the final state carries no lemma.
__global__ void lookup_spans_kernel(char const** tok_ptr,
                                    cudf::size_type const* tok_len,
                                    cudf::size_type num_tokens,
                                    GpuState const* states,
                                    GpuTransition const* transitions,
                                    char const* lemmas,
                                    char const** out_ptr,
                                    cudf::size_type* out_len) {
    auto const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;

    char const* word = tok_ptr[t];
    int const n = tok_len[t];
    int state = 0;

    for (int i = 0; i < n; ++i) {
        char const ch = word[i];
        GpuState const& st = states[state];
        bool found = false;
        for (int j = 0; j < st.num_transitions; ++j) {
            GpuTransition const& tr = transitions[st.transition_start_idx + j];
            if (tr.c == ch) { state = tr.next_state; found = true; break; }
        }
        if (!found) { out_ptr[t] = word; out_len[t] = n; return; }
    }

    GpuState const& fin = states[state];
    if (n > 0 && fin.lemma_offset >= 0) {
        int len = 0;
        while (lemmas[fin.lemma_offset + len] != '\0' && len < MAX_WORD_LEN) ++len;
        out_ptr[t] = lemmas + fin.lemma_offset;
        out_len[t] = len;
    } else {
        out_ptr[t] = word;
        out_len[t] = n;
    }
}

// w[t] = output length of token t plus one separator slot; the extra element
// at num_tokens exists so the exclusive scan below produces a total.
__global__ void weights_kernel(cudf::size_type num_tokens,
                               cudf::size_type const* out_len,
                               cudf::size_type* w) {
    auto const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > num_tokens) return;
    w[t] = (t < num_tokens) ? out_len[t] + 1 : 0;
}

__global__ void row_sizes_kernel(cudf::size_type num_rows,
                                 cudf::size_type const* tok_off,
                                 cudf::size_type const* cum,
                                 cudf::size_type* sizes) {
    auto const s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_rows) return;
    auto const lo = tok_off[s], hi = tok_off[s + 1];
    // cum is the exclusive scan of (lemma length + 1), so the span between two
    // token boundaries counts one separator too many: the trailing one.
    sizes[s] = (hi > lo) ? (cum[hi] - cum[lo] - 1) : 0;
}

__global__ void write_kernel(cudf::size_type num_tokens,
                             cudf::size_type const* tok_off,
                             cudf::size_type const* tok_sent,
                             cudf::size_type const* cum,
                             char const* const* out_ptr,
                             cudf::size_type const* out_len,
                             cudf::size_type const* row_off,
                             char* dst) {
    auto const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    auto const s = tok_sent[t];
    char* w = dst + row_off[s] + (cum[t] - cum[tok_off[s]]);
    auto const len = out_len[t];
    char const* src = out_ptr[t];
    for (cudf::size_type i = 0; i < len; ++i) w[i] = src[i];
    if (t + 1 < tok_off[s + 1]) w[len] = ' ';   // separator, except after the last field
}

}  // namespace

void get_trie_device_ptrs(GpuState const** states,
                          GpuTransition const** transitions,
                          char const** lemmas);

static std::unique_ptr<cudf::column> fused_lemmatize_sentences(
    cudf::column_view const& sentences,
    GpuState const* d_states,
    GpuTransition const* d_transitions,
    char const* d_lemmas) {
    auto const num_rows = sentences.size();
    if (num_rows == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto const grid_rows = (num_rows + kBlock - 1) / kBlock;

    auto d_in_view = cudf::column_device_view::create(sentences, stream);

    // --- token boundaries -------------------------------------------------
    rmm::device_uvector<cudf::size_type> tok_off(num_rows + 1, stream, mr);
    count_fields_kernel<<<grid_rows, kBlock, 0, stream.value()>>>(
        *d_in_view, num_rows, tok_off.data());
    thrust::exclusive_scan(rmm::exec_policy(stream), tok_off.begin(),
                           tok_off.end(), tok_off.begin());

    cudf::size_type num_tokens = 0;
    cudaMemcpyAsync(&num_tokens, tok_off.data() + num_rows, sizeof(cudf::size_type),
                    cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();
    if (num_tokens == 0) {
        return cudf::make_strings_column(
            num_rows,
            cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                      num_rows + 1, cudf::mask_state::UNALLOCATED,
                                      stream, mr),
            rmm::device_buffer{}, sentences.null_count(),
            cudf::copy_bitmask(sentences, stream, mr));
    }

    rmm::device_uvector<char const*>      tok_ptr (num_tokens, stream, mr);
    rmm::device_uvector<cudf::size_type>  tok_len (num_tokens, stream, mr);
    rmm::device_uvector<cudf::size_type>  tok_sent(num_tokens, stream, mr);
    emit_fields_kernel<<<grid_rows, kBlock, 0, stream.value()>>>(
        *d_in_view, num_rows, tok_off.data(), tok_ptr.data(), tok_len.data(),
        tok_sent.data());

    // --- traversal --------------------------------------------------------
    rmm::device_uvector<char const*>     out_ptr(num_tokens, stream, mr);
    rmm::device_uvector<cudf::size_type> out_len(num_tokens, stream, mr);
    auto const grid_tok = (num_tokens + kBlock - 1) / kBlock;
    lookup_spans_kernel<<<grid_tok, kBlock, 0, stream.value()>>>(
        tok_ptr.data(), tok_len.data(), num_tokens, d_states, d_transitions,
        d_lemmas, out_ptr.data(), out_len.data());

    // --- output layout ----------------------------------------------------
    rmm::device_uvector<cudf::size_type> w  (num_tokens + 1, stream, mr);
    rmm::device_uvector<cudf::size_type> cum(num_tokens + 1, stream, mr);
    weights_kernel<<<(num_tokens + kBlock) / kBlock, kBlock, 0, stream.value()>>>(
        num_tokens, out_len.data(), w.data());
    // Scanning num_tokens+1 elements makes cum[num_tokens] the running total,
    // which row_sizes_kernel and write_kernel both index.
    thrust::exclusive_scan(rmm::exec_policy(stream), w.begin(), w.end(), cum.begin());

    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows + 1,
        cudf::mask_state::UNALLOCATED, stream, mr);
    auto* d_row_off = offsets_col->mutable_view().data<cudf::size_type>();
    row_sizes_kernel<<<grid_rows, kBlock, 0, stream.value()>>>(
        num_rows, tok_off.data(), cum.data(), d_row_off);
    thrust::exclusive_scan(rmm::exec_policy(stream), d_row_off,
                           d_row_off + num_rows + 1, d_row_off);

    // Output length is a cudf::size_type, so a single call is limited to 2 GiB
    // of output characters -- the same ceiling the classic strings layout
    // imposes on the staged path, reached here at roughly 170 M tokens.
    cudf::size_type total_bytes = 0;
    cudaMemcpyAsync(&total_bytes, d_row_off + num_rows, sizeof(cudf::size_type),
                    cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();

    rmm::device_buffer chars(total_bytes, stream, mr);
    write_kernel<<<grid_tok, kBlock, 0, stream.value()>>>(
        num_tokens, tok_off.data(), tok_sent.data(), cum.data(), out_ptr.data(),
        out_len.data(), d_row_off, static_cast<char*>(chars.data()));

    return cudf::make_strings_column(
        num_rows, std::move(offsets_col), std::move(chars),
        sentences.null_count(), cudf::copy_bitmask(sentences, stream, mr));
}


std::unique_ptr<cudf::column> lemmatize_sentences_fused(cudf::column_view const& sentences) {
    GpuState const* states = nullptr;
    GpuTransition const* transitions = nullptr;
    char const* lemmas = nullptr;
    get_trie_device_ptrs(&states, &transitions, &lemmas);
    return fused_lemmatize_sentences(sentences, states, transitions, lemmas);
}
