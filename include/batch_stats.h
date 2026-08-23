// Batch-size instrumentation for the JNI lemmatization path.
//
// Chapter 4 finds that which ordering strategy pays depends on one deployment
// parameter: the number of tokens in a batch. Below roughly 8.2 M tokens a
// prefix sort repays itself on a single pass; above it, ordering must be
// amortized over repeated traversal. The production pipeline's actual batch
// sizes were never measured, so neither regime could be selected on evidence.
// This records them.
//
// Disabled unless LEMMATIZER_BATCH_LOG names a writable path, in which case a
// CSV of per-call records is written there and a summary to <path>.summary at
// process exit. When enabled the pipeline is synchronized between stages, so
// the per-stage timings are meaningful but the total is pessimistic; the batch
// sizes, which are the point, are unaffected.
#pragma once

#include <cstdint>

namespace batch_stats {

bool enabled();

struct Call {
    std::int64_t sentences   = 0;
    std::int64_t tokens      = 0;
    std::int64_t token_bytes = 0;
    double split_ms   = 0.0;
    double explode_ms = 0.0;
    double lookup_ms  = 0.0;
    double group_ms   = 0.0;
    double join_ms    = 0.0;
    double total_ms   = 0.0;
};

void record(Call const& c);

// Single-pass break-even for the prefix sort, Section 4.7. Batches below this
// size are in the regime where ordering pays with no reuse at all.
constexpr std::int64_t kOneShotBreakEven = 8'200'000;

}  // namespace batch_stats
