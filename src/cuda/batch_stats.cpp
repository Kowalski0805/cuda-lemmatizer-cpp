#include "batch_stats.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace batch_stats {
namespace {

std::mutex g_mu;
std::vector<Call> g_calls;
std::FILE* g_csv = nullptr;
std::string g_path;
bool g_init = false;
bool g_on = false;

// Percentile over a sorted vector, linear interpolation.
double pct(std::vector<double> const& v, double p) {
    if (v.empty()) return 0.0;
    double const x = p * (static_cast<double>(v.size()) - 1) / 100.0;
    auto const lo = static_cast<std::size_t>(x);
    auto const hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (x - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

// The token-weighted median: the batch size b such that half of all tokens are
// processed in batches of at most b. This, not the per-call median, is what
// decides the regime -- a handful of large batches can carry most of the work
// while most calls are small.
double token_weighted_pct(std::vector<Call> const& calls, double p) {
    std::vector<Call> s(calls);
    std::sort(s.begin(), s.end(),
              [](Call const& a, Call const& b) { return a.tokens < b.tokens; });
    std::int64_t const total =
        std::accumulate(s.begin(), s.end(), std::int64_t{0},
                        [](std::int64_t a, Call const& c) { return a + c.tokens; });
    if (total == 0) return 0.0;
    std::int64_t const target = static_cast<std::int64_t>(p / 100.0 * static_cast<double>(total));
    std::int64_t seen = 0;
    for (auto const& c : s) {
        seen += c.tokens;
        if (seen >= target) return static_cast<double>(c.tokens);
    }
    return static_cast<double>(s.back().tokens);
}

void write_summary() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_csv) { std::fclose(g_csv); g_csv = nullptr; }
    if (g_calls.empty()) return;

    std::FILE* f = std::fopen((g_path + ".summary").c_str(), "w");
    if (!f) return;

    // The first call carries one-time initialization -- the 260 MB trie is
    // loaded and the first device allocations are made inside it -- which is
    // per-process, not per-batch. It is reported separately and excluded from
    // every mean below, exactly as preparation timing is handled in Section 3.6.
    Call const first = g_calls.front();
    std::vector<Call> steady(g_calls.begin() + (g_calls.size() > 1 ? 1 : 0), g_calls.end());
    std::vector<Call>& calls = steady;

    std::int64_t const n = static_cast<std::int64_t>(calls.size());
    std::int64_t const tok = std::accumulate(
        calls.begin(), calls.end(), std::int64_t{0},
        [](std::int64_t a, Call const& c) { return a + c.tokens; });
    std::int64_t const sent = std::accumulate(
        calls.begin(), calls.end(), std::int64_t{0},
        [](std::int64_t a, Call const& c) { return a + c.sentences; });
    double const ms = std::accumulate(
        calls.begin(), calls.end(), 0.0,
        [](double a, Call const& c) { return a + c.total_ms; });

    std::vector<double> t;
    t.reserve(calls.size());
    for (auto const& c : calls) t.push_back(static_cast<double>(c.tokens));
    std::sort(t.begin(), t.end());

    std::fprintf(f, "lemmatization batch-size summary\n");
    std::fprintf(f, "================================\n\n");
    std::fprintf(f, "first call (excluded below: carries one-time init)\n");
    std::fprintf(f, "  tokens               %12lld\n", (long long)first.tokens);
    std::fprintf(f, "  total, ms            %12.1f   of which lookup %.1f\n\n",
                 first.total_ms, first.lookup_ms);
    std::fprintf(f, "calls                  %12lld\n", (long long)n);
    std::fprintf(f, "sentences              %12lld\n", (long long)sent);
    std::fprintf(f, "tokens                 %12lld\n", (long long)tok);
    std::fprintf(f, "wall time, ms          %12.1f\n", ms);
    if (ms > 0) std::fprintf(f, "throughput, Mtok/s     %12.1f\n", tok / ms / 1000.0);

    std::fprintf(f, "\ntokens per call\n");
    for (double p : {0.0, 25.0, 50.0, 75.0, 90.0, 99.0, 100.0})
        std::fprintf(f, "  p%-5.4g              %12.0f\n", p, pct(t, p));
    std::fprintf(f, "  mean                 %12.0f\n",
                 static_cast<double>(tok) / static_cast<double>(n));

    std::fprintf(f, "\ntoken-weighted batch size (half the tokens arrive in\n"
                    "batches at most this large -- the statistic that selects\n"
                    "the regime, since a few large calls can carry most of the work)\n");
    for (double p : {50.0, 90.0})
        std::fprintf(f, "  p%-5.4g              %12.0f\n", p,
                     token_weighted_pct(calls, p));

    // Where the work actually lands relative to the single-pass break-even.
    std::int64_t below = 0, above = 0, calls_below = 0;
    for (auto const& c : calls) {
        if (c.tokens < kOneShotBreakEven) { below += c.tokens; ++calls_below; }
        else                              { above += c.tokens; }
    }
    std::fprintf(f, "\nagainst the Section 4.7 break-even of %lld tokens\n",
                 (long long)kOneShotBreakEven);
    std::fprintf(f, "  calls below            %12lld  (%.1f %%)\n",
                 (long long)calls_below, 100.0 * (double)calls_below / (double)n);
    std::fprintf(f, "  tokens in those calls  %12lld  (%.1f %%)\n",
                 (long long)below, tok ? 100.0 * (double)below / (double)tok : 0.0);
    std::fprintf(f, "  tokens at or above     %12lld  (%.1f %%)\n",
                 (long long)above, tok ? 100.0 * (double)above / (double)tok : 0.0);

    double const frac_below = tok ? (double)below / (double)tok : 0.0;
    std::fprintf(f, "\nrecommended configuration: ");
    if (frac_below > 0.8)
        std::fprintf(f, "A3, prefix sort without compaction.\n"
                        "  Most tokens arrive in batches inside the single-pass window, where\n"
                        "  ordering repays itself with no reuse and no extra device memory.\n");
    else if (frac_below < 0.2)
        std::fprintf(f, "A6b, coarse sort with compaction, only if a batch is\n"
                        "  traversed more than once. Batches are past the single-pass window,\n"
                        "  so on one pass no ordering pays and the baseline should be kept.\n");
    else
        std::fprintf(f, "split by batch size at run time.\n"
                        "  The distribution straddles the break-even; neither configuration is\n"
                        "  right for all calls, so select per call on the token count.\n");

    std::fprintf(f, "\nmean stage split, ms per call\n");
    auto avg = [&](double Call::* m) {
        return std::accumulate(calls.begin(), calls.end(), 0.0,
                               [&](double a, Call const& c) { return a + c.*m; }) /
               static_cast<double>(n);
    };
    std::fprintf(f, "  split                %12.3f\n", avg(&Call::split_ms));
    std::fprintf(f, "  explode              %12.3f\n", avg(&Call::explode_ms));
    std::fprintf(f, "  lookup               %12.3f\n", avg(&Call::lookup_ms));
    std::fprintf(f, "  groupby              %12.3f\n", avg(&Call::group_ms));
    std::fprintf(f, "  join                 %12.3f\n", avg(&Call::join_ms));
    std::fprintf(f, "  total                %12.3f\n", avg(&Call::total_ms));
    // What any faster traversal can be worth end to end. The lookup kernel is
    // the only stage ordering touches; Amdahl bounds the rest.
    double const total = avg(&Call::total_ms);
    double const frac = total > 0 ? avg(&Call::lookup_ms) / total : 0.0;
    std::fprintf(f, "\nlookup is %.1f %% of pipeline time, so the end-to-end gain\n"
                    "from a faster traversal is bounded by Amdahl's law:\n", 100.0 * frac);
    for (double sp : {1.25, 1.94, 3.33, 1e9}) {
        double const e2e = 1.0 / ((1.0 - frac) + frac / sp);
        if (sp > 1e8) std::fprintf(f, "  traversal made free     %6.2fx end to end\n", e2e);
        else          std::fprintf(f, "  traversal %.2fx         %6.2fx end to end\n", sp, e2e);
    }
    std::fprintf(f, "The 1.25x and 3.33x rows are A3 and A6 of Table 4.1; the last is\n"
                    "the ceiling on any possible improvement to the traversal kernel.\n");

    std::fprintf(f, "\nStage timings are taken with the pipeline synchronized between\n"
                    "stages and are therefore pessimistic in total; batch sizes are not\n"
                    "affected. The lookup row is the kernel this study optimizes; the\n"
                    "others bound how much end-to-end gain any ordering can deliver.\n");
    std::fclose(f);
}

void init() {
    if (g_init) return;
    g_init = true;
    char const* p = std::getenv("LEMMATIZER_BATCH_LOG");
    if (!p || !*p) return;
    g_path = p;
    g_csv = std::fopen(g_path.c_str(), "w");
    if (!g_csv) return;
    std::fprintf(g_csv, "sentences,tokens,token_bytes,split_ms,explode_ms,"
                        "lookup_ms,group_ms,join_ms,total_ms\n");
    g_on = true;
    std::atexit(write_summary);
}

}  // namespace

bool enabled() {
    std::lock_guard<std::mutex> lk(g_mu);
    init();
    return g_on;
}

void record(Call const& c) {
    std::lock_guard<std::mutex> lk(g_mu);
    init();
    if (!g_on) return;
    g_calls.push_back(c);
    std::fprintf(g_csv, "%lld,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                 (long long)c.sentences, (long long)c.tokens, (long long)c.token_bytes,
                 c.split_ms, c.explode_ms, c.lookup_ms, c.group_ms, c.join_ms, c.total_ms);
    std::fflush(g_csv);
}

}  // namespace batch_stats
