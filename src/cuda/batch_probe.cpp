// batch_probe — drives the production sentence pipeline from C++ so that the
// batch-size instrumentation can be exercised without a JVM.
//
// The instrumentation lives in the shared library and records whatever the
// real caller submits. This driver verifies that it records correctly and
// characterizes the pipeline at known batch sizes; it does NOT substitute for
// running the Java application, which is the only thing that can reveal the
// batch sizes production actually uses.
//
//   LEMMATIZER_BATCH_LOG=probe.csv ./batch_probe --corpus fiction_pp.txt \
//       --sentences 12 --batches 4000000,400000,40000
//
// --sentences is words per synthetic sentence; --batches is a comma-separated
// list of token counts, each submitted as one call.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/reduction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "lemmatizer.h"

namespace {

std::vector<std::string> read_tokens(std::string const& path, std::size_t want) {
    std::vector<std::string> v;
    v.reserve(std::min<std::size_t>(want, 4u << 20));
    std::ifstream in(path);
    if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string line;
    while (v.size() < want && std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) v.push_back(line);
    }
    if (v.empty()) { std::cerr << "no tokens in " << path << "\n"; std::exit(1); }
    return v;
}

// Assemble tokens into sentences of `per` words, as the Java caller delivers
// them: one string per sentence, words separated by single spaces.
std::vector<std::string> make_sentences(std::vector<std::string> const& toks,
                                        std::size_t tokens, std::size_t per) {
    std::vector<std::string> out;
    out.reserve(tokens / per + 1);
    std::string s;
    std::size_t in_s = 0;
    for (std::size_t i = 0; i < tokens; ++i) {
        if (in_s) s += ' ';
        s += toks[i % toks.size()];
        if (++in_s == per) { out.push_back(std::move(s)); s.clear(); in_s = 0; }
    }
    if (!s.empty()) out.push_back(std::move(s));
    return out;
}

std::unique_ptr<cudf::column> to_column(std::vector<std::string> const& rows) {
    std::vector<char> chars;
    std::vector<int32_t> offs;
    offs.reserve(rows.size() + 1);
    std::size_t total = 0;
    for (auto const& r : rows) total += r.size();
    chars.reserve(total);
    offs.push_back(0);
    for (auto const& r : rows) {
        chars.insert(chars.end(), r.begin(), r.end());
        offs.push_back(static_cast<int32_t>(chars.size()));
    }

    rmm::device_uvector<char> d_chars(chars.size(), rmm::cuda_stream_default);
    cudaMemcpy(d_chars.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);

    auto offsets_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offs.size()), cudf::mask_state::UNALLOCATED);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(), offs.data(),
               offs.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    return cudf::make_strings_column(
        static_cast<cudf::size_type>(rows.size()), std::move(offsets_col),
        rmm::device_buffer{d_chars.data(), chars.size(), rmm::cuda_stream_default},
        0, rmm::device_buffer{});
}

std::vector<std::size_t> parse_list(std::string const& s) {
    std::vector<std::size_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) v.push_back(std::stoull(item));
    return v;
}

}  // namespace

// Pathological inputs for the splitting semantics: the fused path claims to
// reproduce split_record(" ") + join_list_elements(" ") exactly, which means
// empty fields from consecutive, leading and trailing separators must survive.
int self_test() {
    std::vector<std::string> rows = {
        "",
        " ",
        "  ",
        "   ",
        "слово",
        " слово",
        "слово ",
        " слово ",
        "слово  слово",
        "слово слово слово",
        "не-словникове-слово слово",
        "  двома  пробілами  скрізь  ",
        "а",
        "а б в г д е є ж з и і ї й к л м н о п р с т у ф х ц ч ш щ ь ю я",
    };
    auto col = to_column(rows);
    auto staged = lemmatize_sentences(col->view());
    auto fused  = lemmatize_sentences_fused(col->view());

    // Pull both back to the host and compare row by row, so a mismatch names
    // the input that caused it rather than just failing.
    auto host = [](cudf::column_view v) {
        auto scv = cudf::strings_column_view(v);
        std::vector<int32_t> offs(scv.size() + 1);
        cudaMemcpy(offs.data(), scv.offsets().data<int32_t>(),
                   offs.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        std::vector<char> chars(offs.back());
        if (!chars.empty())
            cudaMemcpy(chars.data(), scv.chars_begin(cudf::get_default_stream()),
                       chars.size(), cudaMemcpyDeviceToHost);
        std::vector<std::string> out;
        for (std::size_t i = 0; i + 1 < offs.size(); ++i)
            out.emplace_back(chars.data() + offs[i], offs[i + 1] - offs[i]);
        return out;
    };
    auto a = host(staged->view());
    auto b = host(fused->view());

    int bad = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        bool const ok = (i < a.size() && i < b.size() && a[i] == b[i]);
        if (!ok) ++bad;
        std::cout << (ok ? "  ok   " : "  FAIL ") << "[" << rows[i] << "]\n";
        if (!ok) {
            std::cout << "        staged [" << (i < a.size() ? a[i] : "<missing>") << "]\n";
            std::cout << "        fused  [" << (i < b.size() ? b[i] : "<missing>") << "]\n";
        }
    }
    std::cout << (bad ? "SELF-TEST FAILED" : "self-test passed")
              << ": " << rows.size() - bad << "/" << rows.size() << " rows match\n";
    return bad;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--selftest") return self_test();

    std::string corpus = "fiction_pp.txt";
    std::size_t per = 12;
    std::vector<std::size_t> batches{1000000};
    int repeats = 1;

    for (int i = 1; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--corpus")    corpus = argv[++i];
        else if (a == "--sentences") per = std::stoull(argv[++i]);
        else if (a == "--batches")   batches = parse_list(argv[++i]);
        else if (a == "--repeats")   repeats = std::stoi(argv[++i]);
    }
    if (per == 0) per = 1;

    if (!std::getenv("LEMMATIZER_BATCH_LOG"))
        std::cout << "note: LEMMATIZER_BATCH_LOG is unset; the run will proceed "
                     "but nothing will be recorded.\n";

    std::size_t const need = *std::max_element(batches.begin(), batches.end());
    std::cout << "loading up to " << need << " tokens from " << corpus << " ...\n";
    auto toks = read_tokens(corpus, need);
    std::cout << "loaded " << toks.size() << " distinct token slots\n";

    auto timed = [](auto&& fn) {
        cudaDeviceSynchronize();
        auto const t0 = std::chrono::high_resolution_clock::now();
        auto out = fn();
        cudaDeviceSynchronize();
        return std::make_pair(std::move(out),
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
    };

    std::printf("\n%12s %12s %12s %12s %9s %s\n",
                "tokens", "sentences", "staged ms", "fused ms", "speedup", "identical");
    for (int r = 0; r < repeats; ++r) {
        for (std::size_t n : batches) {
            auto rows = make_sentences(toks, n, per);
            auto col = to_column(rows);

            auto [a, staged_ms] = timed([&] { return lemmatize_sentences(col->view()); });
            auto [b, fused_ms]  = timed([&] { return lemmatize_sentences_fused(col->view()); });

            // Bit-for-bit comparison of the two result columns, on device.
            bool same = (a->size() == b->size());
            if (same) {
                auto eq = cudf::binary_operation(a->view(), b->view(),
                    cudf::binary_operator::NULL_EQUALS,
                    cudf::data_type{cudf::type_id::BOOL8});
                auto all = cudf::reduce(eq->view(),
                    *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                    cudf::data_type{cudf::type_id::BOOL8});
                same = static_cast<cudf::numeric_scalar<bool>*>(all.get())->value();
            }
            std::printf("%12zu %12zu %12.2f %12.2f %8.2fx %s\n",
                        n, rows.size(), staged_ms, fused_ms,
                        staged_ms / fused_ms, same ? "yes" : "NO  <-- MISMATCH");
        }
    }
    std::cout << "done\n";
    return 0;
}
