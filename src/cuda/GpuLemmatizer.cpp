#include <jni.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cudf/groupby.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime.h>

#include "lemmatizer.h"  // includes lemmatize_batch()
#include "batch_stats.h"

namespace {

    constexpr char const* RUNTIME_ERROR_CLASS = "java/lang/RuntimeException";

    /**
     * @brief Throw a Java exception
     *
     * @param env The Java environment
     * @param class_name The fully qualified Java class name of the exception
     * @param msg The message string to associate with the exception
     */
    void throw_java_exception(JNIEnv* env, char const* class_name, char const* msg) {
        jclass ex_class = env->FindClass(class_name);
        if (ex_class != NULL) {
            env->ThrowNew(ex_class, msg);
        }
    }

    using clk = std::chrono::high_resolution_clock;

    // Stage boundary. Without instrumentation this is a no-op and the stages
    // stay pipelined; with it the device is synchronized so that each stage's
    // elapsed time is its own rather than its successor's queueing time.
    struct Stopwatch {
        bool const on;
        clk::time_point mark;
        explicit Stopwatch(bool enabled) : on(enabled), mark(clk::now()) {}
        double lap() {
            if (!on) return 0.0;
            cudaDeviceSynchronize();
            auto const now = clk::now();
            double const ms = std::chrono::duration<double, std::milli>(now - mark).count();
            mark = now;
            return ms;
        }
    };

}  // anonymous namespace


// The sentence-level pipeline, extracted from the JNI entry point so that it
// can be driven from C++ as well as from Java. Takes a column of sentence
// strings, returns a column of the same length holding lemmatized sentences.
std::unique_ptr<cudf::column> lemmatize_sentences(cudf::column_view const& sentences) {
    bool const instrument = batch_stats::enabled();
    batch_stats::Call rec;
    Stopwatch sw(instrument);
    auto const t0 = clk::now();

    rec.sentences = sentences.size();

    // 1. Split sentence strings into a list of words per sentence.
    auto lists_col = cudf::strings::split_record(sentences, cudf::string_scalar(" "));
    rec.split_ms = sw.lap();

    auto offsets_view = cudf::lists_column_view(lists_col->view()).offsets();
    auto sliced_offsets = cudf::slice(offsets_view, {0, lists_col->size()});

    // 2. Explode to flat (sentence id, word) pairs.
    cudf::table_view input_table({lists_col->view(), sliced_offsets.front()});
    auto exploded = cudf::explode_position(input_table, 0);

    // 3. Exploded table: column 1 = word, column 2 = sentence id.
    auto sentence_ids = exploded->get_column(2);
    auto words        = exploded->get_column(1);
    rec.explode_ms = sw.lap();

    rec.tokens = words.size();
    if (instrument) {
        // Chars in the token column: the last offset. This is the input volume
        // the traversal kernel actually reads.
        auto const scv = cudf::strings_column_view(words.view());
        rec.token_bytes = scv.chars_size(cudf::get_default_stream());
    }

    // 4. Trie lookup.
    std::unique_ptr<cudf::column> lemmas = lemmatize_batch(words.view());
    rec.lookup_ms = sw.lap();

    // 5. Regroup lemmas by sentence.
    std::vector<cudf::groupby::aggregation_request> requests;
    cudf::groupby::aggregation_request req;
    req.values = lemmas->view();
    req.aggregations.push_back(cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
    requests.push_back(std::move(req));

    auto gather_map = cudf::groupby::groupby(
        cudf::table_view({sentence_ids}),
        cudf::null_policy::EXCLUDE
    ).aggregate(cudf::host_span<cudf::groupby::aggregation_request const>(requests));
    rec.group_ms = sw.lap();

    // 6. Join back into sentence strings.
    auto regrouped = std::move(gather_map.second[0].results.front());
    auto result = cudf::strings::join_list_elements(
        regrouped->view(), cudf::string_scalar(" "));
    rec.join_ms = sw.lap();

    if (instrument) {
        rec.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        batch_stats::record(rec);
    }
    return result;
}


extern "C" {
    JNIEXPORT jlong JNICALL
    Java_org_example_GpuLemmatizer_lemmatize(JNIEnv* env, jclass, jlong jWords) {
        // Use a try block to translate C++ exceptions into Java exceptions to avoid
        // crashing the JVM if a C++ exception occurs.
        try {
            // turn the address into a column_view pointer
            auto strs = reinterpret_cast<cudf::column_view const*>(jWords);
            // The fused path produces bit-identical output and is roughly four
            // times faster end to end, but it is opt-in: set LEMMATIZER_FUSED=1
            // to select it, so the change is reversible without a rebuild.
            static bool const fused = [] {
                char const* v = std::getenv("LEMMATIZER_FUSED");
                return v && *v && *v != '0';
            }();
            auto result = fused ? lemmatize_sentences_fused(*strs)
                                : lemmatize_sentences(*strs);
            // take ownership of the column and return the column address to Java
            return reinterpret_cast<jlong>(result.release());
        } catch (std::bad_alloc const& e) {
            auto msg = std::string("Unable to allocate native memory: ") +
                (e.what() == nullptr ? "" : e.what());
            throw_java_exception(env, RUNTIME_ERROR_CLASS, msg.c_str());
        } catch (std::exception const& e) {
            auto msg = e.what() == nullptr ? "" : e.what();
            throw_java_exception(env, RUNTIME_ERROR_CLASS, msg);
        }
        return 0;
    }
}
