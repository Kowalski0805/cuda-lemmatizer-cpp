#include <jni.h>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <cudf/groupby.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/split/split.hpp>

#include "icu_lowercase.h"
#include "lemmatizer.h"  // includes lemmatize_batch()
#include "structs.h"

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

}  // anonymous namespace


extern "C" {
    JNIEXPORT jlong JNICALL
    Java_org_example_GpuLemmatizer_lemmatize(JNIEnv* env, jclass, jlong jWords) {
        // Use a try block to translate C++ exceptions into Java exceptions to avoid
        // crashing the JVM if a C++ exception occurs.
        try {
            // turn the addresses into column_view pointers
            auto strs = reinterpret_cast<cudf::column_view const*>(jWords);

            auto start = std::chrono::high_resolution_clock::now();
            // TMP: full text split
            // 1. Split sentence strings into list of words
            auto lists_col = cudf::strings::split_record(*strs, cudf::string_scalar(" "));

            auto offsets_view = cudf::lists_column_view(lists_col->view()).offsets();
            auto sliced_offsets = cudf::slice(offsets_view, {0, lists_col->size()});

            std::cout << "Number of lists: " << lists_col->size() << std::endl;
            std::cout << "Number of offsets: " << sliced_offsets.front().size() << std::endl;
            // 2. Wrap in table_view to call explode
            cudf::table_view input_table({lists_col->view(), sliced_offsets.front()});
            auto exploded = cudf::explode_position(input_table, 0);

            // 3. Exploded table: column 0 = sentence ID, column 1 = word
            auto sentence_ids = exploded->get_column(2);
            auto words        = exploded->get_column(1);

            std::cout << "Number of sentences: " << strs->size() << std::endl;
            std::cout << "Number of words: " << words.size() << std::endl;

            // run the GPU kernel to compute the word counts
            std::unique_ptr<cudf::column> lemmas = lemmatize_batch(words.view());

            std::cout << "Number of lemmas: " << lemmas->size() << std::endl;

            std::vector<cudf::groupby::aggregation_request> requests;
            cudf::groupby::aggregation_request req;
            req.values = lemmas->view();
            req.aggregations.push_back(cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
            requests.push_back(std::move(req));

            auto gather_map = cudf::groupby::groupby(
                cudf::table_view({sentence_ids}),
                cudf::null_policy::EXCLUDE
            ).aggregate(cudf::host_span<cudf::groupby::aggregation_request const>(requests));

            // 7. Join into sentence strings again
            auto regrouped = std::move(gather_map.second[0].results.front());
            auto result = cudf::strings::join_list_elements(
                regrouped->view(), cudf::string_scalar(" "));

            std::cout << "Number of grouped lemmas: " << result->size() << std::endl;


            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Lemmatization took: " << duration.count() << " ms" << std::endl;

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
