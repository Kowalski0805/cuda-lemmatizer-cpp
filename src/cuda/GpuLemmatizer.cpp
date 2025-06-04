#include <jni.h>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <cudf/column/column.hpp>

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

            // run the GPU kernel to compute the word counts
            std::unique_ptr<cudf::column> result = lemmatize_batch(*strs);

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
