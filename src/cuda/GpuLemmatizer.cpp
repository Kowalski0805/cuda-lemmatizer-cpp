#include <jni.h>
#include <vector>
#include <string>
#include <cstring>

#include "icu_lowercase.h"
#include "lemmatizer.h"  // includes lemmatize_batch()
#include "structs.h"

extern "C"
JNIEXPORT jobjectArray JNICALL Java_org_example_GpuLemmatizer_lemmatize
  (JNIEnv* env, jobject obj, jobjectArray jWords) {

    int num_words = env->GetArrayLength(jWords);

    // Prepare input: extract strings from jWords
    std::vector<std::string> input_words(num_words);
    const char** c_input = new const char*[num_words];

    for (int i = 0; i < num_words; ++i) {
        jstring jstr = (jstring)env->GetObjectArrayElement(jWords, i);
        const char* utf = env->GetStringUTFChars(jstr, nullptr);
        input_words[i] = std::string(utf);
        env->ReleaseStringUTFChars(jstr, utf);
        env->DeleteLocalRef(jstr);
        c_input[i] = input_words[i].c_str();
    }

    // Allocate output array
    char** c_output = new char*[num_words];
    for (int i = 0; i < num_words; ++i)
        c_output[i] = new char[MAX_WORD_LEN];

    // Call CUDA-backed lemmatizer
    lemmatize_batch(c_input, num_words, c_output);

    // Create Java string array
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(num_words, stringClass, nullptr);

    for (int i = 0; i < num_words; ++i) {
        jstring jlemma = env->NewStringUTF(c_output[i]);
        env->SetObjectArrayElement(result, i, jlemma);
        delete[] c_output[i];
    }

    delete[] c_output;
    delete[] c_input;

    return result;
}