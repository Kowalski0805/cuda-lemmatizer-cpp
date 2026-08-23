//
// Created by Illya on 19.04.2025.
//

#ifndef LEMMATIZER_H
#define LEMMATIZER_H
// lemmatizer.h
#pragma once
#include <cudf/column/column.hpp>

void init_trie_data();  // if you expose it too

// Sentence-level pipeline: split -> explode -> trie lookup -> regroup -> join.
// Shared by the JNI entry point and by the batch_probe driver.
std::unique_ptr<cudf::column> lemmatize_sentences(cudf::column_view const& sentences);

// Fused sentence-to-sentence path: one traversal kernel plus byte scans,
// replacing split_record / explode / groupby / join_list_elements entirely.
// Same splitting semantics as split(" ") + join(" "). See fused_lemmatize.cu.
std::unique_ptr<cudf::column> lemmatize_sentences_fused(cudf::column_view const& sentences);

#ifdef __cplusplus
extern "C" {
#endif

    std::unique_ptr<cudf::column> lemmatize_batch(cudf::column_view const& strs);

#ifdef __cplusplus
}
#endif

#endif //LEMMATIZER_H
