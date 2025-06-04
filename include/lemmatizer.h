//
// Created by Illya on 19.04.2025.
//

#ifndef LEMMATIZER_H
#define LEMMATIZER_H
// lemmatizer.h
#pragma once
#include <cudf/column/column.hpp>

void init_trie_data();  // if you expose it too

#ifdef __cplusplus
extern "C" {
#endif

    std::unique_ptr<cudf::column> lemmatize_batch(cudf::column_view const& strs);

#ifdef __cplusplus
}
#endif

#endif //LEMMATIZER_H
