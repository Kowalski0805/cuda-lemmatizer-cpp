//
// Created by Illya on 19.04.2025.
//

#ifndef LEMMATIZER_H
#define LEMMATIZER_H
// lemmatizer.h
#pragma once

void init_trie_data();  // if you expose it too

#ifdef __cplusplus
extern "C" {
#endif

    void lemmatize_batch(const char** input_words, int num_words, char** output_words);

#ifdef __cplusplus
}
#endif

#endif //LEMMATIZER_H
