//
// Created by Illya on 07.04.2025.
//

#ifndef TRIE_H
#define TRIE_H
#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <dawgdic/dawg.h>
#include <dawgdic/dictionary.h>

#include "structs.h"

bool load_flat_dict(const std::string& key_path, const std::string& val_path,
                    std::vector<char>& dict_keys, std::vector<char>& dict_vals, int& dict_size);
bool load_csv_dict(const std::string& path,
                   std::vector<char>& dict_keys,
                   std::vector<char>& dict_vals,
                   int& dict_size,
                   int max_entries = 7000000);
bool load_csv_dict_to_dawg(const std::string& path,
                           dawgdic::Dawg& dawg,
                           dawgdic::Dictionary& dict,
                           std::vector<char>& dict_vals);
bool load_csv_dict_to_dawg(const std::string& path,
                            dawgdic::Dictionary& dict,
                            std::vector<char>& dict_vals);

void save_dawg(const std::string& path, const dawgdic::Dictionary& dawg);
void load_dawg(const std::string& path, dawgdic::Dictionary& dawg);
void init_trie_data();
void lemmatize_batch(const char** input_words, int num_words, char** output_words);


template <typename T>
void save_bin_vector(const std::string& path, const std::vector<T>& vec) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return;
    }
    file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
}

template <typename T>
void load_bin_vector(const std::string& path, std::vector<T>& vec) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << path << "\n";
        return;
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    vec.resize(size / sizeof(T));
    file.read(reinterpret_cast<char*>(vec.data()), size);
}


#endif //TRIE_H
