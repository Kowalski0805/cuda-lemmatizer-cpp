//
// Created by Illya on 07.04.2025.
//

#include "trie.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <dawgdic/dawg-builder.h>
#include <dawgdic/dawg.h>
#include <dawgdic/dictionary-builder.h>
#include <vector>
#include <dawgdic/dictionary.h>
#include "../../include/icu_lowercase.h"
#include "structs.h"

bool load_flat_dict(const std::string& key_path, const std::string& val_path,
                    std::vector<char>& dict_keys, std::vector<char>& dict_vals, int& dict_size) {
    std::ifstream fkeys(key_path, std::ios::binary | std::ios::ate);
    std::ifstream fvals(val_path, std::ios::binary | std::ios::ate);

    if (!fkeys.is_open()) {
        std::cout << "Failed to open file " << key_path << std::endl;
        return false;
    }
    if (!fvals.is_open()) {
        std::cout << "Failed to open file " << val_path << std::endl;
        return false;
    }

    if (!fkeys || !fvals) {
        return false;
    }

    std::streamsize size_keys = fkeys.tellg();
    std::streamsize size_vals = fvals.tellg();

    if (size_keys != size_vals || size_keys % MAX_WORD_LEN != 0) {
        if (size_keys != size_vals) {
            std::cout << "Dictionary files have different sizes.\n";
        } else {
            std::cout << "Dictionary files are not aligned to word length.\n";
        }
        return false;
    }
    dict_size = size_keys / MAX_WORD_LEN;

    dict_keys.resize(size_keys);
    dict_vals.resize(size_vals);

    fkeys.seekg(0);
    fkeys.read(dict_keys.data(), size_keys);

    fvals.seekg(0);
    fvals.read(dict_vals.data(), size_vals);

    return true;
}

bool load_csv_dict(const std::string& path,
                   std::vector<char>& dict_keys,
                   std::vector<char>& dict_vals,
                   int& dict_size,
                   int max_entries) {

    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open CSV file: " << path << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    std::unordered_set<std::string> seen;  // Avoid duplicates

    std::vector<std::string> keys, vals;

    while (std::getline(file, line) && keys.size() < max_entries) {
        std::istringstream ss(line);
        std::string wordform, lemma;

        // Parse wordform and lemma
        std::getline(ss, wordform, ',');
        std::getline(ss, lemma, ',');

        if (wordform.empty() || lemma.empty()) continue;

        std::string key = wordform;
        std::string val = lemma;

        // Optionally deduplicate by wordform+lemma
        std::string pair_key = key + '\0' + val;
        if (!seen.insert(pair_key).second) continue;

        keys.push_back(key);
        vals.push_back(val);
    }

    dict_size = keys.size();
    dict_keys.resize(dict_size * MAX_WORD_LEN, 0);
    dict_vals.resize(dict_size * MAX_WORD_LEN, 0);

    for (int i = 0; i < dict_size; ++i) {
        strncpy(&dict_keys[i * MAX_WORD_LEN], keys[i].c_str(), MAX_WORD_LEN);
        strncpy(&dict_vals[i * MAX_WORD_LEN], vals[i].c_str(), MAX_WORD_LEN);
    }

    return true;
}

bool load_csv_dict_to_dawg(const std::string& path,
                           dawgdic::Dawg& dawg,
                           dawgdic::Dictionary& dict,
                           std::vector<char>& dict_vals) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open CSV file: " << path << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line);  // skip header

    std::unordered_set<std::string> seen;
    std::vector<std::pair<std::string, std::string>> entries;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string wordform, lemma;

        std::getline(ss, wordform, ',');
        std::getline(ss, lemma, ',');

        if (wordform.empty() || lemma.empty()) continue;
        if (wordform.find('-') != std::string::npos) continue;

        std::string key = lowercase_ukr(wordform);
        std::string val = lemma;

        std::string pair_key = key + '\0' + val;
        if (!seen.insert(pair_key).second) continue;

        entries.emplace_back(key, val);
    }

    // Sort entries by key
    std::sort(entries.begin(), entries.end());

    dawgdic::DawgBuilder builder;
    std::vector<std::string> vals;

    int i = 0;
    for (const auto& [key, val] : entries) {
        if (!builder.Insert(key.c_str(), key.size(), dict_vals.size())) {
            std::cerr << "Failed to insert into DawgBuilder: " << key << "\n";
            return false;
        }
        for (char c : val) dict_vals.push_back(c);
        dict_vals.push_back('\0');  // Null-terminate the value
        // i++;
        // vals.push_back(val);
    }

    // dict_size = vals.size();
    // dict_vals.resize(dict_size * MAX_WORD_LEN, 0);
    // for (int i = 0; i < dict_size; ++i) {
    //     strncpy(&dict_vals[i * MAX_WORD_LEN], vals[i].c_str(), MAX_WORD_LEN);
    // }

    builder.Finish(&dawg);
    dawgdic::DictionaryBuilder::Build(dawg, &dict);

    return true;
}

void build_gpu_trie_from_csv(
    const std::string& path,
    std::vector<GpuState>& states,
    std::vector<GpuTransition>& transitions,
    std::vector<char>& lemma_buffer)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(file, line); // skip header

    std::vector<TempTrieNode> temp_nodes;
    temp_nodes.emplace_back();  // root

    std::unordered_map<std::string, int> seen;
    int next_offset = 0;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string wordform, lemma;
        if (!std::getline(ss, wordform, ',') || !std::getline(ss, lemma, ',')) continue;
        if (wordform.find('-') != std::string::npos) continue;

        std::string key = lowercase_ukr(wordform);
        if (seen.count(key)) continue;
        seen[key] = next_offset;

        // append lemma to buffer
        int lemma_offset = next_offset;
        for (char c : lemma) lemma_buffer.push_back(c);
        lemma_buffer.push_back('\0');
        next_offset = lemma_buffer.size();

        // insert into trie
        int node = 0;
        for (char ch : key) {
            if (!temp_nodes[node].children.count(ch)) {
                int new_node = temp_nodes.size();
                temp_nodes[node].children[ch] = new_node;
                temp_nodes.emplace_back();
            }
            node = temp_nodes[node].children[ch];
        }

        temp_nodes[node].lemma_offset = lemma_offset;
    }

    // flatten trie into GPU structures
    for (const auto& node : temp_nodes) {
        GpuState gpu_state;
        gpu_state.transition_start_idx = transitions.size();
        gpu_state.num_transitions = node.children.size();
        gpu_state.lemma_offset = node.lemma_offset;

        for (const auto& [c, target] : node.children) {
            transitions.push_back({c, target});
        }

        states.push_back(gpu_state);
    }

    std::cout << "Built GPU trie with " << states.size()
              << " states, " << transitions.size()
              << " transitions, and " << lemma_buffer.size()
              << " bytes of lemma buffer.\n";
}


bool load_csv_dict_to_dawg(const std::string& path,
                           dawgdic::Dictionary& dict,
                           std::vector<char>& dict_vals) {
    dawgdic::Dawg dawg;
    return load_csv_dict_to_dawg(path, dawg, dict, dict_vals);
}


void save_dawg(const std::string& path, const dawgdic::Dictionary& dawg) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return;
    }
    dawg.Write(&file);
}

void load_dawg(const std::string& path, dawgdic::Dictionary& dawg) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << path << "\n";
        return;
    }
    dawg.Read(&file);
}
