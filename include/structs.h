//
// Created by Illya on 19.04.2025.
//

#ifndef STRUCTS_H
#define STRUCTS_H
#pragma once
#define MAX_WORD_LEN 32
#include <unordered_map>
#include <cstdint>

    struct TempTrieNode {
        std::unordered_map<char, int> children;
        int lemma_offset = -1;
    };

    struct GpuState {
        int transition_start_idx;  // index into transitions[]
        int num_transitions;       // how many edges from this state
        int lemma_offset;          // offset into flat lemma buffer (or -1)
    };

    struct GpuTransition {
        char c;
        int next_state;
    };

    struct Utf8TempTrieNode {
        std::unordered_map<uint16_t, int> children;
        int lemma_offset = -1;
    };

    struct Utf8Transition {
        uint16_t c;
        int next_state;
    };
#endif //STRUCTS_H
