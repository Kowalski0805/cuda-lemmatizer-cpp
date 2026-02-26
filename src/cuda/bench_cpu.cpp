#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "structs.h"
#include "trie.h"
#include "icu_lowercase.h"

static std::string_view cpu_lookup(
    std::string_view word,
    const std::vector<GpuState>& states,
    const std::vector<GpuTransition>& transitions,
    const std::vector<char>& lemmas)
{
    int state = 0;
    for (char ch : word) {
        const GpuState& s = states[static_cast<size_t>(state)];
        bool found = false;
        for (int j = 0; j < s.num_transitions; ++j) {
            if (transitions[static_cast<size_t>(s.transition_start_idx + j)].c == ch) {
                state = transitions[static_cast<size_t>(s.transition_start_idx + j)].next_state;
                found = true;
                break;
            }
        }
        if (!found) return word;
    }
    const GpuState& fs = states[static_cast<size_t>(state)];
    if (fs.lemma_offset >= 0) {
        const char* p = lemmas.data() + fs.lemma_offset;
        return {p, strnlen(p, MAX_WORD_LEN)};
    }
    return word;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }
    const std::string input_path = argv[1];
    const bool has_output = (argc >= 3);
    const std::string output_path = has_output ? argv[2] : "";

    // Load trie data
    std::vector<GpuState> h_states;
    std::vector<GpuTransition> h_transitions;
    std::vector<char> h_lemmas;
    load_bin_vector("gpu_states.bin", h_states);
    load_bin_vector("gpu_transitions.bin", h_transitions);
    load_bin_vector("gpu_lemmas.bin", h_lemmas);

    if (h_states.empty()) {
        std::cerr << "Failed to load trie data. Run from cmake-build-debug/ where .bin files live.\n";
        return 1;
    }

    // Read input file
    std::ifstream fin(input_path);
    if (!fin) {
        std::cerr << "Cannot open input file: " << input_path << "\n";
        return 1;
    }
    std::vector<std::string> lines;
    {
        std::string line;
        while (std::getline(fin, line)) lines.push_back(std::move(line));
    }

    // Tokenize and lowercase; track per-line word counts for output reconstruction
    std::vector<int> line_word_count(lines.size(), 0);
    std::vector<std::string> words;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::istringstream ss(lines[i]);
        std::string token;
        while (ss >> token) {
            words.push_back(lowercase_ukr(token));
            ++line_word_count[i];
        }
    }
    const long long total_words = static_cast<long long>(words.size());

    // --- TIMED REGION START ---
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> result_words(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        auto lemma = cpu_lookup(words[i], h_states, h_transitions, h_lemmas);
        result_words[i] = std::string(lemma);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    // --- TIMED REGION END ---

    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double throughput = (ms > 0.0) ? (static_cast<double>(total_words) / (ms / 1000.0)) : 0.0;
    std::cerr << "Words: " << total_words
              << "  Time: " << ms << " ms"
              << "  Throughput: " << static_cast<long long>(throughput) << " words/sec\n";

    // Write output, preserving line structure
    std::ostream* out_ptr = &std::cout;
    std::ofstream fout;
    if (has_output) {
        fout.open(output_path);
        if (!fout) {
            std::cerr << "Cannot open output file: " << output_path << "\n";
            return 1;
        }
        out_ptr = &fout;
    }

    int word_idx = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        for (int j = 0; j < line_word_count[i]; ++j) {
            if (j > 0) *out_ptr << ' ';
            *out_ptr << result_words[word_idx++];
        }
        *out_ptr << '\n';
    }

    return 0;
}
