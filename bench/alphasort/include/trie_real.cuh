// trie_real.cuh — the production lemmatizer trie, loaded from the repository's
// flat binaries (gpu_states.bin / gpu_transitions.bin / gpu_lemmas.bin).
//
// Drop-in replacement for the synthetic `trie.cuh`: same `TrieDev` /
// `trieLookup` interface, so bench.cu is agnostic to which one is compiled in
// (-DUSE_REAL_TRIE selects this one).
//
// Layout is byte-identical to `include/structs.h` in the parent project, since
// the .bin files are raw dumps of std::vector<GpuState> / <GpuTransition>:
//   GpuState      { int32 transition_start_idx; int32 num_transitions;
//                   int32 lemma_offset; }        -> 12 B
//   GpuTransition { char c; int32 next_state; }  ->  8 B (4-byte aligned)
//
// The traversal below is a faithful copy of `lookup_kernel` in
// src/cuda/lemmatizer_kernel.cu: one trie level per *byte*, linear scan over
// the (unsorted) transition list of each state. Keeping the scan linear rather
// than "improving" it to a binary search is deliberate — the paper measures the
// effect of input ordering on the deployed kernel, not on an idealised one.
#pragma once
#include "common.cuh"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct RealState {
  int32_t transition_start_idx;
  int32_t num_transitions;
  int32_t lemma_offset;
};

struct RealTransition {
  char c;
  int32_t next_state;
};

static_assert(sizeof(RealState) == 12, "GpuState layout mismatch");
static_assert(sizeof(RealTransition) == 8, "GpuTransition layout mismatch");

struct TrieDev {
  const RealState* states;
  const RealTransition* transitions;
  const char* lemmas;
  int32_t numStates;
  int32_t numTransitions;
  int32_t lemmaBytes;
  int touchLemma;  // 1: also read the lemma string (second random-access stream)
};

template <typename T>
static bool loadBinVec(const std::string& path, std::vector<T>& v) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize sz = f.tellg();
  if (sz <= 0 || sz % (std::streamsize)sizeof(T)) return false;
  f.seekg(0);
  v.resize((size_t)sz / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), sz);
  return (bool)f;
}

struct TrieDevStorage {
  RealState* states = nullptr;
  RealTransition* transitions = nullptr;
  char* lemmas = nullptr;
  int32_t nStates = 0, nTrans = 0, nLemmaBytes = 0;

  // dir: directory holding the three .bin files (e.g. "." or "../..").
  bool load(const std::string& dir, int touchLemma) {
    std::vector<RealState> hS;
    std::vector<RealTransition> hT;
    std::vector<char> hL;
    if (!loadBinVec(dir + "/gpu_states.bin", hS)) {
      fprintf(stderr, "cannot read %s/gpu_states.bin\n", dir.c_str());
      return false;
    }
    if (!loadBinVec(dir + "/gpu_transitions.bin", hT)) {
      fprintf(stderr, "cannot read %s/gpu_transitions.bin\n", dir.c_str());
      return false;
    }
    if (!loadBinVec(dir + "/gpu_lemmas.bin", hL)) {
      fprintf(stderr, "cannot read %s/gpu_lemmas.bin\n", dir.c_str());
      return false;
    }
    nStates = (int32_t)hS.size();
    nTrans = (int32_t)hT.size();
    nLemmaBytes = (int32_t)hL.size();
    touch_ = touchLemma;

    CUDA_CHECK(cudaMalloc(&states, hS.size() * sizeof(RealState)));
    CUDA_CHECK(cudaMalloc(&transitions, hT.size() * sizeof(RealTransition)));
    CUDA_CHECK(cudaMalloc(&lemmas, hL.size()));
    CUDA_CHECK(cudaMemcpy(states, hS.data(), hS.size() * sizeof(RealState),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(transitions, hT.data(),
                          hT.size() * sizeof(RealTransition),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lemmas, hL.data(), hL.size(),
                          cudaMemcpyHostToDevice));
    const double mb = (hS.size() * sizeof(RealState) +
                       hT.size() * sizeof(RealTransition) + hL.size()) /
                      1048576.0;
    printf("real trie: %d states, %d transitions, %d lemma bytes (%.1f MB on "
           "device)\n",
           nStates, nTrans, nLemmaBytes, mb);
    return true;
  }

  TrieDev dev() const {
    return TrieDev{states,      transitions, lemmas, nStates,
                   nTrans,      nLemmaBytes, touch_};
  }

  void free() {
    cudaFree(states);
    cudaFree(transitions);
    cudaFree(lemmas);
  }

 private:
  int touch_ = 0;
};

// Returns the lemma offset for a recognised wordform, or -1 for OOV / no lemma
// — i.e. exactly the branch the production kernel takes before emitting the
// output string. With touchLemma the lemma bytes are read as well, so the
// second (also permutation-dependent) memory stream is included in the measured
// access pattern; the returned value then mixes in a hash of those bytes so the
// reads cannot be optimised away and checksums still compare across variants.
__device__ inline int32_t trieLookup(const TrieDev t, const uint8_t* w,
                                     uint32_t len) {
  int32_t state = 0;
  for (uint32_t i = 0; i < len; ++i) {
    const char ch = (char)w[i];
    const RealState s = t.states[state];
    int32_t next = -1;
    for (int32_t j = 0; j < s.num_transitions; ++j) {
      const RealTransition tr = t.transitions[s.transition_start_idx + j];
      if (tr.c == ch) {
        next = tr.next_state;
        break;
      }
    }
    if (next < 0) return -1;  // OOV
    state = next;
  }
  const int32_t off = t.states[state].lemma_offset;
  if (off < 0) return -1;
  if (!t.touchLemma) return off;

  int32_t h = off;
  for (int32_t i = 0; i < 64; ++i) {
    const char c = t.lemmas[off + i];
    if (c == '\0') break;
    h = h * 31 + (int32_t)(uint8_t)c;
  }
  return h & 0x7fffffff;
}
