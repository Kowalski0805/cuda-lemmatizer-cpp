// trie.cuh — a self-contained byte-level flat trie.
//
// This is a stand-in for the dissertation's GPU-resident trie so the benchmark
// compiles and runs on its own. The memory-access pattern (pointer-chasing
// through node/edge arrays, one level per input byte, binary search over the
// sorted edge list of each node) is representative of the real structure, so
// the *relative* effect of input ordering on cache locality carries over.
// Swap `TrieDev` / `trieLookup` for the real lemmatizer to get paper numbers.
#pragma once
#include "common.cuh"
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

struct TrieBuilder {
  struct Node {
    std::map<uint8_t, int> ch;  // ordered => flattened edges are sorted
    int32_t lemma = -1;
  };
  std::vector<Node> nodes;
  TrieBuilder() { nodes.emplace_back(); }

  void insert(const uint8_t* w, uint32_t len, int32_t lemmaId) {
    int cur = 0;
    for (uint32_t i = 0; i < len; ++i) {
      auto it = nodes[cur].ch.find(w[i]);
      if (it == nodes[cur].ch.end()) {
        int nx = (int)nodes.size();
        nodes[cur].ch.emplace(w[i], nx);
        nodes.emplace_back();
        cur = nx;
      } else {
        cur = it->second;
      }
    }
    nodes[cur].lemma = lemmaId;
  }
};

struct TrieFlatHost {
  std::vector<uint32_t> firstEdge;  // per node: index into edge arrays
  std::vector<uint16_t> numEdges;   // per node: number of outgoing edges
  std::vector<int32_t> lemma;       // per node: lemma id or -1
  std::vector<uint8_t> edgeLabel;   // sorted within each node
  std::vector<uint32_t> edgeChild;
};

// Flatten in BFS order: shallow (hot) levels end up contiguous at the front of
// the arrays, which mirrors how the real trie benefits from L2 residency of
// the top of the tree.
inline TrieFlatHost flattenTrie(const TrieBuilder& tb) {
  const size_t N = tb.nodes.size();
  std::vector<int> order;
  order.reserve(N);
  std::vector<int> remap(N, -1);
  std::deque<int> q;
  q.push_back(0);
  remap[0] = 0;
  order.push_back(0);
  while (!q.empty()) {
    int u = q.front();
    q.pop_front();
    for (const auto& kv : tb.nodes[u].ch) {
      remap[kv.second] = (int)order.size();
      order.push_back(kv.second);
      q.push_back(kv.second);
    }
  }

  TrieFlatHost f;
  f.firstEdge.resize(N);
  f.numEdges.resize(N);
  f.lemma.resize(N);
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const TrieBuilder::Node& nd = tb.nodes[order[oi]];
    f.firstEdge[oi] = (uint32_t)f.edgeLabel.size();
    f.numEdges[oi] = (uint16_t)nd.ch.size();
    f.lemma[oi] = nd.lemma;
    for (const auto& kv : nd.ch) {
      f.edgeLabel.push_back(kv.first);
      f.edgeChild.push_back((uint32_t)remap[kv.second]);
    }
  }
  return f;
}

struct TrieDev {
  const uint32_t* firstEdge;
  const uint16_t* numEdges;
  const int32_t* lemma;
  const uint8_t* edgeLabel;
  const uint32_t* edgeChild;
};

struct TrieDevStorage {
  uint32_t* firstEdge = nullptr;
  uint16_t* numEdges = nullptr;
  int32_t* lemma = nullptr;
  uint8_t* edgeLabel = nullptr;
  uint32_t* edgeChild = nullptr;

  TrieDev upload(const TrieFlatHost& f) {
    auto up = [](auto*& dst, const auto& vec) {
      using T = typename std::remove_reference<decltype(vec)>::type::value_type;
      CUDA_CHECK(cudaMalloc(&dst, vec.size() * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(dst, vec.data(), vec.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
    };
    up(firstEdge, f.firstEdge);
    up(numEdges, f.numEdges);
    up(lemma, f.lemma);
    up(edgeLabel, f.edgeLabel);
    up(edgeChild, f.edgeChild);
    return TrieDev{firstEdge, numEdges, lemma, edgeLabel, edgeChild};
  }
  void free() {
    cudaFree(firstEdge);
    cudaFree(numEdges);
    cudaFree(lemma);
    cudaFree(edgeLabel);
    cudaFree(edgeChild);
  }
};

__device__ inline int32_t trieLookup(const TrieDev t, const uint8_t* w,
                                     uint32_t len) {
  uint32_t cur = 0;
  for (uint32_t i = 0; i < len; ++i) {
    const uint8_t c = w[i];
    uint32_t lo = t.firstEdge[cur];
    uint32_t hi = lo + t.numEdges[cur];
    int64_t found = -1;
    while (lo < hi) {  // binary search over sorted edge labels
      uint32_t mid = (lo + hi) >> 1;
      uint8_t l = t.edgeLabel[mid];
      if (l == c) {
        found = (int64_t)mid;
        break;
      }
      if (l < c)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (found < 0) return -1;  // OOV
    cur = t.edgeChild[found];
  }
  return t.lemma[cur];
}
