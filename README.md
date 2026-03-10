# cuda-lemmatizer-cpp — Benchmark Guide

GPU-accelerated Ukrainian morphological lemmatizer. Trie lookup runs entirely on the GPU via CUDA kernels + RAPIDS cuDF.

---

## Prerequisites

- CUDA 12.8 at `/usr/local/cuda-12.8/`
- RAPIDS (cuDF + RMM) via conda at `~/miniconda3`
- g++-13 as host compiler
- Run all benchmarks **from `cmake-build-debug/`** — executables load `gpu_states.bin`, `gpu_transitions.bin`, `gpu_lemmas.bin` by relative path

---

## Build

```bash
# From project root
cmake -B cmake-build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_PREFIX_PATH=$HOME/miniconda3 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13

# Build all bench targets at once
cmake --build cmake-build-debug --target \
  lemmatize_cpu \
  lemmatize_gpu \
  lemmatize_gpu_loop \
  lemmatize_gpu_stride \
  lemmatize_gpu_stride_loop \
  lemmatize_gpu_bsearch
```

---

## Input files

| File | Words | Use |
|------|-------|-----|
| `articles.txt` | ~761k | standard benchmark input |
| `articles_big.txt` | ~166M | stress / throughput test |

All bench commands below use `articles.txt`. Swap in `articles_big.txt` for larger runs.

---

## Benchmark suite

All commands must be run from `cmake-build-debug/`.

```bash
cd cmake-build-debug
```

---

### 1. CPU baseline

Measures pure CPU trie lookup (no CUDA, no cuDF).

```bash
./lemmatize_cpu ../articles.txt
```

Output:
```
Words: 761625  Time: 308 ms  Throughput: 2.47M words/sec
```

---

### 2. GPU one-shot (`lemmatize_gpu`)

Single pass end-to-end: preprocess → H2D → kernel → D2H. Use this to measure real-world latency.

```bash
./lemmatize_gpu ../articles.txt
# with lemmatized output written to file:
./lemmatize_gpu ../articles.txt output.txt
```

Output breakdown:
```
Words: 761625
  Preprocess (I/O+tokenize+lowercase+arrays): 747 ms
  H2D (upload words+trie):                    326 ms
  Kernel:                                      13 ms  (58M words/sec)
  D2H (download results):                      16 ms
  GPU total (H2D+kernel+D2H):                 357 ms
  End-to-end (preprocess+GPU):               1104 ms
```

**Note:** Preprocess dominates (ICU lowercasing on 761k words). Kernel is 13 ms.

---

### 3. GPU loop — steady-state throughput (`lemmatize_gpu_loop`)

Uploads data once, hammers the kernel repeatedly for N seconds. Measures pure GPU throughput after L2 cache warms up (~800 iters).

```bash
# Run for 30 seconds (recommended for stable avg)
./lemmatize_gpu_loop ../articles.txt 30

# Quick 10-second run
./lemmatize_gpu_loop ../articles.txt 10
```

Output:
```
Preprocess (I/O+tokenize+lowercase+arrays): 699 ms
Running for 30s  words=761625  blocks=5951  threads=128
iter  5000  avg 0.696 ms  throughput 1093.94M words/sec
...
=== Final ===
  Iters:       39866
  Words/iter:  761625
  Avg kernel:  0.698 ms
  Peak kernel: 12.648 ms   ← cold first iter
  Throughput:  1091.36M words/sec
```

**Key number: ~1.09B words/sec steady-state.**

---

### 4. GPU stride-layout loop (`lemmatize_gpu_stride_loop`)

Same loop benchmark but using fixed-stride input layout (words packed at `MAX_WORD_LEN` intervals) instead of cuDF packed offsets.

```bash
./lemmatize_gpu_stride_loop ../articles.txt 30
./lemmatize_gpu_stride_loop ../articles.txt 10
```

---

### 5. GPU stride one-shot (`lemmatize_gpu_stride`)

Single-pass with stride layout.

```bash
./lemmatize_gpu_stride ../articles.txt
```

---

### 6. GPU binary-search trie (`lemmatize_gpu_bsearch`)

Variant using binary search over sorted transitions instead of linear scan.

```bash
./lemmatize_gpu_bsearch ../articles.txt
```

---

## Running the full suite at once

```bash
cd cmake-build-debug

echo "=== CPU ===" && \
./lemmatize_cpu ../articles.txt && \

echo "=== GPU one-shot ===" && \
./lemmatize_gpu ../articles.txt && \

echo "=== GPU loop (30s) ===" && \
./lemmatize_gpu_loop ../articles.txt 30 && \

echo "=== GPU stride loop (30s) ===" && \
./lemmatize_gpu_stride_loop ../articles.txt 30 && \

echo "=== GPU bsearch ===" && \
./lemmatize_gpu_bsearch ../articles.txt
```

---

## Timing phases explained

| Phase | What it measures | CUDA mechanism |
|-------|-----------------|----------------|
| Preprocess | File I/O + tokenize + `lowercase_ukr()` (ICU) + build offset arrays | `chrono::high_resolution_clock` |
| H2D | `cudaMemcpy` of words + trie to device | `chrono` + `cudaDeviceSynchronize` |
| Kernel | Pure GPU trie traversal | `cudaEvent` (hardware timestamp) |
| D2H | `cudaMemcpy` results back to host | `chrono` |
| GPU total | H2D + kernel + D2H | `chrono` |
| End-to-end | Preprocess + GPU total | sum |

Loop benchmarks report only **Kernel** time (H2D is one-time setup, D2H is skipped).
