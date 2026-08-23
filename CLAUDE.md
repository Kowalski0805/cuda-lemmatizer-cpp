# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated Ukrainian morphological lemmatizer. Takes Ukrainian wordforms and maps them to their base (lemma) forms using a trie data structure that runs entirely on the GPU via CUDA kernels and RAPIDS cuDF.

## Build Commands

```bash
# Configure (from project root)
cmake -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug

# Build all targets
cmake --build cmake-build-debug

# Build only the shared library
cmake --build cmake-build-debug --target lemmatizer

# Build only the debug executable
cmake --build cmake-build-debug --target cuda_exe

# Run the debug executable (must be run from cmake-build-debug/ where data files live)
cd cmake-build-debug && ./cuda_exe
```

**Note:** The executable must be run from `cmake-build-debug/` because it loads data files (`gpu_states.bin`, `gpu_transitions.bin`, `gpu_lemmas.bin`, `morph_uk.dawg`, `dict_vals.bin`) by relative path.

## Architecture

### Two Build Targets

- **`liblemmatizer.so`** — the main deliverable, a JNI shared library consumed by a Java caller
- **`cuda_exe`** — a standalone C++ debug/test executable (links against `liblemmatizer.so`)

### Data Pipeline

Dictionary source: `ukr_morph_dict.csv` — CSV of `(wordform, lemma)` pairs for Ukrainian.

Two dictionary representations are used:
1. **GPU trie** (`gpu_states.bin`, `gpu_transitions.bin`, `gpu_lemmas.bin`) — flat trie structure serialized for GPU. Built once from CSV via `build_gpu_trie_from_csv()`, then loaded at runtime by `init_trie_data()`.
2. **DAWG** (`morph_uk.dawg`, `dict_vals.bin`) — a directed acyclic word graph for CPU-side lookup, built with the `dawgdic` library.

### Component Responsibilities

| File | Role |
|------|------|
| `src/cuda/trie.cpp` | CPU-side: loads CSV/flat-binary/DAWG dictionaries, builds GPU trie structures |
| `src/cuda/lemmatizer_kernel.cu` | All CUDA kernels: `lookup_kernel`, `normalize_kernel`, `dawg_lookup_kernel`, `sizes_kernel`, `lemmatize_kernel` |
| `src/cuda/lemmatizer.cpp` | C++ API: `init_trie_data()` (singleton load) + `lemmatize_batch()` returning a cuDF string column |
| `src/cuda/GpuLemmatizer.cpp` | JNI bridge: sentence-level pipeline — splits sentences → lemmatizes words → re-joins per sentence |
| `src/cuda/icu_lowercase.cpp` | ICU-based Ukrainian lowercasing (`lowercase_ukr()`) applied before trie lookup |
| `include/structs.h` | Core GPU data structures: `GpuState`, `GpuTransition`, `TempTrieNode` |
| `src/cuda/main.cu` | Debug `main()`: exercises `main_dawg()` (CPU DAWG lookup) and `main_gpu()` (GPU trie lookup) |

### GPU Trie Layout

The trie is flattened into two parallel arrays:
- `GpuState[]` — one per trie node: `{transition_start_idx, num_transitions, lemma_offset}`
- `GpuTransition[]` — all edges packed contiguously: `{c, next_state}`
- `char[]` — flat lemma string buffer, indexed by `lemma_offset`

`MAX_WORD_LEN = 32` bytes (fixed-width buffers for the older `normalize_kernel`/`dawg_lookup_kernel`). The primary `lookup_kernel` uses cuDF `string_view` and is not limited by this.

### JNI Data Flow (`GpuLemmatizer.lemmatize`)

1. Receives `jlong` pointer to a cuDF `column_view` of sentence strings
2. `cudf::strings::split_record(sentences, " ")` → list-of-words column
3. `cudf::explode_position()` → flat (sentence_id, word) pairs
4. `lemmatize_batch(words)` → GPU trie lookup via `lookup_kernel`
5. `cudf::groupby` + `collect_list` → re-group lemmas by sentence
6. `cudf::strings::join_list_elements` → sentence strings of lemmas
7. Returns `jlong` pointer to result column (Java takes ownership)

### Key Dependencies

- **CUDA 12.8** at `/usr/local/cuda-12.8/`, architecture `89` (RTX 4xxx)
- **RAPIDS cuDF + RMM** — GPU dataframe/memory management
- **ICU** (uc, i18n) — Unicode lowercasing
- **JDK 21** at `/usr/lib/jvm/java-21-openjdk-amd64/`
- **dawgdic** (header-only, in `include/` or system) — DAWG builder/lookup

### `init_trie_data()` Singleton

`lemmatizer.cpp` maintains static GPU pointers (`d_states`, `d_transitions`, `d_lemmas`) that are allocated and filled once on first call to `lemmatize_batch()`. There is no cleanup/destructor — intended for library lifetime.
