# CUDA 12.8 vs 13.0 — toolkit sensitivity check

Generated 2026-09-07. Same patched `src/bench.cu` compiled by both toolkits,
runs interleaved per `n` to control for thermal drift. Corpus `wiki_sample_50m.txt`,
key `alpha`, reps 3, rounds 5, real 9.35M-state trie.

`results/` (the paper's numbers) is untouched; everything here is separate.

## Provenance

- 12.8 binary: `bench/alphasort/bench_real`, nvcc 12.8 + g++-13
- 13.0 binary: scratch build, nvcc 13.0 + system gcc 15 (no -ccbin needed)
- Patch required for 13.0: `cub::CountingInputIterator` -> `thrust::counting_iterator`
  (removed in CCCL 3.0). Verified codegen-neutral: all 13 own kernels are
  byte-identical SASS pre- vs post-patch under 12.8.

## Result: no conclusion changes

Speedup vs baseline (12.8 / 13.0):

| variant | 1M | 2M | 5M | 10M | 20M | 35M | 50M |
|---|---|---|---|---|---|---|---|
| 2 gpu-sort-8B | 3.02/3.03 | 3.22/3.22 | 1.65/1.49 | 1.31/1.17 | 1.07/1.06 | 0.99/0.99 | 0.96/0.94 |
| 2b gpu-full | 2.93/2.95 | 3.07/3.07 | 1.68/1.68 | 1.20/1.14 | 1.01/1.03 | 0.96/0.95 | 0.93/0.89 |
| 3 gpu-prefix | 2.69/2.70 | 2.85/2.84 | 1.94/1.95 | 1.55/1.39 | 1.21/1.23 | 1.18/1.18 | 1.17/1.13 |
| 4 gpu-partition | 1.99/2.01 | 2.07/2.06 | 1.46/1.82 | 1.58/1.73 | 1.45/1.50 | 1.48/1.48 | 1.42/1.43 |
| 6 sort+compact | 6.14/6.20 | 5.16/5.17 | 5.90/5.91 | 5.32/4.94 | 3.39/3.46 | 2.67/2.68 | 2.43/2.37 |
| 6b sort-2B+compact | 3.81/3.83 | 3.67/3.66 | 3.88/3.88 | 4.18/3.89 | 3.15/3.45 | 3.33/3.33 | 3.14/3.12 |

**Crossover reproduces identically.** Exact ordering first drops below 1.00x at
**35M on both toolkits**, for both `2 gpu-sort-8B` and `2b gpu-full` — consistent
with the paper's "~27M" claim (crossover lies between 20M and 35M).

**CUB 2.x -> 3.0 prep cost**: worst case +-15%, no systematic direction. Not a
regression. An earlier concern that CUB 3.0 might invalidate the prep-stage
numbers is NOT borne out.

## Free noise-floor calibration

The own kernels are byte-identical SASS across toolkits, so every `lookup ms`
difference here is *provably* measurement noise rather than codegen. All 8
outliers >=4% fall inside the run's own self-reported max/min spread:

| variant | n | 12.8 | 13.0 | delta | max/min |
|---|---|---|---|---|---|
| 4 gpu-partition | 5M | 1.845 | 1.479 | -19.8% | 1.276 |
| 4 gpu-partition | 10M | 3.883 | 3.306 | -14.9% | 1.272 |
| 2 gpu-sort-8B | 5M | 1.626 | 1.806 | +11.1% | 1.263 |
| 6b sort-2B+compact | 20M | 3.526 | 3.197 | -9.3% | 1.265 |
| 0 baseline | 10M | 6.146 | 5.714 | -7.0% | 1.213 |
| 4 gpu-partition | 50M | 19.530 | 18.672 | -4.4% | 1.098 |
| 4 gpu-partition | 20M | 7.657 | 7.336 | -4.2% | 1.074 |
| 0 baseline | 50M | 27.779 | 26.636 | -4.1% | 1.100 |

Two things follow, both worth stating in the threats-to-validity section:
1. The `max/min` column never under-reports true spread — the instrumentation is honest.
2. Differences below ~20% at the 5-10M scales are NOT resolvable by a single run.

---

# Part 2: the shipped library under CUDA 13

Built against an **isolated** conda env `rapids13` (`libcudf 26.02.01
cuda13_260205_5b9658c4` — same version AND build hash as the base env's cuda12
build, so cuDF behaviour is identical by construction). The base env was not
touched.

To recreate the env (it is deleted after use; ~877 MB):

```
conda create -n rapids13 -y -c rapidsai -c conda-forge -c nvidia \
  "libcudf=26.02.01=cuda13*" "librmm=26.02.00=cuda13*" "cuda-version=13"
```

Then:

```
cmake -B build-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_PREFIX_PATH=/home/kowalski0805/miniconda3/envs/rapids13
```

## Both open unknowns resolved — both fine

1. **All 6 sources of `lemmatizer` compile clean** against cuda13 cuDF headers
   (`GpuLemmatizer.cpp`, `batch_stats.cpp`, `fused_lemmatize.cu`, `lemmatizer.cpp`,
   `lemmatizer_kernel.cu`, `trie.cpp`). `batch_probe` also builds. No source
   changes needed anywhere outside `bench/alphasort/src/bench.cu`.
2. **`LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` is harmless** under CCCL 3.0 —
   no error, no warning. It can stay or go.

`batch_probe --selftest`: **14/14 pathological rows match** under CUDA 13.

## Fusion speedup reproduces

Mean over 4 repeats, cold-start outlier dropped:

| tokens | 12.8 staged | 12.8 fused | 12.8 | 13.0 staged | 13.0 fused | 13.0 |
|---|---|---|---|---|---|---|
| 200 K | 12.41 | 1.79 | **6.93x** | 12.87 | 1.80 | **7.14x** |
| 1 M | 21.88 | 4.79 | **4.58x** | 22.79 | 4.79 | **4.76x** |
| 4 M | 46.81 | 10.72 | **4.36x** | 49.68 | 11.32 | **4.40x** |
| 10 M | 102.42 | 24.09 | **4.26x** | 106.16 | 25.04 | **4.24x** |

Every row bit-identical on device, on both toolkits. Absolute times run ~2-6 %
slower under 13.0, but staged and fused move together so the ratio is preserved —
which is the quantity the paper claims.

## Incidental finding: the 12.8 build loads TWO CUDA runtimes

```
cmake-build-debug/liblemmatizer.so  ->  libcudart.so.13  (/usr/local/cuda -> 13)
                                        libcudart.so.12  (conda)
build-cuda13/liblemmatizer.so      ->  libcudart.so.13  only
```

The existing build pulls in both, because `-DCUDAToolkit_ROOT=/usr/local/cuda`
resolves to 13 while cuDF drags in 12. It evidently works, but the CUDA 13 build
is strictly cleaner here. Setting `-DCUDAToolkit_ROOT=/usr/local/cuda-12.8` fixes
it for the 12.8 build without any other change.

## Build-system change

`CMakeLists.txt` now guards the hardcoded nvcc path with `if(NOT DEFINED
CMAKE_CUDA_COMPILER)`, so `-DCMAKE_CUDA_COMPILER=...` works. Verified
default-preserving: no override resolves to 12.8.93, override gives 13.0.88.
