# alphasort-bench

Input-ordering strategies for GPU trie lemmatization: does pre-sorting the
input stream (alphabetically and/or by length) pay for itself through improved
cache locality during trie traversal — and can an O(n) approximate ordering
capture most of the benefit without ever materializing a sorted corpus?

## Build & run

Two backends from one source. `bench_real` is the one that produces paper
numbers; `bench_synth` exists so the benchmark still builds and runs anywhere.

```bash
make            # -> bench_synth, self-contained stand-in trie
make real       # -> bench_real,  production trie from gpu_*.bin
make ARCH=sm_86 # default is sm_89 (RTX 40xx)
```

`bench_real` loads `gpu_states.bin`, `gpu_transitions.bin` and
`gpu_lemmas.bin` (9.35 M states, 260 MB on device) via `include/trie_real.cuh`,
which reproduces the traversal of `lookup_kernel` in
`src/cuda/lemmatizer_kernel.cu` exactly — one level per byte, linear scan over
each state's unsorted transition list. **Run from the repository root**, where
those files live:

```bash
./bench/alphasort/bench_real --words fiction_pp.txt --n 20000000 \
    --trie . --reps 3 --rounds 7 --key alpha
```

Corpora are one token per line, UTF-8; the loader lowercases Ukrainian inline
so tokens actually hit the dictionary (94.8 % on fiction, ~85 % on wiki).

Options:

| flag         | default   | meaning |
|--------------|-----------|---------|
| `--words`    | (none)    | corpus file, one word per line; synthetic data if absent |
| `--n`        | 20M       | number of word tokens (0 = whole file) |
| `--trie`     | `.`       | directory holding the three `gpu_*.bin` files (`bench_real`) |
| `--vocab`    | 300k      | synthetic vocabulary size (`bench_synth` only) |
| `--key`      | `alpha`   | `alpha` \| `len` \| `lenalpha` — the alpha/len comparison axis |
| `--prefix`   | 4         | X bytes for variant 3 (radix sort restricted to top X bytes) |
| `--binbits`  | 16        | bin key width for variants 4/5 (16 bits ≈ first Cyrillic letter) |
| `--batch`    | 2M        | words per batch in the streaming pipeline |
| `--streams`  | 4         | worker streams in the streaming pipeline |
| `--reps`     | 5         | lookup-kernel repetitions per timed sample |
| `--rounds`   | 7         | interleaved measurement rounds (median reported) |
| `--warm`     | 2000      | warm-up milliseconds before timing |
| `--prepreps` | 5         | permutation builds per variant (median reported) |
| `--lower`    | 1         | lowercase the corpus on load |
| `--lemma`    | 0         | also read the lemma string (adds the second random stream) |
| `--nocpu`    | 0         | skip variant 1 (its host sort costs tens of seconds at 50 M) |

## Measurement discipline (do not remove)

Two artifacts were found here, and both changed conclusions:

1. **Clock drift.** Without root, clocks cannot be pinned, and an idle GPU
   drops to 210 MHz core / 405 MHz memory against 3105 / 11501 max. Variant 1's
   multi-second host sort parks the clocks, so anything timed after it measures
   a ramping GPU. Timing variants in declaration order produced a fake monotone
   "improvement" that tracked execution order. Hence: all permutations are built
   first, then the kernels are timed **round-robin over `--rounds` rounds**
   after a busy warm-up, and medians are reported with max/min as a drift
   indicator.
2. **Single-shot prep.** The first build of an ordering absorbs CUDA module
   load and first-`cudaMalloc` cost — per-process, not per-batch — which
   inflated prep and produced a non-monotone one-shot curve. Hence
   `--prepreps` and median-of-k prep. This moved the headline break-even
   estimate by 28 %.

## Sweeps

```bash
bash bench/alphasort/run_scale_sweep.sh                  # 1 M -> 50 M, key=alpha
KEY=len TAG=len bash bench/alphasort/run_scale_sweep.sh  # other key modes
TAG=fine NS="500000 1000000 2000000 4000000" ROUNDS=9 \
    bash bench/alphasort/run_scale_sweep.sh              # custom grid
python3 bench/alphasort/parse_sweep.py alpha             # table + CSV + crossovers
```

`parse_sweep.py` prints speedup, absolute and per-token lookup cost, one-shot
economics (`prep + lookup` vs baseline), and interpolates each variant's
one-shot break-even corpus size. See `PAPER_PLAN.md` for results and the plan.

Every variant scatters results through the permutation (`out[originalIdx]`),
so downstream order is always preserved and **no sorted copy of the corpus is
ever built** — reordering exists only as a transient index array.

## The variants

**0 — baseline.** Natural input order. Reference for both kernel time and
checksum (all variants must match it).

**1 — cpu-sort.** Host `std::sort` of an index array by a 64-bit prefix key
(first 8 bytes, big-endian-packed so integer order = bytewise lexicographic
order), with full `memcmp` tie-break beyond 8 bytes. Measures the classic
"sort on CPU while GPU does something else" option; in a pipeline its cost can
be hidden behind transfers, but here it is charged to `prep`.

**2 — gpu-sort (8 B).** `cub::DeviceRadixSort::SortPairs` over the full 64-bit
key. For Ukrainian UTF-8 (2 bytes per Cyrillic letter) this orders by the
first 4 letters — already deeper than the trie levels where locality matters.

**2b — gpu-full.** Exact full-depth sort, entirely on GPU. One full radix pass
(identical to variant 2), then iterative refinement: runs of equal keys are
detected (persistent head flags + inclusive scan), the next 8 bytes are packed
as depth-d keys, and each tie run is sorted with `cub::DeviceSegmentedSort`.
Two details matter: (a) boundaries only ever *accumulate*, so a later pass can
never reorder across an earlier boundary; (b) runs of **identical tokens** —
enormous under Zipf — are detected via all-zero depth keys (NUL bytes cannot
occur in text) and excluded, so they are never re-sorted and cannot keep the
loop alive. The loop terminates when no active tie runs remain; the pass count
is printed. The expected result: prep ≈ variant 2 + ε (ties beyond 8 bytes are
rare), and lookup time identical to variant 2 — the saturation argument that
justifies variants 3/4. For `--key len`/`lenalpha` the refinement resumes at
the correct byte offset, so the final order is exact (len, alpha) either way.

**3 — gpu-prefix (X B).** Same sort restricted to bit range
`[(8−X)·8, 64)`, i.e. only the top X bytes participate. Fewer radix passes →
cheaper prep. The hypothesis: locality benefit saturates at small X because
the top trie levels are L2-resident anyway and the win comes from grouping
warps onto shared mid-level subtries.

**4 — gpu-partition.** Exactly one hand-rolled radix pass: histogram over the
top `binbits` bits of the key, `cub::DeviceScan::ExclusiveSum`, scatter. O(n),
two data-touching kernels, produces bin-level (approximate) ordering — words
sharing a first letter become contiguous, order within a bin is arbitrary.
This is the "avoid the sort, keep the locality" candidate.

**6 / 6b — sort+compact.** Sort as in variant 2 (8 bytes) or at 2-byte
precision (6b), then *materialise the words in that order* with a gather pass,
so the lookup reads them sequentially instead of through the permutation. This
is the variant the profiler demanded: reordering alone wins warp uniformity but
loses input-stream coalescing (~2 sectors per word instead of one shared line
per warp, ≈3.7× DRAM traffic), and compaction is what buys the coalescing back.
The compact copy is a transient device buffer — the corpus on disk and the
output order are untouched, and results still scatter to `out[originalIdx]`.
6b is the scale-invariant winner: 0.15 → 0.16 ns/token from 1 M to 50 M tokens,
overtaking 6 at ~20 M.

**5 — streaming.** Double-buffered batched pipeline: a copy stream does
H2D + key generation + partition for batch k+1 while worker streams run
lookups for batch k (slot reuse is guarded by device-side events, no host
sync in the loop). The bin space is split into `--streams` contiguous slices
and the slice→stream mapping is **static**, so worker stream *s* processes the
same first-letter range in every batch — the "feed the same kernel similar
data" scheme. Run twice automatically: without partition (pure pipeline
baseline) and with it.

## What to measure for the paper

- **prep / lookup / total** per variant (printed): the core trade-off curve.
  The interesting quantity is `Δlookup` (locality gain) vs `prep` (ordering
  cost), and where each strategy sits relative to the amortization break-even.
- **L2 hit rate** during the lookup kernel, per variant:
  ```bash
  ncu --kernel-name lookupKernel \
      --metrics lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,\
  smsp__sass_average_branch_targets_threads_uniform.pct ./bench --reps 1
  ```
  The branch-uniformity metric is the one that responds to `--key len`
  (uniform loop trip counts within a warp) as opposed to `--key alpha`
  (shared trie paths within a warp) — that separation is the alpha-vs-len
  story in one profiler line.
- **Sensitivity sweeps:** `--prefix 2/4/8`, `--binbits 8/16/24`,
  `--streams 1/2/4/8`, batch size. Expect variant 3/4 curves to flatten
  early — that saturation point is a result in itself.
- **Zipf skew matters:** with real corpora, high-frequency tokens repeat, so
  even the baseline gets incidental cache reuse; report vocab entropy or
  type/token ratio alongside throughput.

## Honest caveats (worth stating in the paper)

- On a single GPU the L2 is shared across all SMs, so the multi-stream
  variant does **not** give each kernel a private cache. What the static
  bin→stream mapping buys is (a) copy/compute overlap and (b) a *partitioned
  working set*: concurrently resident kernels touch disjoint subtries instead
  of thrashing each other's lines, plus temporal reuse of the same subtrie by
  the same stream across consecutive batches. Frame it as working-set
  partitioning, not cache isolation.
- Word-byte reads stay uncoalesced in all variants (variable-length strings
  read through a permutation). The measured effect is therefore isolated to
  trie-node access locality, which is the quantity of interest. A gather pass
  that also compacts the strings into sorted order is a possible extra
  variant, but it reintroduces exactly the "sorted rebuild" cost the design
  avoids.
- Variant 5's segment kernels launch a full-batch-sized grid and let
  out-of-segment threads exit immediately — simple and fully async, at the
  cost of some idle-thread launch overhead. With `--binbits 16` and Ukrainian
  first-letter frequencies, segments are also load-imbalanced (о-, п-, з-
  initial words dominate); per-stream times can be captured with events if
  imbalance needs quantifying.
- The bundled trie is a stand-in with a representative access pattern
  (binary search over sorted edges, one level per byte, BFS-flattened
  layout). Swap `TrieDev`/`trieLookup` in `include/trie.cuh` for the real
  9.35M-state lemmatizer to produce paper numbers; the harness does not care
  what the lookup does as long as it returns an `int32_t`.

## Porting to Metal

The same experiment maps 1:1: variant 2/3 → `MPSMatrixSort` is a poor fit, so
use a compute-shader radix sort or sort on CPU (variant 1 is *cheaper* to
justify on Apple silicon because unified memory removes the perm-array
upload); variant 4/5 → identical histogram/scan/scatter shaders, with
`MTLSharedEvent` replacing CUDA events and one command queue per worker
"stream". The zero-copy angle strengthens the streaming variant on Metal —
no staging copies at all.
