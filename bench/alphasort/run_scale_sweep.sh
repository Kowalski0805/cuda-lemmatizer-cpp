#!/usr/bin/env bash
# Scale sweep: one corpus, varying token count, so the touched working set
# grows while everything else is held fixed. The question is where the corpus
# outgrows the 64 MB L2 and natural order loses its streaming advantage.
# Run from the repository root (the .bin trie files load by relative path).
set -u
CORPUS=${CORPUS:-wiki_sample_50m.txt}
KEY=${KEY:-alpha}
# TAG names the output series; defaults to KEY. NS overrides the token grid.
TAG=${TAG:-$KEY}
NS=${NS:-"1000000 2000000 5000000 10000000 20000000 35000000 50000000"}
ROUNDS=${ROUNDS:-5}
REPS=${REPS:-3}
OUT=bench/alphasort/results
mkdir -p "$OUT"

for n in $NS; do
  echo "=== n=$n key=$KEY ==="
  ./bench/alphasort/bench_real --words "$CORPUS" --n "$n" --trie . \
      --reps "$REPS" --rounds "$ROUNDS" --key "$KEY" --nocpu 1 \
      > "$OUT/scale_${TAG}_${n}.txt" 2>&1
  grep -E "^(loaded|0 baseline|3 gpu-prefix|6b sort)" \
      "$OUT/scale_${TAG}_${n}.txt"
done
echo "done"
