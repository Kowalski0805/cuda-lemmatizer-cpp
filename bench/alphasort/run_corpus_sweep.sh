#!/usr/bin/env bash
# Corpus sweep: token count held fixed, corpus varied, so the only thing that
# moves is the input distribution (type/token ratio, Zipf skew, hit rate).
# The scale sweep answers "does the result hold as the corpus grows?"; this one
# answers "does it hold on a different corpus?" — the generality objection.
#
# Two common points: 750 K, which every corpus can supply (articles has only
# 762 K tokens), and 18 M, which fiction and wiki can both supply.
set -u
ROUNDS=${ROUNDS:-7}
REPS=${REPS:-3}
KEY=${KEY:-alpha}
OUT=bench/alphasort/results
mkdir -p "$OUT"

run() {  # corpus tag n
  echo "=== $2 n=$3 ==="
  ./bench/alphasort/bench_real --words "$1" --n "$3" --trie . \
      --reps "$REPS" --rounds "$ROUNDS" --key "$KEY" --nocpu 1 \
      > "$OUT/corpus_${2}_${3}.txt" 2>&1
  grep -E "^(loaded|hit rate|0 baseline|3 gpu-prefix|4 gpu-part|6 sort|6b sort)" \
      "$OUT/corpus_${2}_${3}.txt"
}

run articles_pp.txt      articles   750000
run fiction_pp.txt       fiction    750000
run wiki_sample_50m.txt  wiki       750000
run fiction_pp.txt       fiction   5000000
run wiki_sample_50m.txt  wiki      5000000
run fiction_pp.txt       fiction  18000000
run wiki_sample_50m.txt  wiki     18000000
echo "done"
