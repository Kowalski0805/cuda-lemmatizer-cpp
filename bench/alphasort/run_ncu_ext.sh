#!/usr/bin/env bash
# Extended profiler sweep: the same seven variants at every scale the timing
# sweep covers, so the cost model can be fitted against 49 observations rather
# than 21. Two metrics are new and change what can be tested:
#
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
#       Transactions per token measured DIRECTLY, instead of inferred from the
#       DRAM-traffic identity. Removes the inference from the model's nu.
#   smsp__thread_inst_executed_per_inst_executed.ratio
#       Threads per executed instruction. Divided by 32 this is e of Eq. (7)
#       itself, not the branch-uniformity proxy standing in for it.
#
# Three more support a memory-level-parallelism term, which is the factor
# Eq. (3) is missing: occupancy, the long-scoreboard stall ratio, and the
# request count that turns sectors into sectors-per-request.
#
# Run from the repository root. Nothing else may use the GPU meanwhile.
set -u
NCU=${NCU:-ncu}
CORPUS=${CORPUS:-wiki_sample_50m.txt}
NS=${NS:-"1000000 2000000 5000000 10000000 20000000 35000000 50000000"}
OUT=bench/alphasort/results
mkdir -p "$OUT"

METRICS=\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio

for n in $NS; do
  echo "=== ncu n=$n ==="
  # --warm 0 --rounds 1 --reps 1 makes the first seven launches the correctness
  # pass, one per variant in table order; --nocpu 1 drops A1's host sort.
  "$NCU" --kernel-name regex:lookup --launch-count 7 --csv --metrics "$METRICS" \
      ./bench/alphasort/bench_real --words "$CORPUS" --n "$n" --trie . \
      --reps 1 --rounds 1 --warm 0 --nocpu 1 \
      2> "$OUT/ncu_ext_${n}.err" \
    | grep -E '^"(ID|[0-9])' > "$OUT/ncu_ext_${n}.csv"
  echo "  rows: $(($(wc -l < "$OUT/ncu_ext_${n}.csv") - 1))  (expect 70)"
done
echo "done"
