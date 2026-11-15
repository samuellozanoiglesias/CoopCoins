#!/usr/bin/env bash
# Before the full run: unit tests, then the WHOLE pipeline (train -> evaluate -> analyse, cross-k included) on a
# tiny copy of the three arms (2 maps, 2 seeds, 2% of the games) in $SMOKE_DIR, then a timing benchmark of one
# training update per arm with the FULL configs and the estimated wall time.
#   GPU=0 ./scripts/k_smoke.sh             BENCH=0 skips the benchmark; MAX_PARALLEL_RUNS is used by it
# The benchmark searches the largest number of vmapped runs that fits in GPU memory (unless MAX_PARALLEL_RUNS is set)
# and prints RECOMMENDED MAX_PARALLEL_RUNS=... per arm: use it for k_train.sh / k_all.sh.
# Must end with "K SMOKE TEST PASSED".
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SMOKE_DIR=${SMOKE_DIR:-/tmp/coopcoins_k_smoke}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "== unit tests"
$PY -m pytest -q tests
echo "== tiny pipeline in $SMOKE_DIR"
rm -rf "$SMOKE_DIR"
export INFLUENCE=$($PY scripts/make_smoke_configs.py --out "$SMOKE_DIR")
FORCE=1 ./scripts/k_all.sh
out="$SMOKE_DIR/data/influence/analysis"
for f in k_sweep/k_decomposition.csv k_sweep/k_trend_tests_S.csv k_sweep/k_teacher.csv \
         k_sweep/k_training_diagnostics.csv; do
  [[ -s "$out/$f" ]] || { echo "missing $out/$f"; exit 1; }
done
for src in k0 k2 k10; do
  tid=$($PY scripts/kcfg.py "$INFLUENCE" "$src" training_id)
  for f in exp1/decomposition.csv exp1/accounting_check.csv exp1/best_response.csv exp1/tie_stats.csv \
           exp2/probe_memory_sweep.csv exp2/memory_swap_effects.csv exp4/observed_vs_true.csv; do
    [[ -s "$out/$tid/$f" ]] || { echo "missing $out/$tid/$f"; exit 1; }
  done
done
if [[ ${BENCH:-1} == 1 ]]; then
  echo "== benchmark (full configs, one update per arm)"
  unset INFLUENCE
  for k in 0 2 10; do
    if [[ -n ${MAX_PARALLEL_RUNS:-} ]]; then
      $PY scripts/benchmark.py --config "configs/experiments/payoff_k$k.yaml" --max_parallel_runs "$MAX_PARALLEL_RUNS"
    else
      $PY scripts/benchmark.py --config "configs/experiments/payoff_k$k.yaml" --auto
    fi
  done
fi
echo "K SMOKE TEST PASSED"
