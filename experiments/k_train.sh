#!/usr/bin/env bash
# Train the theft-penalty arms (generalist, 3 pairs x 10 seeds each).
#   GPU=0 MAX_PARALLEL_RUNS=0 PARALLEL=0 ./experiments/k_train.sh
# PARALLEL=1 launches the arms as separate processes on the same GPU (else one after another).
# Relaunching skips completed runs (final checkpoint present) and restarts incomplete ones.
# Logs: <data_root>/logs/k_train_<source>_<time>.log
source "$(dirname "$0")/k_common.sh"
PARALLEL=${PARALLEL:-0}
pids=()
for src in $SOURCES; do
  cfg=$(kcfg "$src" experiment); tid=$(kcfg "$src" training_id)
  if kcfg "$src" trained; then say "$src ($tid): already trained, skipped"; continue; fi
  log="$LOGS/k_train_${src}_$(stamp).log"
  say "$src: training $tid ($cfg) -> $log"
  cmd=($PY train.py --config "$cfg" --training_id "$tid" --max_parallel_runs "$MAX_PARALLEL_RUNS")
  if [[ $PARALLEL == 1 ]]; then "${cmd[@]}" > "$log" 2>&1 & pids+=($!)
  else "${cmd[@]}" > "$log" 2>&1 || { say "$src FAILED, see $log"; exit 1; }; fi
done
fail=0
for p in "${pids[@]+"${pids[@]}"}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { say "a training FAILED (see $LOGS/k_train_*.log)"; exit 1; }
say "training done"
