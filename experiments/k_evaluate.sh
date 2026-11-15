#!/usr/bin/env bash
# Per arm: simulations of the trained pairs (scaling summaries), cross-play (focal seed s vs partner seed s+1 in
# every cell, plus the co-trained control) and probes (needs cross-play: memory sweep / swap).
#   GPU=0 ./experiments/k_evaluate.sh          (FORCE=1 to redo arms already evaluated)
# Logs: <data_root>/logs/k_evaluate_<source>_<time>.log
source "$(dirname "$0")/k_common.sh"
for src in $SOURCES; do
  cfg=$(kcfg "$src" experiment); tid=$(kcfg "$src" training_id)
  kcfg "$src" trained || { say "$src ($tid): training incomplete, run k_train.sh first"; exit 1; }
  marker="$DATA_ROOT/influence/$tid/.evaluate_done"
  if [[ -f $marker && $FORCE != 1 ]]; then say "$src: already evaluated, skipped"; continue; fi
  log="$LOGS/k_evaluate_${src}_$(stamp).log"
  say "$src: simulate + crossplay + probes -> $log"
  {
    $PY simulate.py --config "$cfg" --training_id "$tid"
    $PY crossplay.py --influence "$INFLUENCE" --exp "$src"
    $PY probe.py --influence "$INFLUENCE" --exp "$src"
  } > "$log" 2>&1 || { say "$src FAILED, see $log"; exit 1; }
  touch "$marker"
done
say "evaluation done"
