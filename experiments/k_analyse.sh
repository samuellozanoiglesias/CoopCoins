#!/usr/bin/env bash
# Per arm: influence exp1 (decomposition, cells, co-adaptation, ties, best response, accounting check), exp2
# (probes, memory sweep / swap), exp4 (observational vs true), scaling summaries (analyze.py); then the cross-k
# analysis (analyze_k.py). No exp3.
#   ./experiments/k_analyse.sh
# Outputs: <data_root>/influence/analysis/<training_id>/exp{1,2,4}/, <data_root>/analysis/payoff_k<k>_<training_id>/,
#          <data_root>/influence/analysis/k_sweep/       Logs: <data_root>/logs/k_analyse_*.log
source "$(dirname "$0")/k_common.sh"
for src in $SOURCES; do
  cfg=$(kcfg "$src" experiment); tid=$(kcfg "$src" training_id)
  log="$LOGS/k_analyse_${src}_$(stamp).log"
  say "$src: per-arm analyses -> $log"
  {
    $PY analyze_influence.py --influence "$INFLUENCE" --source "$src" --exp 1
    $PY analyze_influence.py --influence "$INFLUENCE" --source "$src" --exp 2
    $PY analyze_influence.py --influence "$INFLUENCE" --source "$src" --exp 4
    $PY analyze.py --config "$cfg" --training_id "$tid"
  } > "$log" 2>&1 || { say "$src FAILED, see $log"; exit 1; }
done
log="$LOGS/k_analyse_cross_k_$(stamp).log"
say "cross-k analysis -> $log"
$PY analyze_k.py --influence "$INFLUENCE" --sources "${SOURCES// /,}" > "$log" 2>&1 || { say "FAILED, see $log"; exit 1; }
tail -n 20 "$log"
say "analysis done: $DATA_ROOT/influence/analysis/k_sweep/"
