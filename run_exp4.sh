#!/usr/bin/env bash
# Experiment 4: what would an observational study conclude? Uses the cross-play data of experiments 1 and 3
# (run those first; experiment 3 is optional). Analysis only, a few minutes.
#   ./run_exp4.sh
source "$(dirname "$0")/influence_common.sh"
LOG="$DATA_ROOT/logs/influence_exp4_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "log: $LOG"
{
  echo "### exp4 analysis  $(date)"
  python analyze_influence.py --influence "$INFLUENCE" --exp 4
  echo "### done           $(date)"
} 2>&1 | tee "$LOG"
