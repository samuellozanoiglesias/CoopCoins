#!/usr/bin/env bash
# Experiment 1: strategic vs ecological influence, decomposed with cross-play of the EXISTING scaling_v1 agents.
# No training. Every map of every training set (generalist + specialists), 3 x 3 cells + co-trained controls.
#   nohup ./run_exp1.sh > /dev/null 2>&1 &
source "$(dirname "$0")/influence_common.sh"
LOG="$DATA_ROOT/logs/influence_exp1_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "log: $LOG"
{
  echo "### exp1 cross-play  $(date)"
  python crossplay.py --influence "$INFLUENCE" --exp exp1
  echo "### exp1 analysis    $(date)"
  python analyze_influence.py --influence "$INFLUENCE" --exp 1
  echo "### done             $(date)"
} 2>&1 | tee "$LOG"
