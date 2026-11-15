#!/usr/bin/env bash
# Experiment 2: policy probes of the EXISTING scaling_v1 agents (synthetic + state-matched). No training.
# Run after run_exp1.sh to also get the policy-vs-behaviour comparison.
#   nohup ./run_exp2.sh > /dev/null 2>&1 &
source "$(dirname "$0")/influence_common.sh"
LOG="$DATA_ROOT/logs/influence_exp2_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "log: $LOG"
{
  echo "### exp2 probes      $(date)"
  python probe.py --influence "$INFLUENCE" --exp exp1
  echo "### exp2 analysis    $(date)"
  python analyze_influence.py --influence "$INFLUENCE" --exp 2
  echo "### done             $(date)"
} 2>&1 | tee "$LOG"
