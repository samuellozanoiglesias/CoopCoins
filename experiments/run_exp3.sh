#!/usr/bin/env bash
# Experiment 3: influence vs scarcity (fewer coins than agents) and group size.
# Trains the scarcity maps (configs/experiments/scarcity.yaml, ~3.5 h on the GPU), then cross-play + probes,
# then the analysis (which also reuses the coins = agents specialists of scaling_v1 from experiment 1).
#   nohup ./run_exp3.sh > /dev/null 2>&1 &
#   STEPS="crossplay probe analyze" ./run_exp3.sh      # skip training (already done)
source "$(dirname "$0")/influence_common.sh"
STEPS=${STEPS:-"train crossplay probe analyze"}
EXP=$(cfg_get exp3 experiment)
TID=$(cfg_get exp3 training_id)
SEEDS=${SEEDS:-10}
LOG="$DATA_ROOT/logs/influence_exp3_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "log: $LOG"
{
  for step in $STEPS; do
    echo "### exp3 $step  $(date)"
    case $step in
      train)     python train.py --config "$EXP" --training_id "$TID" --seeds "$SEEDS" ;;
      crossplay) python crossplay.py --influence "$INFLUENCE" --exp exp3 ;;
      probe)     python probe.py --influence "$INFLUENCE" --exp exp3 ;;
      analyze)   python analyze_influence.py --influence "$INFLUENCE" --exp 3 ;;
    esac
  done
  echo "### done  $(date)"
} 2>&1 | tee "$LOG"
