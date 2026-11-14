#!/usr/bin/env bash
# Scaling experiment: constant density (grid 3..10) and constant grid/agents ratio (grid 3, 6, 9);
# pairs ind-ind, ind-coop, ind-comp; 10 training seeds; 20 simulated games per trained pair; analysis.
#
#   ./run_scaling.sh                                  # everything, sequentially
#   nohup ./run_scaling.sh > /dev/null 2>&1 &
#   ONLY=all_maps ./run_scaling.sh                    # just the generalist
#   TRAINING_ID=<id> STEPS="simulate analyze" ./run_scaling.sh   # redo later steps of an existing run
#
# On several GPUs / CPU nodes, run the training sets in parallel with the same TRAINING_ID, e.g.
#   TRAINING_ID=run1 ONLY=all_maps STEPS=train ./run_scaling.sh &
#   TRAINING_ID=run1 ONLY=grid3_agents2,grid4_agents4,grid5_agents6 STEPS=train ./run_scaling.sh &
#   ...then: TRAINING_ID=run1 STEPS="simulate analyze" ./run_scaling.sh
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${CONFIG:-configs/experiments/scaling.yaml}
SEEDS=${SEEDS:-10}
NUM_SIMS=${NUM_SIMS:-20}
TRAINING_ID=${TRAINING_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
STEPS=${STEPS:-"train simulate analyze"}
ONLY_ARG=()
[[ -n "${ONLY:-}" ]] && ONLY_ARG=(--only "$ONLY")
RENDER_ARG=()
[[ "${RENDER:-0}" == "1" ]] && RENDER_ARG=(--render)

DATA_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_root'])")
mkdir -p "$DATA_ROOT/logs"
LOG="$DATA_ROOT/logs/$(basename "$CONFIG" .yaml)_${TRAINING_ID}${ONLY:+_$ONLY}.log"
echo "training_id=$TRAINING_ID  steps=[$STEPS]  log=$LOG"

{
  for step in $STEPS; do
    echo "### $step  $(date)"
    case $step in
      train)    python train.py    --config "$CONFIG" --training_id "$TRAINING_ID" --seeds "$SEEDS" "${ONLY_ARG[@]}" ;;
      simulate) python simulate.py --config "$CONFIG" --training_id "$TRAINING_ID" --num_sims "$NUM_SIMS" \
                                   "${ONLY_ARG[@]}" "${RENDER_ARG[@]}" ;;
      analyze)  python analyze.py  --config "$CONFIG" --training_id "$TRAINING_ID" ;;
    esac
  done
  echo "### done  $(date)"
} 2>&1 | tee "$LOG"
