#!/usr/bin/env bash
# Full scaling experiment on one GPU:
#   - generalist (all_maps) and the 8 specialists train at the same time, sharing the GPU
#   NOTE: run_scaling.sh is not part of this repository snapshot; for the theft-penalty study use scripts/k_*.sh
#   - when both finish: simulate (20 games per trained pair, with GIFs) and analyze
#
#   chmod +x launch.sh
#   nohup ./launch.sh > launch.out 2>&1 &
#
# Logs:     /data/samuel_lozano/CoopCoins/logs/scaling_<TRAINING_ID>*.log
# Analysis: /data/samuel_lozano/CoopCoins/analysis/scaling_<TRAINING_ID>/
set -euo pipefail
cd "$(dirname "$0")"

export TRAINING_ID=${TRAINING_ID:-scaling_v1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # lets two JAX processes share the GPU

SPECIALISTS=grid3_agents2,grid4_agents4,grid5_agents6,grid6_agents8,grid7_agents10,grid8_agents14,grid9_agents18,grid10_agents22

echo "[$(date)] training_id=$TRAINING_ID: starting generalist and specialists"
STEPS=train ONLY=all_maps     ./run_scaling.sh > /dev/null 2>&1 & PID_GEN=$!
STEPS=train ONLY=$SPECIALISTS ./run_scaling.sh > /dev/null 2>&1 & PID_SPEC=$!

FAIL=0
wait $PID_GEN  || { echo "[$(date)] generalist training FAILED (see its log)"; FAIL=1; }
wait $PID_SPEC || { echo "[$(date)] specialist training FAILED (see its log)"; FAIL=1; }
[[ $FAIL == 0 ]] || exit 1
echo "[$(date)] training done"

STEPS="simulate analyze" RENDER=1 ./run_scaling.sh > /dev/null 2>&1
echo "[$(date)] simulation and analysis done: /data/samuel_lozano/CoopCoins/analysis/scaling_${TRAINING_ID}/"
