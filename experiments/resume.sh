#!/usr/bin/env bash
# Finish scaling_v1 after the out-of-memory crash, using ONE GPU process (sharing the GPU was ~4-5x slower).
#
# Kept as they are (finished):   grid3_agents2 grid4_agents4 grid5_agents6 grid6_agents8 grid7_agents10 grid8_agents14
# Deleted and retrained (crashed at game 1000): all_maps grid9_agents18
# Trained (never started):       grid10_agents22 grid6_agents4 grid9_agents6
#
#   chmod +x resume.sh
#   nohup ./resume.sh > resume.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

export TRAINING_ID=scaling_v1
DATA=/data/samuel_lozano/CoopCoins
TODO=grid6_agents4,grid9_agents6,grid9_agents18,grid10_agents22,all_maps   # short ones first

# partial runs would get duplicated rows in training_stats.csv, so they start from scratch
for set in all_maps grid9_agents18; do
  for d in "$DATA"/$set/*/Training_$TRAINING_ID; do
    [[ -d "$d" ]] && { echo "removing partial run $d"; rm -rf "$d"; }
  done
done

echo "[$(date)] training $TODO"
STEPS=train ONLY=$TODO ./run_scaling.sh > /dev/null 2>&1 || { echo "[$(date)] training FAILED, see $DATA/logs/"; exit 1; }
echo "[$(date)] training done"

STEPS="simulate analyze" RENDER=1 ./run_scaling.sh > /dev/null 2>&1
echo "[$(date)] simulation and analysis done: $DATA/analysis/scaling_${TRAINING_ID}/"
