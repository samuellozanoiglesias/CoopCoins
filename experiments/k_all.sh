#!/usr/bin/env bash
# Whole theft-penalty study: train -> evaluate -> analyse.
#   nohup ./experiments/k_all.sh > k_all.out 2>&1 &
#   ./experiments/k_all.sh --from evaluate            # resume at a stage
#   ./experiments/k_all.sh --from train --to train    # one stage
# Environment variables: see experiments/k_common.sh (GPU, MAX_PARALLEL_RUNS, PARALLEL, SOURCES, INFLUENCE, FORCE).
set -euo pipefail
here="$(dirname "$0")"
stages=(train evaluate analyse)
from=train; to=analyse
while [[ $# -gt 0 ]]; do
  case $1 in
    --from) from=$2; shift 2 ;;
    --to) to=$2; shift 2 ;;
    *) echo "usage: $0 [--from train|evaluate|analyse] [--to train|evaluate|analyse]"; exit 2 ;;
  esac
done
idx() { for i in "${!stages[@]}"; do [[ ${stages[$i]} == "$1" ]] && { echo "$i"; return; }; done; echo "bad stage $1" >&2; exit 2; }
a=$(idx "$from"); b=$(idx "$to")
for ((i = a; i <= b; i++)); do
  "$here/k_${stages[$i]}.sh"
done
