#!/usr/bin/env bash
# Speed test on GPU and CPU with the FULL experiment settings (a few updates each, then extrapolated).
#   ./speed_test.sh                 -> GPU (all sets) then CPU (small sets), ~20-40 min
#   UPDATES=10 ./speed_test.sh      -> faster, a bit noisier
set -euo pipefail
cd "$(dirname "$0")"
UPDATES=${UPDATES:-20}
OUT=speed_test_$(date +%Y-%m-%d_%H-%M-%S).log
{
  echo "### machine"; nproc; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "### GPU, all training sets"
  python speed_test.py --updates "$UPDATES"
  echo "### CPU (single process), small training sets"
  JAX_PLATFORMS=cpu python speed_test.py --updates "$UPDATES" \
      --sets grid3_agents2,grid4_agents4,grid5_agents6,grid6_agents4,grid6_agents8,grid9_agents6
  echo "### CPU, 6 processes at once (contention check): grid3_agents2 each"
  for i in 1 2 3 4 5 6; do
    JAX_PLATFORMS=cpu python speed_test.py --updates "$UPDATES" --sets grid3_agents2 > /tmp/cpu_par_$i.log 2>&1 &
  done
  wait
  grep -h "training_set=" /tmp/cpu_par_*.log
} 2>&1 | tee "$OUT"
echo "log: $(pwd)/$OUT"
