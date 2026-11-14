#!/usr/bin/env bash
# Smoke test: whole pipeline on tiny settings (~5-10 min). Writes to /tmp/coopcoins_smoke, not to your data.
#   ./smoke_test.sh            -> must end with "SMOKE TEST PASSED"
set -euo pipefail
cd "$(dirname "$0")"
OUT=/tmp/coopcoins_smoke
rm -rf "$OUT"
mkdir -p "$OUT"

python - <<PY
import yaml
c = yaml.safe_load(open("configs/experiments/scaling.yaml"))
c["data_root"] = "$OUT"
c["families"]["density"]["grid_sizes"] = [3, 4]   # grid3_agents2, grid4_agents4
c["families"]["ratio"]["grid_sizes"] = [3, 6]     # grid3_agents2, grid6_agents4
c["train"].update(num_episodes=20, num_checkpoints=2, log_every=2)
c["simulation"]["num_gifs"] = 1
yaml.safe_dump(c, open("$OUT/smoke.yaml", "w"), sort_keys=False)
PY

echo "=== devices"; python -c "import jax; print(jax.devices())"
CONFIG=$OUT/smoke.yaml SEEDS=2 NUM_SIMS=3 TRAINING_ID=smoke RENDER=1 ./run_scaling.sh

echo "=== generalist on a map it never saw (grid5_agents6)"
python simulate.py --dirs $OUT/all_maps/individualistic_competitive/Training_smoke --maps grid5_agents6 --num_sims 3

echo "=== checking outputs"
python - <<PY
from pathlib import Path
root, bad = Path("$OUT"), []
sets = ["all_maps", "grid3_agents2", "grid4_agents4", "grid6_agents4"]
pairs = ["individualistic_individualistic", "individualistic_cooperative", "individualistic_competitive"]
for s in sets:
    for p in pairs:
        d = root / s / p / "Training_smoke"
        for f in ["config.yaml", "training_stats.csv", "training_curves.png", "summary.yaml",
                  "checkpoints/seed_0/episode_0.pkl", "checkpoints/seed_1/episode_20.pkl"]:
            if not (d / f).exists(): bad.append(str(d / f))
        maps = ["grid3_agents2", "grid4_agents4", "grid6_agents4"] if s == "all_maps" else [s]
        for m in maps:
            sd = d / "simulations" / m / "episode_20_stochastic"
            for f in ["games.csv", "summary.yaml", "report.png", "gifs/train_seed_0_sim_seed_0.gif"]:
                if not (sd / f).exists(): bad.append(str(sd / f))
if not (root / "all_maps/individualistic_competitive/Training_smoke/simulations/grid5_agents6").exists():
    bad.append("unseen-map simulation")
a = root / "analysis" / "smoke_smoke"
for f in ["behaviour_summary.csv", "contrasts.csv", "generalization_gap.csv", "scaling_density_generalist.png",
          "influence_vs_size_density.png", "generalization_gap.png"]:
    if not (a / f).exists(): bad.append(str(a / f))
print("\n".join("MISSING " + b for b in bad) if bad else "SMOKE TEST PASSED  (outputs in $OUT)")
PY
