#!/usr/bin/env python3
"""Write tiny copies of the three payoff configs + an influence config for experiments/k_smoke.sh.

    python experiments/make_smoke_configs.py --out /tmp/coopcoins_k_smoke
Same code path as the real study; only data_root, maps, seeds, games and simulation counts are reduced.
"""
import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", default="3,2;4,4", help="grid,agents;grid,agents (coins = agents)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=100, help="games per map (100 = 2%% of 5000)")
    ap.add_argument("--num_sims", type=int, default=4)
    ap.add_argument("--pool_size", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    maps = [[int(x) for x in m.split(",")] for m in a.maps.split(";")]
    icfg = yaml.safe_load(open(ROOT / "configs" / "influence_k.yaml"))
    for key, spec in icfg["experiments"].items():
        cfg = yaml.safe_load(open(ROOT / spec["experiment"]))
        cfg = copy.deepcopy(cfg)
        cfg["data_root"] = str(out / "data")
        cfg["families"] = {"density": {"keep": "explicit", "maps": [[g, n, n] for g, n in maps]}}
        cfg["seeds"] = a.seeds
        cfg["train"]["num_episodes"] = a.episodes
        cfg["simulation"]["num_sims"] = a.num_sims
        path = out / "configs" / Path(spec["experiment"]).name
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        spec["experiment"] = str(path)
    icfg["crossplay"]["num_sims"] = a.num_sims
    icfg["probes"] = {"pool_sims": 1, "pool_size": a.pool_size}
    yaml.safe_dump(icfg, open(out / "influence_k.yaml", "w"), sort_keys=False)
    print(out / "influence_k.yaml")


if __name__ == "__main__":
    main()
