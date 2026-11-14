#!/usr/bin/env python3
"""Cross-play for the influence decomposition: focal policies trained with partner A play partner policies of
attitude B, for every A, B in `partners`, on every map of every training set.

    python crossplay.py --exp exp1          # trainings of configs/influence.yaml -> experiments.exp1
    python crossplay.py --exp exp3 --sets grid6_agents8_coins4

Output: <data_root>/influence/<training_id>/crossplay_games.csv, one row per
(training set, map, train_partner A, play_partner B, pairing, focal seed, sim seed, team) with per-agent game totals.
pairing = cross (focal seed s vs partner seed s + shift; used for the decomposition)
        | co    (the original co-trained pair, same run; control for co-adaptation, only when A == B)
"""
import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from coopcoins.influence import combine, env_tools, load_team, partner_dirs, set_maps, sim_keys
from coopcoins.simulation import SIM_STATS
from coopcoins.utils import _yaml, load_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", default="configs/influence.yaml")
    ap.add_argument("--exp", required=True, help="key under experiments: in the influence config (exp1, exp3)")
    ap.add_argument("--sets", default=None, help="comma list of training sets (default: all of the experiment)")
    ap.add_argument("--num_sims", type=int, default=None)
    a = ap.parse_args()

    icfg = _yaml(a.influence)
    spec = icfg["experiments"][a.exp]
    cfg = load_experiment(spec["experiment"])
    tid = spec["training_id"]
    focal, partners = icfg["focal"], icfg["partners"]
    shift = icfg["crossplay"]["seed_shift"]
    num_sims = a.num_sims or icfg["crossplay"]["num_sims"]
    keys = sim_keys(num_sims)
    out = Path(cfg["data_root"]) / "influence" / tid
    out.mkdir(parents=True, exist_ok=True)
    path = out / "crossplay_games.csv"
    path.unlink(missing_ok=True)

    for tset in cfg["training_sets"]:
        if a.sets and tset["name"] not in a.sets.split(","):
            continue
        dirs = partner_dirs(cfg, tset["name"], focal, partners, tid)
        focal_p = {A: load_team(d, 0) for A, d in dirs.items()}      # policies of the focal team, by training partner
        partner_p = {B: load_team(d, 1) for B, d in dirs.items()}    # partner policies, by attitude
        for m in set_maps(dirs[partners[0]]):
            env, sim, _ = env_tools(m)
            t0 = time.time()
            frames = []
            cells = [(A, B, "cross") for A in partners for B in partners] + [(A, A, "co") for A in partners]
            for A, B, pairing in cells:
                fp, seeds, _ = focal_p[A]
                pp, pseeds, _ = partner_p[B]
                s = shift if pairing == "cross" else 0
                coef = jnp.array([cfg["attitudes"][focal], cfg["attitudes"][B]])
                res = np.asarray(sim(combine(fp, pp, s), coef, keys))          # (S, sims, stats, N)
                S = res.shape[0]
                for k in (0, 1):
                    frames.append(pd.DataFrame({
                        "training_id": tid, "training_set": tset["name"],
                        "training": "generalist" if tset["name"] == "all_maps" else "specialist",
                        "map": m["name"], "grid_size": m["grid_size"], "num_agents": m["num_agents"],
                        "num_coins": m["num_coins"], "episode_len": m["episode_len"],
                        "train_partner": A, "play_partner": B, "pairing": pairing,
                        "focal_seed": np.repeat(seeds, num_sims),
                        "partner_seed": np.repeat(np.roll(pseeds, -s), num_sims),
                        "sim_seed": np.tile(np.arange(num_sims), S), "team": k,
                        **{st: res[:, :, j][..., env.agent_team == k].mean(-1).ravel()
                           for j, st in enumerate(SIM_STATS)}}))
            pd.concat(frames).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"[{tset['name']} -> {m['name']}] {len(cells)} cells x {S} seeds x {num_sims} games "
                  f"in {time.time() - t0:.1f}s", flush=True)
    print(f"written {path}")


if __name__ == "__main__":
    main()
