#!/usr/bin/env python3
"""Policy probes: what do the trained policies choose, independently of what the partner leaves on the board?

1. Synthetic probes. Two-coin decision states (own coin vs other team's coin) varying distances, opponent
   threats and opponent targets. Per policy we fit log P(own) - log P(other) on the factors
   (see coopcoins.influence.probe_design): the policy's revealed preference weights.
2. State-matched probes. Real decision states of the focal team collected in games against partner B
   ("world B"); every focal policy (trained with A) is evaluated on the SAME states, so differences between
   policies are purely strategic (the ecology is held fixed exactly).

    python probe.py --exp exp1
Outputs in <data_root>/influence/<training_id>/:
    probe_coefficients.csv   per (set, map, pair, team, seed): preference weights + mean P(stay)
    probe_states_mean.csv    per (set, map, pair, team, synthetic state): probabilities averaged over seeds
    probe_matched.csv        per (set, map, world B, focal policy trained with A, seed): P(own), P(other), P(stay)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coopcoins.influence import (PROBE_TERMS, choice_probs, combine, env_tools, load_team, partner_dirs,
                                 probe_policy, set_maps, sim_keys, state_pool)
from coopcoins.utils import _yaml, load_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", default="configs/influence.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sets", default=None)
    a = ap.parse_args()

    icfg = _yaml(a.influence)
    spec = icfg["experiments"][a.exp]
    cfg = load_experiment(spec["experiment"])
    tid = spec["training_id"]
    focal, partners = icfg["focal"], icfg["partners"]
    shift = icfg["crossplay"]["seed_shift"]
    pc = icfg["probes"]
    out = Path(cfg["data_root"]) / "influence" / tid
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"probe_{k}.csv" for k in ("coefficients", "states_mean", "matched")}
    for p in paths.values():
        p.unlink(missing_ok=True)
    rng = np.random.default_rng(0)
    keys = sim_keys(pc["pool_sims"])

    def append(name, df):
        df.to_csv(paths[name], mode="a", header=not paths[name].exists(), index=False)

    for tset in cfg["training_sets"]:
        if a.sets and tset["name"] not in a.sets.split(","):
            continue
        dirs = partner_dirs(cfg, tset["name"], focal, partners, tid)
        teams = {(A, k): load_team(d, k) for A, d in dirs.items() for k in (0, 1)}
        base = {"training_id": tid, "training_set": tset["name"],
                "training": "generalist" if tset["name"] == "all_maps" else "specialist"}
        for m in set_maps(dirs[partners[0]]):
            t0 = time.time()
            env, _, collect = env_tools(m)
            ctx = {**base, "map": m["name"], "grid_size": m["grid_size"], "num_agents": m["num_agents"],
                   "num_coins": m["num_coins"]}
            # 1. synthetic probes: every team policy of every pair
            for (A, k), (params, seeds, _) in teams.items():
                attitude, trained_with = (focal, A) if k == 0 else (A, focal)
                beta, stay, pr, factors = probe_policy(params, m["grid_size"], m["num_agents"])
                info = {**ctx, "pair": f"{focal}_{A}", "team": k, "attitude": attitude, "trained_with": trained_with}
                append("coefficients", pd.DataFrame({**info, "seed": seeds, "p_stay": stay,
                                                     **{t: beta[:, i] for i, t in enumerate(PROBE_TERMS)}}))
                append("states_mean", pd.DataFrame({**info, **factors, "p_own": pr[..., 0].mean(0),
                                                    "p_other": pr[..., 1].mean(0), "p_stay": pr[..., 2].mean(0)}))
            # 2. state-matched probes: focal policies on the states of world B
            for B in partners:
                fp, _, _ = teams[(B, 0)]
                pp, _, _ = teams[(B, 1)]
                coins, mask = state_pool(collect(combine(fp, pp, shift), keys), pc["pool_size"], rng)
                for A in partners:
                    params, seeds, _ = teams[(A, 0)]
                    p_own, p_other, p_stay = choice_probs(params, coins, mask)
                    share = p_other / np.maximum(p_own + p_other, 1e-9)
                    append("matched", pd.DataFrame({**ctx, "world": B, "trained_with": A, "seed": seeds,
                                                    "n_states": len(coins), "p_own": p_own.mean(1),
                                                    "p_other": p_other.mean(1), "p_stay": p_stay.mean(1),
                                                    "steal_share": share.mean(1)}))
            print(f"[{tset['name']} -> {m['name']}] probes in {time.time() - t0:.1f}s", flush=True)
    print(f"written {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
