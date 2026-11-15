#!/usr/bin/env python3
"""Policy probes: what do the trained policies choose, independently of what the partner leaves on the board?

1. Synthetic probes. Two-coin decision states (own coin vs other team's coin) varying distances, opponent
   threats and opponent targets. Per policy we fit log P(own) - log P(other) on the factors
   (see coopcoins.influence.probe_design): the policy's revealed preference weights.
   Baseline: memory features = 0 (every team policy). Memory sweep (focal team): the same states with the memory
   features set to the focal team's mean memory in world B (cell trained with B, plays B, cross pairing, same map,
   from crossplay_games.csv), for every partner B.
   Pseudo-replication: with memory = 0 the synthetic states depend only on the map's maximum torus distance and on
   whether the team has teammates (coopcoins.influence.probe_state_set). A generalist therefore gives identical
   baseline rows on such maps; all but the first map of each state set are flagged pseudo_replicate = True.
2. State-matched probes. Real decision states of the focal team collected in games against partner B
   ("world B"); every focal policy (trained with A) is evaluated on the SAME states, so differences between
   policies are purely strategic (the ecology is held fixed exactly).
   Memory swap: on the same real states, the memory features are replaced by the mean memory of world V (every
   partner V, plus "actual" = unchanged): the in-game strategic channel of a fixed policy.

    python probe.py --exp exp1
    python probe.py --influence configs/influence_k.yaml --exp k2      (needs crossplay.py --exp k2 first)
Outputs in <data_root>/influence/<training_id>/ (every file has payoff_own, payoff_other, payoff_penalty, k):
    probe_coefficients.csv   per (set, map, pair, team, memory_source, seed): preference weights + mean P(stay)
    probe_states_mean.csv    per (set, map, pair, team, synthetic state): baseline probabilities averaged over seeds
    probe_matched.csv        per (set, map, world B, focal policy trained with A, seed): P(own), P(other), P(stay)
    probe_memory_swap.csv    per (set, map, world W, trained with A, memory_source V, seed): same quantities
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coopcoins.env import MEMORY, N_BASE
from coopcoins.influence import (PROBE_TERMS, choice_probs, combine, env_tools, load_team, partner_dirs,
                                 probe_policy, probe_state_set, set_maps, sim_keys, state_pool, with_memory,
                                 world_memory)
from coopcoins.utils import _yaml, load_experiment, payoff_columns


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
    cp_path = out / "crossplay_games.csv"
    if not cp_path.exists():
        raise FileNotFoundError(f"{cp_path}: the memory sweep / swap need cross-play first (crossplay.py)")
    crossplay = pd.read_csv(cp_path)
    paths = {k: out / f"probe_{k}.csv" for k in ("coefficients", "states_mean", "matched", "memory_swap")}
    for p in paths.values():
        p.unlink(missing_ok=True)
    rng = np.random.default_rng(0)
    keys = sim_keys(pc["pool_sims"])
    pay = payoff_columns(cfg["payoff"])

    def append(name, df):
        df.to_csv(paths[name], mode="a", header=not paths[name].exists(), index=False)

    def mem_cols(mem):
        return {f: float(v) for f, v in zip(MEMORY, mem)}

    for tset in cfg["training_sets"]:
        if a.sets and tset["name"] not in a.sets.split(","):
            continue
        dirs = partner_dirs(cfg, tset["name"], focal, partners, tid)
        teams = {(A, k): load_team(d, k) for A, d in dirs.items() for k in (0, 1)}
        base = {"training_id": tid, "training_set": tset["name"],
                "training": "generalist" if tset["name"] == "all_maps" else "specialist", **pay}
        seen_sets = set()
        for m in sorted(set_maps(dirs[partners[0]]), key=lambda x: (x["grid_size"], x["num_agents"])):
            t0 = time.time()
            env, _, collect = env_tools(m)
            pset = probe_state_set(m["grid_size"], m["num_agents"])
            ctx = {**base, "map": m["name"], "grid_size": m["grid_size"], "num_agents": m["num_agents"],
                   "num_coins": m["num_coins"]}
            pseudo = pset in seen_sets
            seen_sets.add(pset)
            wmem = world_memory(crossplay, tset["name"], m["name"], partners)
            # 1. synthetic probes: baseline (memory 0) for every team policy; memory sweep for the focal team
            for (A, k), (params, seeds, _) in teams.items():
                attitude, trained_with = (focal, A) if k == 0 else (A, focal)
                info = {**ctx, "pair": f"{focal}_{A}", "team": k, "attitude": attitude, "trained_with": trained_with}
                sources = [("zero", np.zeros(len(MEMORY), np.float32))] + \
                          ([(B, wmem[B]) for B in partners] if k == 0 else [])
                for src, mem in sources:
                    beta, stay, pr, factors = probe_policy(params, m["grid_size"], m["num_agents"], mem)
                    append("coefficients", pd.DataFrame({
                        **info, "memory_source": src, **mem_cols(mem), "probe_state_set": pset,
                        "pseudo_replicate": bool(pseudo and src == "zero"), "seed": seeds, "p_stay": stay,
                        **{t: beta[:, i] for i, t in enumerate(PROBE_TERMS)}}))
                    if src == "zero":
                        append("states_mean", pd.DataFrame({**info, "memory_source": src, "probe_state_set": pset,
                                                            "pseudo_replicate": pseudo, **factors,
                                                            "p_own": pr[..., 0].mean(0), "p_other": pr[..., 1].mean(0),
                                                            "p_stay": pr[..., 2].mean(0)}))
            # 2. state-matched probes: focal policies on the states of world B, plus the memory swap
            for B in partners:
                fp, _, _ = teams[(B, 0)]
                pp, _, _ = teams[(B, 1)]
                coins, mask = state_pool(collect(combine(fp, pp, shift), keys), pc["pool_size"], rng)
                actual = coins[:, 0, N_BASE:].mean(0)
                for A in partners:
                    params, seeds, _ = teams[(A, 0)]
                    for src in ["actual"] + list(partners):
                        c = coins if src == "actual" else with_memory(coins, mask, wmem[src])
                        p_own, p_other, p_stay = choice_probs(params, c, mask)
                        share = p_other / np.maximum(p_own + p_other, 1e-9)
                        row = {**ctx, "world": B, "trained_with": A, "seed": seeds, "n_states": len(coins),
                               "p_own": p_own.mean(1), "p_other": p_other.mean(1), "p_stay": p_stay.mean(1),
                               "steal_share": share.mean(1)}
                        if src == "actual":
                            append("matched", pd.DataFrame({**row, **mem_cols(actual)}))
                        append("memory_swap", pd.DataFrame({
                            **row, "memory_source": src, **mem_cols(actual if src == "actual" else wmem[src])}))
            print(f"[{tset['name']} -> {m['name']}] probes in {time.time() - t0:.1f}s"
                  f"{' (baseline synthetic states duplicate an earlier map: ' + pset + ')' if pseudo else ''}",
                  flush=True)
    print(f"written {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
