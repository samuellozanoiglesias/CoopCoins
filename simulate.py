#!/usr/bin/env python3
"""Play games with trained team policies: every training seed x `num_sims` games, on one or more maps.
Game length = the experiment's episode_len (same as training). Team policies work on any map.

    python simulate.py --config configs/experiments/scaling.yaml --training_id <id> --num_sims 20
    python simulate.py --dirs <data_root>/all_maps/*/Training_<id> --maps grid12_agents32   # any map, even unseen
    python simulate.py ... --render            # also write GIFs (off by default)

Writes <training_dir>/simulations/<map>/episode_<n>_<stochastic|greedy>/
    games.csv     one row per (train_seed, sim_seed, team): per-agent totals of the game
    summary.yaml  team means and std over training seeds
    report.png    team comparison, one dot per training seed
    gifs/         only with --render
"""
import argparse
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from coopcoins.env import make_env
from coopcoins.plots import simulation_plot
from coopcoins.simulation import SIM_STATS, make_recorder, make_simulator
from coopcoins.utils import (load_checkpoint, load_experiment, load_yaml, map_from_name, save_yaml, stack_teams,
                             training_dir)


_SIMS = {}


def get_simulator(m, deterministic):
    """Compile once per (map, policy mode) and reuse across training directories."""
    key = (m["name"], m["episode_len"], str(m["payoff"]), deterministic)
    if key not in _SIMS:
        env = make_env(**m)
        _SIMS[key] = (env, make_simulator(env, deterministic))
    return _SIMS[key]


def available_episodes(d):
    per_seed = [{int(re.findall(r"\d+", p.stem)[0]) for p in s.glob("episode_*.pkl")}
                for s in (d / "checkpoints").glob("seed_*")]
    return sorted(set.intersection(*per_seed))


def eval_maps(cfg, spec):
    if spec == "trained":
        return cfg["maps"]
    if spec == "all":
        return cfg["experiment_maps"]
    names = spec.split(",") if isinstance(spec, str) else spec
    base = {"episode_len": cfg["maps"][0]["episode_len"], "payoff": cfg["maps"][0]["payoff"]}
    return [map_from_name(n, base) for n in names]


def simulate_dir(d, maps=None, checkpoints=None, num_sims=None, deterministic=None, render=False, num_gifs=None):
    d = Path(d)
    cfg = load_yaml(d / "config.yaml")
    sc = cfg["simulation"]
    maps = eval_maps(cfg, maps or sc.get("maps", "trained"))
    checkpoints = checkpoints or sc.get("checkpoints", "last")
    num_sims = num_sims or sc.get("num_sims", 20)
    deterministic = sc.get("deterministic", False) if deterministic is None else deterministic
    num_gifs = sc.get("num_gifs", 2) if num_gifs is None else num_gifs
    mode = "greedy" if deterministic else "stochastic"

    episodes = available_episodes(d)
    episodes = episodes[-1:] if checkpoints == "last" else episodes if checkpoints == "all" else \
        [int(e) for e in str(checkpoints).split(",")]
    seed_dirs = sorted((d / "checkpoints").glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    coef = jnp.array(cfg["reward_coef"])
    pair = cfg["pair"]
    sim_seeds = np.arange(num_sims)
    keys = jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), 7919))(jnp.asarray(sim_seeds))

    for ep in episodes:
        ckpts = [load_checkpoint(s / f"episode_{ep}.pkl") for s in seed_dirs]
        train_seeds = [c["seed"] for c in ckpts]
        params = jax.tree.map(lambda *x: jnp.stack(x), *[stack_teams(c) for c in ckpts])
        for m in maps:
            env, sim = get_simulator(m, deterministic)
            t0 = time.time()
            res = np.asarray(sim(params, coef, keys))  # (R, S, stats, N)
            R, S = res.shape[:2]
            out = d / "simulations" / m["name"] / f"episode_{ep}_{mode}"
            out.mkdir(parents=True, exist_ok=True)
            games = pd.concat([pd.DataFrame({
                "map": m["name"], "grid_size": env.grid_size, "num_agents": env.num_agents,
                "train_seed": np.repeat(train_seeds, S), "sim_seed": np.tile(sim_seeds, R), "team": k,
                "attitude": pair[k], "partner": pair[1 - k],
                **{st: res[:, :, j][..., env.agent_team == k].mean(-1).ravel() for j, st in enumerate(SIM_STATS)}})
                for k in (0, 1)])
            games.to_csv(out / "games.csv", index=False)
            per_seed = games.groupby(["team", "train_seed"])[SIM_STATS].mean()
            summary = {"pair": pair, "map": m["name"], "episode": ep, "trained_on": [x["name"] for x in cfg["maps"]],
                       "games_per_training_seed": int(S), "game_len": env.episode_len, "policy": mode,
                       "values": "per agent per game, averaged over the team's members"}
            for k in (0, 1):
                summary[f"team{k}_{pair[k]}"] = {"mean": per_seed.loc[k].mean().round(3).to_dict(),
                                                  "std_over_training_seeds": per_seed.loc[k].std().round(3).to_dict()}
            save_yaml(summary, out / "summary.yaml")
            simulation_plot(games, out / "report.png", pair, m["name"])
            if render:
                from coopcoins.render import render_gif
                (out / "gifs").mkdir(exist_ok=True)
                rec = make_recorder(env, deterministic)
                for i in range(min(num_gifs, R)):
                    s0, traj = rec(jax.tree.map(lambda x: x[i], params), coef, keys[0])
                    render_gif(s0, traj, env, out / "gifs" / f"train_seed_{train_seeds[i]}_sim_seed_0.gif",
                               title=f"{m['name']} | {pair[0]} (red) vs {pair[1]} (blue) | seed {train_seeds[i]}")
            mm = per_seed.groupby(level=0).mean()
            print(f"[{d.parent.parent.name}/{d.parent.name} -> {m['name']} ep {ep}] {R}x{S} games, "
                  f"{time.time() - t0:.1f}s | per agent: other coins red/blue {mm.other_coins[0]:.1f}/"
                  f"{mm.other_coins[1]:.1f}, score {mm.pure_reward[0]:.1f}/{mm.pure_reward[1]:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="simulate every training set and pair of this experiment")
    ap.add_argument("--training_id", default=None)
    ap.add_argument("--only", default=None, help="comma list of training sets (all_maps and/or map names)")
    ap.add_argument("--dirs", nargs="*", default=[], help="or explicit training directories")
    ap.add_argument("--maps", default=None, help='"trained", "all", or comma list like grid5_agents6,grid12_agents32')
    ap.add_argument("--checkpoints", default=None, help='"last", "all" or comma list of episodes')
    ap.add_argument("--num_sims", type=int, default=None)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--render", action="store_true", help="write GIFs (default: off)")
    ap.add_argument("--num_gifs", type=int, default=None)
    a = ap.parse_args()
    dirs = list(a.dirs)
    if a.config:
        assert a.training_id, "--training_id is required with --config"
        cfg = load_experiment(a.config)
        for tset in cfg["training_sets"]:
            if a.only and tset["name"] not in a.only.split(","):
                continue
            dirs += [training_dir(cfg, tset["name"], pair, a.training_id) for pair in cfg["pairs"]]
    for d in dirs:
        simulate_dir(d, a.maps, a.checkpoints, a.num_sims, a.deterministic, a.render, a.num_gifs)


if __name__ == "__main__":
    main()
