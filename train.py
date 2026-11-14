#!/usr/bin/env python3
"""Train every (training set x attitude pair x seed) of an experiment.

Training sets (see `training:` in the config): `all_maps` (one generalist pair of team policies trained
on every map) and/or one specialist set per map. Pairs x seeds of a set run as one vmapped program.

    python train.py --config configs/experiments/scaling.yaml --training_id <id>
    python train.py --config configs/experiments/scaling.yaml --only all_maps          # one set
    python train.py --config configs/experiments/scaling.yaml --only grid3_agents2,grid6_agents4

<data_root>/<set>/<red_blue>/Training_<id>/
    config.yaml  training_stats.csv  training_curves.png  summary.yaml
    checkpoints/seed_<s>/episode_<n>.pkl     {'team_0': params, 'team_1': params, ...}
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from coopcoins.env import STATS, make_env
from coopcoins.plots import training_plot
from coopcoins.ppo import make_trainer
from coopcoins.utils import load_experiment, pair_name, save_checkpoint, save_yaml, timestamp, training_dir


def train_set(cfg, tset, training_id):
    tc, seeds, pairs = cfg["train"], cfg["seeds"], cfg["pairs"]
    max_coins = max(m["num_coins"] for m in tset["maps"])
    envs = [make_env(**m, max_coins=max_coins) for m in tset["maps"]]
    init_v, chunk_v = make_trainer(envs, tc)
    n_updates = tc["num_episodes"] // tc["num_envs"]
    n_ckpt, log_every = tc["num_checkpoints"], tc.get("log_every", 1)
    chunk = n_updates // n_ckpt
    assert n_updates % n_ckpt == 0 and chunk % log_every == 0, "num_episodes/num_envs must split into checkpoints x log_every"
    print(f"=== training set {tset['name']}: {[m['name'] for m in tset['maps']]}, "
          f"{tc['num_episodes']} games per map of {cfg['episode_len']} steps ===", flush=True)

    dirs = {}
    for pair in pairs:
        d = training_dir(cfg, tset["name"], pair, training_id)
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        dirs[pair_name(pair)] = d
        save_yaml({"name": cfg["name"], "data_root": cfg["data_root"], "training_set": tset["name"],
                   "maps": tset["maps"], "pair": pair, "reward_coef": [cfg["attitudes"][a] for a in pair],
                   "attitudes": cfg["attitudes"], "seeds": seeds, "train": tc, "simulation": cfg["simulation"],
                   "experiment_maps": list(cfg["maps"].values()), "training_id": training_id,
                   "jax_backend": jax.default_backend()}, d / "config.yaml")

    runs = [(pair, s) for pair in pairs for s in seeds]
    bs = tc.get("max_parallel_runs") or len(runs)
    t0 = time.time()
    for b in range(0, len(runs), bs):
        batch = runs[b:b + bs]
        keys = jnp.stack([jax.random.PRNGKey(s) for _, s in batch])
        coefs = jnp.array([[cfg["attitudes"][a] for a in pair] for pair, _ in batch])
        runner = init_v(keys, coefs)

        def checkpoint(update):
            ep = update * tc["num_envs"]
            for i, (pair, s) in enumerate(batch):
                save_checkpoint(dirs[pair_name(pair)] / "checkpoints" / f"seed_{s}" / f"episode_{ep}.pkl",
                                jax.tree.map(lambda x: x[i], runner.params), pair=pair, seed=s, episode=ep,
                                reward_coef=np.asarray(coefs[i]).tolist(), hidden=tc["hidden"],
                                trained_on=[m["name"] for m in tset["maps"]])

        checkpoint(0)
        for c in range(n_ckpt):
            tc0 = time.time()
            runner, (mets, losses) = chunk_v(runner, chunk)
            mets, losses = jax.device_get((mets, losses))   # leaves (R, chunk, 2)
            blk = lambda x: x.reshape(x.shape[0], -1, log_every, 2).mean(2)  # (R, rows, 2)
            upd = (np.arange(chunk // log_every) + 1) * log_every + c * chunk
            frames = {pn: [] for pn in dirs}
            for i, (pair, s) in enumerate(batch):
                for mi, m in enumerate(tset["maps"]):
                    for k in (0, 1):
                        row = {"seed": s, "update": upd, "episode": upd * tc["num_envs"], "map": m["name"], "team": k,
                               "attitude": pair[k]}
                        row.update({st: blk(mets[mi][st])[i, :, k] for st in ["pure_reward", "attitude_reward", *STATS]})
                        row.update({ls: blk(losses[ls])[i, :, k] for ls in losses})
                        frames[pair_name(pair)].append(pd.DataFrame(row))
            for pn, fr in frames.items():
                if fr:
                    path = dirs[pn] / "training_stats.csv"
                    pd.concat(fr).to_csv(path, mode="a", header=not path.exists(), index=False)
            checkpoint((c + 1) * chunk)
            for pair in {pair_name(p): p for p, _ in batch}.values():
                d = dirs[pair_name(pair)]
                training_plot(d / "training_stats.csv", d / "training_curves.png", pair, cfg["episode_len"])
            print(f"  runs {b + 1}-{b + len(batch)}/{len(runs)} | game {(c + 1) * chunk * tc['num_envs']}/"
                  f"{tc['num_episodes']} per map | {time.time() - tc0:.1f}s", flush=True)

    elapsed = time.time() - t0
    for d in dirs.values():
        write_summary(d, elapsed)


def write_summary(d, elapsed):
    """Training report: per map and team, per-agent means per game over the last 10% of training."""
    df = pd.read_csv(d / "training_stats.csv")
    last = df[df.episode > 0.9 * df.episode.max()]
    cols = ["pure_reward", "attitude_reward", *STATS, "entropy"]
    per_seed = last.groupby(["map", "team", "seed"])[cols].mean()
    summary = {"wall_time_s_whole_batch": round(elapsed, 1), "last_10pct": {}}
    for (mp, team), g in per_seed.groupby(level=[0, 1]):
        summary["last_10pct"].setdefault(mp, {})[f"team{team}"] = {
            "mean_over_seeds": g.mean().round(3).to_dict(), "std_over_seeds": g.std().round(3).to_dict()}
    save_yaml(summary, d / "summary.yaml")
    m = per_seed.groupby(level=[0, 1]).mean()
    print(f"  [{d.parent.name}] " + " | ".join(
        f"{mp}: other coins red/blue {m.loc[(mp, 0), 'other_coins']:.1f}/{m.loc[(mp, 1), 'other_coins']:.1f}"
        for mp in m.index.get_level_values(0).unique()[:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/scaling.yaml")
    ap.add_argument("--training_id", default=None, help="default: current timestamp")
    ap.add_argument("--only", default=None, help="comma list of training sets (all_maps and/or map names)")
    ap.add_argument("--seeds", type=int, default=None, help="override: seeds 0..N-1")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    if args.seeds:
        cfg["seeds"] = list(range(args.seeds))
    tid = args.training_id or timestamp()
    print(f"training_id = {tid}")
    for tset in cfg["training_sets"]:
        if args.only and tset["name"] not in args.only.split(","):
            continue
        train_set(cfg, tset, tid)


if __name__ == "__main__":
    main()
