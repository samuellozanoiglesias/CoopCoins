#!/usr/bin/env python3
"""Speed test: real training/simulation speed of every training set of the full experiment, on this machine.

Uses the FULL experiment settings (all pairs x seeds in one batch, real maps), runs a few PPO updates,
and extrapolates to the full num_episodes. Backend = whatever JAX sees (prefix JAX_PLATFORMS=cpu for CPU).

    python speed_test.py                          # every training set, default backend
    JAX_PLATFORMS=cpu python speed_test.py --sets grid3_agents2,grid4_agents4
Results: printed table + <data_root>/speed_tests/<backend>_<time>.csv
"""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from coopcoins.env import make_env
from coopcoins.ppo import make_trainer
from coopcoins.simulation import make_simulator
from coopcoins.utils import load_experiment, timestamp


def gpu_peak_mb():
    try:
        s = jax.devices()[0].memory_stats()
        return round(s.get("peak_bytes_in_use", 0) / 2**20)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/scaling.yaml")
    ap.add_argument("--sets", default=None, help="comma list of training sets (default: all)")
    ap.add_argument("--updates", type=int, default=20, help="timed PPO updates per set")
    a = ap.parse_args()

    cfg = load_experiment(a.config)
    tc = cfg["train"]
    R = len(cfg["pairs"]) * len(cfg["seeds"])
    n_full = tc["num_episodes"] // tc["num_envs"]
    backend = jax.default_backend()
    print(f"backend={backend} devices={jax.devices()} runs per set={R} full updates={n_full}\n", flush=True)

    rows = []
    for tset in cfg["training_sets"]:
        if a.sets and tset["name"] not in a.sets.split(","):
            continue
        P = max(m["num_coins"] for m in tset["maps"])
        envs = [make_env(**m, max_coins=P) for m in tset["maps"]]
        init_v, chunk_v = make_trainer(envs, tc)
        keys = jnp.stack([jax.random.PRNGKey(i) for i in range(R)])
        coefs = jnp.tile(jnp.array([[1.0, 0.0], [0.7071, 0.7071]]), (R, 1, 1))
        t0 = time.time()
        r = init_v(keys, coefs)
        r, out = chunk_v(r, a.updates)
        jax.block_until_ready(out)
        compile_s = time.time() - t0
        t0 = time.time()
        r, out = chunk_v(r, a.updates)
        jax.block_until_ready(out)
        per_update = (time.time() - t0) / a.updates

        # simulation: 10 training seeds x 20 games on each map of the set
        params = jax.tree.map(lambda x: x[:len(cfg["seeds"])], r.params)
        skeys = jax.random.split(jax.random.PRNGKey(0), cfg["simulation"]["num_sims"])
        sim_s = 0.0
        for m in tset["maps"]:
            env = make_env(**m)
            sim = make_simulator(env)
            jax.block_until_ready(sim(params, jnp.array([[1.0, 0.0], [0.7071, 0.7071]]), skeys))  # compile
            t0 = time.time()
            jax.block_until_ready(sim(params, jnp.array([[1.0, 0.0], [0.7071, 0.7071]]), skeys))
            sim_s += time.time() - t0
        row = {"backend": backend, "training_set": tset["name"], "maps": len(tset["maps"]), "runs": R,
               "compile_s": round(compile_s, 1), "ms_per_update": round(per_update * 1e3, 1),
               "train_full_h": round(per_update * n_full / 3600, 2),
               "sim_per_pair_s": round(sim_s, 2), "gpu_peak_mb": gpu_peak_mb()}
        rows.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    df = pd.DataFrame(rows)
    out = Path(cfg["data_root"]) / "speed_tests"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{backend}_{timestamp()}.csv"
    df.to_csv(path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nTOTAL training if run sequentially on this backend: {df.train_full_h.sum():.1f} h "
          f"(+ compile {df.compile_s.sum() / 60:.1f} min)")
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
