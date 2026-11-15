#!/usr/bin/env python3
"""Time one PPO update of a full experiment config and extrapolate the training wall time.

    python scripts/benchmark.py --config configs/experiments/payoff_k2.yaml --auto
    python scripts/benchmark.py --config configs/experiments/payoff_k2.yaml --max_parallel_runs 10

--auto tries batch sizes (vmapped runs) from all runs downwards, each in a separate process (so GPU memory is fully
released after an out-of-memory failure), and reports the largest size that fits plus its estimate. This is the
value to use as MAX_PARALLEL_RUNS for k_train.sh. Batch size does not change results (each run's random stream
depends only on its seed).
With a fixed --max_parallel_runs: compiles a 2-update chunk for one batch, times a second call, and prints seconds
per update, number of batches, estimated training time and peak device memory.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OOM_EXIT = 3


def candidates(n_runs):
    """Batch sizes from n_runs down to 1, preferring divisors (no short last batch)."""
    divs = [d for d in range(n_runs, 0, -1) if n_runs % d == 0]
    return divs


def run_one(cfg_path, mpr, updates):
    import jax
    import jax.numpy as jnp

    from coopcoins.env import make_env
    from coopcoins.ppo import make_trainer
    from coopcoins.utils import load_experiment

    cfg = load_experiment(cfg_path)
    tc = cfg["train"]
    runs = [(p, s) for p in cfg["pairs"] for s in cfg["seeds"]]
    bs = mpr or len(runs)
    n_batches = -(-len(runs) // bs)
    n_updates = tc["num_episodes"] // tc["num_envs"]
    dev = jax.devices()[0]
    total = 0.0
    try:
        for tset in cfg["training_sets"]:
            mc = max(m["num_coins"] for m in tset["maps"])
            envs = [make_env(**m, max_coins=mc) for m in tset["maps"]]
            init_v, chunk_v = make_trainer(envs, tc)
            batch = runs[:bs]
            keys = jnp.stack([jax.random.PRNGKey(s) for _, s in batch])
            coefs = jnp.array([[cfg["attitudes"][x] for x in p] for p, _ in batch])
            r = init_v(keys, coefs)
            t0 = time.time()
            r, out = chunk_v(r, updates)
            jax.block_until_ready(out)
            t_compile = time.time() - t0
            t0 = time.time()
            r, out = chunk_v(r, updates)
            jax.block_until_ready(out)
            per_update = (time.time() - t0) / updates
            est = per_update * n_updates * n_batches
            total += est
            mem = ""
            stats = dev.memory_stats() if hasattr(dev, "memory_stats") else None
            if stats and "peak_bytes_in_use" in stats:
                mem = (f", peak memory {stats['peak_bytes_in_use'] / 2**30:.2f} GiB"
                       f" of {stats.get('bytes_limit', 0) / 2**30:.2f} GiB")
            print(f"[{cfg['name']} | {tset['name']} | backend {jax.default_backend()}] {len(batch)} runs vmapped, "
                  f"{len(tset['maps'])} maps: compile+{updates} updates {t_compile:.1f}s, {per_update:.3f}s per update "
                  f"per batch, {n_batches} batch(es) x {n_updates} updates -> {est / 3600:.2f} h{mem}", flush=True)
    except Exception as e:                                     # jaxlib XlaRuntimeError
        if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
            print(f"[{cfg['name']}] {bs} runs vmapped: OUT OF MEMORY", flush=True)
            sys.exit(OOM_EXIT)
        raise
    print(f"ESTIMATE {cfg['name']}: {total / 3600:.2f} h of training with MAX_PARALLEL_RUNS={bs} "
          f"(excluding compilation and checkpoint I/O)", flush=True)


def auto(cfg_path, updates):
    from coopcoins.utils import load_experiment
    cfg = load_experiment(cfg_path)
    n_runs = len(cfg["pairs"]) * len(cfg["seeds"])
    for bs in candidates(n_runs):
        res = subprocess.run([sys.executable, __file__, "--config", cfg_path, "--max_parallel_runs", str(bs),
                              "--updates", str(updates)], capture_output=True, text=True)
        lines = [ln for ln in res.stdout.splitlines() if ln.startswith(("[", "ESTIMATE"))]
        print("\n".join(lines), flush=True)
        if res.returncode == 0:
            print(f"RECOMMENDED MAX_PARALLEL_RUNS={bs} for {cfg['name']} (largest batch that fits; if training runs "
                  f"out of memory later, use the next smaller divisor of {n_runs})", flush=True)
            return
        if res.returncode != OOM_EXIT:
            sys.stderr.write(res.stderr[-4000:])
            sys.exit(f"benchmark failed for MAX_PARALLEL_RUNS={bs} (not an out-of-memory error)")
    sys.exit("even 1 run does not fit in device memory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_parallel_runs", type=int, default=None, help="0 = all runs in one batch")
    ap.add_argument("--auto", action="store_true", help="search the largest batch that fits in memory")
    ap.add_argument("--updates", type=int, default=2)
    a = ap.parse_args()
    if a.auto:
        auto(a.config, a.updates)
    else:
        if a.max_parallel_runs is None:
            from coopcoins.utils import load_experiment
            a.max_parallel_runs = load_experiment(a.config)["train"].get("max_parallel_runs") or 0
        run_one(a.config, a.max_parallel_runs, a.updates)


if __name__ == "__main__":
    main()
