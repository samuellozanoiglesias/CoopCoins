#!/usr/bin/env python3
"""Social influence and scaling analysis of an experiment's simulations.

    python analyze.py --config configs/experiments/scaling.yaml --training_id <id>

Unit of replication = training seed (its simulated games are averaged first).
Each team of each pair is a "focal attitude" with the other team's attitude as "partner".
training = generalist (trained on all maps) | specialist (trained only on the map it plays).

Behaviour (per agent; rates are per game step, the game length is the same on every map):
  own_rate, other_rate   own-team / other-team coins taken per step
  steal_frac             other / (own + other): share of collected coins that were stolen
  restraint              missed_other / (missed_other + other): P(leave an adjacent enemy coin)
  efficiency             own / (own + missed_own): P(take an adjacent own coin)
Outcomes: pure_rate (score / step), coherence_rate (attitude reward / step, the poster's coherence),
  welfare_rate (mean score of everybody / step), advantage_rate (my team - other team, per agent / step)

Outputs (<data_root>/analysis/<experiment>_<training_id>/):
  behaviour_per_seed.csv, behaviour_summary.csv (mean + 95% bootstrap CI over training seeds)
  contrasts.csv          b(A | P2) - b(A | P1) per map, training type and metric, CI + permutation p
  generalization_gap.csv generalist - specialist on each map
  scaling_<family>_<training>.png, influence_vs_size_<family>.png, generalization_gap.png
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coopcoins.utils import load_experiment, save_yaml, training_dir

BEHAVIOUR = ["own_rate", "other_rate", "steal_frac", "restraint", "efficiency"]
OUTCOMES = ["pure_rate", "coherence_rate", "welfare_rate", "advantage_rate"]
METRICS = BEHAVIOUR + OUTCOMES
KEYS = ["training", "trained_on", "map", "grid_size", "num_agents", "attitude", "partner"]
RNG = np.random.default_rng(0)


def collect(cfg, tid, policy):
    rows = []
    for tset in cfg["training_sets"]:
        for pair in cfg["pairs"]:
            d = training_dir(cfg, tset["name"], pair, tid)
            for mdir in sorted((d / "simulations").glob("*")):
                eps = sorted(mdir.glob(f"episode_*_{policy}"), key=lambda p: int(p.name.split("_")[1]))
                if not eps:
                    continue
                g = pd.read_csv(eps[-1] / "games.csv")
                T = cfg["episode_len"]
                # per (seed, team): average the games of each training seed
                s = g.groupby(["train_seed", "team", "attitude", "partner", "map", "grid_size", "num_agents"]
                              ).mean(numeric_only=True).reset_index()
                other = s.set_index(["train_seed", "team"])["pure_reward"]
                opp = other.reindex(list(zip(s.train_seed, 1 - s.team))).values
                taken = s.own_coins + s.other_coins
                rows.append(pd.DataFrame({
                    "training": "generalist" if tset["name"] == "all_maps" else "specialist",
                    "trained_on": tset["name"], "map": s["map"], "grid_size": s.grid_size, "num_agents": s.num_agents,
                    "attitude": s.attitude, "partner": s.partner, "seed": s.train_seed,
                    "own_rate": s.own_coins / T, "other_rate": s.other_coins / T,
                    "steal_frac": s.other_coins / taken.replace(0, np.nan),
                    "restraint": s.missed_other / (s.missed_other + s.other_coins).replace(0, np.nan),
                    "efficiency": s.own_coins / (s.own_coins + s.missed_own).replace(0, np.nan),
                    "pure_rate": s.pure_reward / T, "coherence_rate": s.attitude_reward / T,
                    "welfare_rate": (s.pure_reward + opp) / 2 / T, "advantage_rate": (s.pure_reward - opp) / T}))
    df = pd.concat(rows, ignore_index=True)
    # specialists only count on their own map (transfer runs stay in behaviour_per_seed.csv)
    df["on_training_map"] = (df.training == "generalist") | (df.trained_on == df["map"])
    # self-play pairs (e.g. ind vs ind) give two rows per seed: average them
    return df.groupby(KEYS + ["on_training_map", "seed"])[METRICS].mean().reset_index()


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def diff_test(a, b, n_boot=5000, n_perm=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    d = b.mean() - a.mean()
    boots = RNG.choice(b, (n_boot, len(b))).mean(1) - RNG.choice(a, (n_boot, len(a))).mean(1)
    pooled = np.concatenate([a, b])
    perm = np.array([RNG.permutation(pooled) for _ in range(n_perm)])
    null = perm[:, len(a):].mean(1) - perm[:, :len(a)].mean(1)
    return d, np.percentile(boots, 2.5), np.percentile(boots, 97.5), (np.sum(np.abs(null) >= abs(d)) + 1) / (n_perm + 1)


def summarize(df):
    out = []
    for keys, g in df.groupby(KEYS + ["on_training_map"]):
        row = dict(zip(KEYS + ["on_training_map"], keys), n_seeds=len(g))
        for m in METRICS:
            lo, hi = boot_ci(g[m])
            row.update({m: g[m].mean(), f"{m}_lo": lo, f"{m}_hi": hi})
        out.append(row)
    return pd.DataFrame(out)


def contrasts(df):
    out = []
    main = df[df.on_training_map]
    for (tr, mp, gs, n, A), g in main.groupby(["training", "map", "grid_size", "num_agents", "attitude"]):
        partners = sorted(g.partner.unique())
        for p1, p2 in itertools.combinations(partners, 2):
            for m in METRICS:
                d, lo, hi, p = diff_test(g[g.partner == p1][m], g[g.partner == p2][m])
                out.append({"training": tr, "map": mp, "grid_size": gs, "num_agents": n, "attitude": A,
                            "partner_1": p1, "partner_2": p2, "metric": m, "mean_1": g[g.partner == p1][m].mean(),
                            "mean_2": g[g.partner == p2][m].mean(), "diff": d, "lo": lo, "hi": hi, "p_perm": p})
    return pd.DataFrame(out)


def generalization_gap(df):
    out = []
    main = df[df.on_training_map]
    for (mp, gs, n, A, P), g in main.groupby(["map", "grid_size", "num_agents", "attitude", "partner"]):
        sp, ge = g[g.training == "specialist"], g[g.training == "generalist"]
        if len(sp) and len(ge):
            for m in METRICS:
                d, lo, hi, p = diff_test(sp[m], ge[m])
                out.append({"map": mp, "grid_size": gs, "num_agents": n, "attitude": A, "partner": P, "metric": m,
                            "specialist": sp[m].mean(), "generalist": ge[m].mean(),
                            "gap_generalist_minus_specialist": d, "lo": lo, "hi": hi, "p_perm": p})
    return pd.DataFrame(out)


def _err(ax, x, y, lo, hi, **kw):
    ax.errorbar(x, y, yerr=[np.abs(y - lo), np.abs(hi - y)], capsize=3, marker="o", **kw)


def fig_scaling(summ, families, out):
    metrics = ["own_rate", "other_rate", "steal_frac", "restraint", "pure_rate", "welfare_rate"]
    s = summ[summ.on_training_map]
    for fam, names in families.items():
        for tr in s.training.unique():
            f = s[(s["map"].isin(names)) & (s.training == tr)]
            if f.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
            for ax, m in zip(axes.flat, metrics):
                for (A, P), g in f.groupby(["attitude", "partner"]):
                    g = g.sort_values("grid_size")
                    _err(ax, g.grid_size, g[m], g[f"{m}_lo"], g[f"{m}_hi"], label=f"{A} | partner {P}",
                         ls="-" if A == "individualistic" else ":")
                ax.set_title(m)
                ax.set_xlabel("grid size")
                ax.grid(alpha=0.3)
                ax.set_xticks(sorted(f.grid_size.unique()),
                              [f"{g}\n{n}ag" for g, n in f.groupby("grid_size").num_agents.first().items()], fontsize=8)
            axes[0, 0].legend(fontsize=7, frameon=False)
            fig.suptitle(f"{fam} family, {tr} agents: focal attitude | partner (mean, 95% CI over training seeds)")
            fig.tight_layout()
            fig.savefig(out / f"scaling_{fam}_{tr}.png", dpi=120)
            plt.close(fig)


def fig_influence(con, families, out, focal="individualistic"):
    """How much the partner changes the focal attitude's behaviour, as the map grows."""
    metrics = ["steal_frac", "own_rate", "other_rate", "pure_rate"]
    c = con[con.attitude == focal]
    for fam, names in families.items():
        f = c[c["map"].isin(names)]
        if f.empty:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4))
        combos = sorted({(r.partner_1, r.partner_2) for r in f.itertuples()})
        colors = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(combos)}
        for ax, m in zip(axes, metrics):
            for (tr, p1, p2), g in f[f.metric == m].groupby(["training", "partner_1", "partner_2"]):
                g = g.sort_values("grid_size")
                _err(ax, g.grid_size, g["diff"], g.lo, g.hi, label=f"partner {p2} - {p1} ({tr})",
                     ls="-" if tr == "generalist" else "--", color=colors[(p1, p2)], alpha=1 if tr == "generalist" else 0.6)
            ax.set_xticks(sorted(f.grid_size.unique()))
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"Δ {m} of the {focal} team")
            ax.set_xlabel("grid size")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Social influence on the {focal} team vs map size ({fam} family): partner effect, 95% CI")
        fig.tight_layout()
        fig.savefig(out / f"influence_vs_size_{fam}.png", dpi=120)
        plt.close(fig)


def fig_gap(gap, out):
    if gap.empty:
        return
    metrics = ["steal_frac", "own_rate", "pure_rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        for (A, P), g in gap[gap.metric == m].groupby(["attitude", "partner"]):
            g = g.sort_values(["num_agents", "grid_size"])
            ax.errorbar(range(len(g)), g.gap_generalist_minus_specialist,
                        yerr=[np.abs(g.gap_generalist_minus_specialist - g.lo), np.abs(g.hi - g.gap_generalist_minus_specialist)],
                        marker="o", capsize=3, ls="none", label=f"{A} | {P}")
            ax.set_xticks(range(len(g)), g["map"], rotation=60, fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"generalist - specialist: {m}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "generalization_gap.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/scaling.yaml")
    ap.add_argument("--training_id", required=True)
    ap.add_argument("--policy", default="stochastic", choices=["stochastic", "greedy"])
    a = ap.parse_args()

    cfg = load_experiment(a.config)
    out = Path(cfg["data_root"]) / "analysis" / f"{cfg['name']}_{a.training_id}"
    out.mkdir(parents=True, exist_ok=True)
    families = {k: [m["name"] for m in v] for k, v in cfg["families"].items()}

    df = collect(cfg, a.training_id, a.policy)
    df.to_csv(out / "behaviour_per_seed.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "behaviour_summary.csv", index=False)
    con = contrasts(df)
    con.to_csv(out / "contrasts.csv", index=False)
    gap = generalization_gap(df)
    gap.to_csv(out / "generalization_gap.csv", index=False)

    fig_scaling(summ, families, out)
    fig_influence(con, families, out)
    fig_gap(gap, out)
    save_yaml({"config": a.config, "training_id": a.training_id, "policy": a.policy, "families": families},
              out / "analysis.yaml")

    c = con[(con.attitude == "individualistic") & (con.metric == "steal_frac")]
    if len(c):
        print("Individualistic team, steal_frac: partner effect (95% CI over training seeds)")
        for _, r in c.sort_values(["training", "num_agents"]).iterrows():
            print(f"  {r.training:10s} {r['map']:16s} {r.partner_1:>15s} -> {r.partner_2:15s} "
                  f"{r.mean_1:.3f} -> {r.mean_2:.3f}  diff {r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p_perm:.4f}")
    print(f"\nanalysis written to {out}")


if __name__ == "__main__":
    main()
