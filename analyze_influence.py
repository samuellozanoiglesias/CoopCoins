#!/usr/bin/env python3
"""Analysis of the social-influence study.

    python analyze_influence.py --exp 1   # strategic vs ecological decomposition (cross-play, scaling_v1)
    python analyze_influence.py --exp 2   # policy probes: preference weights and state-matched policy effects
    python analyze_influence.py --exp 3   # decomposition vs scarcity and group size (scarcity_v1 + scaling_v1)
    python analyze_influence.py --exp 4   # what an observational study would conclude vs the true decomposition

Definitions (focal team = red, individualistic; b[A, B] = focal behaviour when TRAINED with partner A and
PLAYING with partner B; every b is a mean over training seeds, cross-seed pairing):
    total       T = b[P, P] - b[R, R]
    ecological  E = 1/2 [(b[R, P] - b[R, R]) + (b[P, P] - b[P, R])]   (same policy, other partner)
    strategic   S = 1/2 [(b[P, R] - b[R, R]) + (b[P, P] - b[R, P])]   (same partner, other policy)
    interaction I = (b[P, P] - b[P, R]) - (b[R, P] - b[R, R])
    E + S = T;  strategic share = |S| / (|S| + |E|)
    co-adaptation C = b_co[A] - b[A, A]   (playing the exact co-trained partner vs another partner of the same type)
95% CIs: bootstrap over training seeds (seed index resampled jointly across the cells it links), 2000 draws.
Outputs: <data_root>/influence/analysis/exp<N>/
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coopcoins.influence import PROBE_TERMS
from coopcoins.utils import _yaml, load_experiment

RNG = np.random.default_rng(0)
NB = 2000
METRICS = ["own_rate", "other_rate", "coin_rate", "steal_frac", "efficiency", "restraint", "pure_rate"]
PLOT_METRICS = ["own_rate", "other_rate", "steal_frac", "pure_rate"]
KEYS = ["training_id", "training_set", "training", "map", "grid_size", "num_agents", "num_coins",
        "train_partner", "play_partner", "pairing"]
COL = {"E": "#2e8b57", "S": "#8e44ad", "T": "#333333"}


# ------------------------------------------------------------------ data
def seed_metrics(df, team=0):
    """Game rows -> one row per (cell, focal seed): rates per agent per step and fractions (sums over games)."""
    d = df[df.team == team]
    g = d.groupby(KEYS + ["focal_seed"])
    s = g[["own_coins", "other_coins", "missed_own", "missed_other", "pure_reward"]].sum()
    n = g.size() * g["episode_len"].first()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "own_rate": s.own_coins / n, "other_rate": s.other_coins / n,
            "coin_rate": (s.own_coins + s.other_coins) / n,
            "steal_frac": s.other_coins / (s.own_coins + s.other_coins),
            "efficiency": s.own_coins / (s.own_coins + s.missed_own),
            "restraint": s.missed_other / (s.missed_other + s.other_coins),
            "pure_rate": s.pure_reward / n,
            "opportunity_other": (s.other_coins + s.missed_other) /
                                 (s.own_coins + s.missed_own + s.other_coins + s.missed_other)})
    out = out.reset_index()
    out["coins_per_agent"] = (out.num_coins / out.num_agents).round(3)
    return out


def load_crossplay(data_root, tid):
    p = Path(data_root) / "influence" / tid / "crossplay_games.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} (run crossplay.py first)")
    return pd.read_csv(p)


def contrasts(partners):
    return list(itertools.combinations(partners, 2))  # (R, P): ref -> new, e.g. ind->coop, ind->comp, coop->comp


# ------------------------------------------------------------------ statistics
def _ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def _p(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())) if len(x) else np.nan


def decompose(g, R, P, metric):
    """g: seed metrics of one (training set, map). Returns point estimates + bootstrap CIs."""
    cell = lambda a, b: g[(g.train_partner == a) & (g.play_partner == b) & (g.pairing == "cross")] \
        .set_index("focal_seed")[metric]
    c = {k: cell(*ab) for k, ab in {"RR": (R, R), "RP": (R, P), "PR": (P, R), "PP": (P, P)}.items()}
    seeds = sorted(set.intersection(*[set(v.index) for v in c.values()]))
    if len(seeds) < 2:
        return None
    x = {k: v.loc[seeds].values for k, v in c.items()}

    def est(idx):
        m = {k: np.nanmean(v[idx], axis=-1) for k, v in x.items()}
        E = 0.5 * ((m["RP"] - m["RR"]) + (m["PP"] - m["PR"]))
        S = 0.5 * ((m["PR"] - m["RR"]) + (m["PP"] - m["RP"]))
        I = (m["PP"] - m["PR"]) - (m["RP"] - m["RR"])
        return E, S, I, E + S, np.abs(S) / (np.abs(S) + np.abs(E))

    n = len(seeds)
    point = est(np.arange(n))
    boot = est(RNG.integers(0, n, (NB, n)))
    row = {"n_seeds": n, **{f"b_{k}": np.nanmean(v) for k, v in x.items()}}
    for name, pt, bs in zip(["ecological", "strategic", "interaction", "total", "strategic_share"], point, boot):
        lo, hi = _ci(bs)
        row.update({name: pt, f"{name}_lo": lo, f"{name}_hi": hi})
        if name != "strategic_share":
            row[f"{name}_p"] = _p(bs)
    return row


def decomposition_table(sm, partners, group=("training_id", "training_set", "training", "map", "grid_size",
                                             "num_agents", "num_coins", "coins_per_agent")):
    rows = []
    for keys, g in sm.groupby(list(group)):
        for R, P in contrasts(partners):
            for m in METRICS:
                r = decompose(g, R, P, m)
                if r:
                    rows.append({**dict(zip(group, keys)), "contrast": f"{R}->{P}", "ref": R, "new": P,
                                 "metric": m, **r})
    return pd.DataFrame(rows)


def coadaptation_table(sm, partners):
    rows = []
    for keys, g in sm.groupby(["training_set", "training", "map", "num_agents", "num_coins"]):
        for A in partners:
            co = g[(g.train_partner == A) & (g.pairing == "co")].set_index("focal_seed")
            cr = g[(g.train_partner == A) & (g.play_partner == A) & (g.pairing == "cross")].set_index("focal_seed")
            seeds = sorted(set(co.index) & set(cr.index))
            if len(seeds) < 2:
                continue
            for m in METRICS:
                d = co.loc[seeds, m].values - cr.loc[seeds, m].values
                bs = np.nanmean(d[RNG.integers(0, len(d), (NB, len(d)))], axis=1)
                lo, hi = _ci(bs)
                rows.append({**dict(zip(["training_set", "training", "map", "num_agents", "num_coins"], keys)),
                             "partner": A, "metric": m, "co_trained": np.nanmean(co.loc[seeds, m]),
                             "cross": np.nanmean(cr.loc[seeds, m]), "coadaptation": np.nanmean(d), "lo": lo, "hi": hi,
                             "p": _p(bs)})
    return pd.DataFrame(rows)


def cells_table(sm):
    rows = []
    for keys, g in sm.groupby(KEYS):
        r = dict(zip(KEYS, keys))
        for m in METRICS:
            v = g[m].values
            lo, hi = _ci(np.nanmean(v[RNG.integers(0, len(v), (NB, len(v)))], axis=1))
            r.update({m: np.nanmean(v), f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _err(ax, x, y, lo, hi, **kw):
    ax.errorbar(x, y, yerr=[np.abs(y - lo), np.abs(hi - y)], capsize=3, marker="o", **kw)


def fig_cells(cells, partners, out, tag):
    """b[A, B] matrices (cross-seed pairing) for the smallest and largest map of each training type."""
    c = cells[cells.pairing == "cross"]
    for tr, ct in c.groupby("training"):
        maps = ct.sort_values("num_agents")["map"].unique()
        maps = [maps[0], maps[-1]] if len(maps) > 1 else list(maps)
        fig, axes = plt.subplots(len(maps), len(PLOT_METRICS), figsize=(3.6 * len(PLOT_METRICS), 3.3 * len(maps)),
                                 squeeze=False)
        for i, mp in enumerate(maps):
            cm = ct[ct["map"] == mp]
            for j, m in enumerate(PLOT_METRICS):
                M = cm.pivot(index="train_partner", columns="play_partner", values=m).loc[partners, partners]
                ax = axes[i, j]
                im = ax.imshow(M.values, cmap="viridis")
                for (a, b), v in np.ndenumerate(M.values):
                    ax.text(b, a, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
                ax.set_xticks(range(len(partners)), [p[:5] for p in partners])
                ax.set_yticks(range(len(partners)), [p[:5] for p in partners])
                ax.set_xlabel("plays with"); ax.set_ylabel("trained with")
                ax.set_title(f"{mp}: {m}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Focal (individualistic) behaviour b[trained with, plays with] — {tr}")
        fig.tight_layout()
        fig.savefig(out / f"cells_{tag}_{tr}.png", dpi=120)
        plt.close(fig)


def fig_decomposition(dec, x, out, name, lines=None, xlabel=None):
    """E, S, total (95% CI) against x, one panel per (metric, contrast). `lines` splits into several curves."""
    cons = dec.contrast.unique()
    fig, axes = plt.subplots(len(PLOT_METRICS), len(cons), figsize=(4.6 * len(cons), 3.0 * len(PLOT_METRICS)),
                             squeeze=False)
    for i, m in enumerate(PLOT_METRICS):
        for j, c in enumerate(cons):
            ax = axes[i, j]
            d = dec[(dec.metric == m) & (dec.contrast == c)]
            groups = d.groupby(lines) if lines else [("", d)]
            for k, (lab, g) in enumerate(groups):
                g = g.sort_values(x)
                ls = ["-", "--", ":", "-."][k % 4]
                for comp, key in (("E", "ecological"), ("S", "strategic"), ("T", "total")):
                    _err(ax, g[x], g[key], g[f"{key}_lo"], g[f"{key}_hi"], color=COL[comp], ls=ls, ms=3,
                         label=f"{key}{' ' + str(lab) if lines else ''}" if (i == 0 and j == 0) else None)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"{m}: {c}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == len(PLOT_METRICS) - 1:
                ax.set_xlabel(xlabel or x)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Influence decomposition on the focal team — {name}")
    fig.tight_layout()
    fig.savefig(out / f"decomposition_{name}.png", dpi=120)
    plt.close(fig)


def fig_share(dec, x, out, name, lines=None, xlabel=None):
    d = dec[dec.metric.isin(["own_rate", "other_rate", "pure_rate"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, m in zip(axes, ["own_rate", "other_rate", "pure_rate"]):
        for (c, *lab), g in d[d.metric == m].groupby(["contrast"] + ([lines] if lines else [])):
            g = g.sort_values(x)
            _err(ax, g[x], g.strategic_share, g.strategic_share_lo, g.strategic_share_hi, ms=3,
                 label=f"{c}{' ' + str(lab[0]) if lab else ''}")
        ax.set_ylim(-0.02, 1.02); ax.set_title(f"strategic share: {m}"); ax.set_xlabel(xlabel or x); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"strategic_share_{name}.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ experiments
def exp1(icfg, out):
    spec = icfg["experiments"]["exp1"]
    cfg = load_experiment(spec["experiment"])
    partners = icfg["partners"]
    sm = seed_metrics(load_crossplay(cfg["data_root"], spec["training_id"]))
    sm.to_csv(out / "seed_metrics.csv", index=False)
    cells = cells_table(sm)
    cells.to_csv(out / "cells.csv", index=False)
    dec = decomposition_table(sm, partners)
    dec.to_csv(out / "decomposition.csv", index=False)
    coadaptation_table(sm, partners).to_csv(out / "coadaptation.csv", index=False)
    fig_cells(cells, partners, out, "exp1")
    for fam, maps in cfg["families"].items():
        names = [m["name"] for m in maps]
        for tr in dec.training.unique():
            d = dec[(dec["map"].isin(names)) & (dec.training == tr)]
            if d["map"].nunique() > 1:
                fig_decomposition(d, "grid_size", out, f"{fam}_{tr}", xlabel=f"grid size ({fam} family)")
                fig_share(d, "grid_size", out, f"{fam}_{tr}", xlabel=f"grid size ({fam} family)")
    print_key(dec)


def print_key(dec):
    k = dec[dec.metric.isin(["own_rate", "steal_frac", "pure_rate"])].sort_values(
        ["training", "metric", "contrast", "num_agents", "num_coins"])
    print("\nKey decomposition (E = ecological, S = strategic, 95% CI over training seeds):")
    for _, r in k.iterrows():
        print(f"  {r.training:10s} {r['map']:24s} {r.metric:10s} {r.contrast:30s} "
              f"T {r.total:+.3f}  E {r.ecological:+.3f} [{r.ecological_lo:+.3f},{r.ecological_hi:+.3f}]  "
              f"S {r.strategic:+.3f} [{r.strategic_lo:+.3f},{r.strategic_hi:+.3f}]  share {r.strategic_share:.2f}")


def probe_shifts(coef, partners):
    """Strategic change of revealed preferences: beta(trained with P) - beta(trained with R), focal team."""
    f = coef[coef.team == 0]
    rows = []
    for keys, g in f.groupby(["training_id", "training_set", "training", "map", "grid_size", "num_agents",
                              "num_coins"]):
        for R, P in contrasts(partners):
            a, b = g[g.trained_with == R], g[g.trained_with == P]
            if len(a) < 2 or len(b) < 2:
                continue
            for t in PROBE_TERMS + ["p_stay"]:
                xa, xb = a[t].values, b[t].values
                bs = xb[RNG.integers(0, len(xb), (NB, len(xb)))].mean(1) - xa[RNG.integers(0, len(xa), (NB, len(xa)))].mean(1)
                lo, hi = _ci(bs)
                rows.append({**dict(zip(["training_id", "training_set", "training", "map", "grid_size", "num_agents",
                                         "num_coins"], keys)), "contrast": f"{R}->{P}", "term": t,
                             "with_ref": xa.mean(), "with_new": xb.mean(), "shift": xb.mean() - xa.mean(),
                             "lo": lo, "hi": hi, "p": _p(bs)})
    return pd.DataFrame(rows)


def matched_effects(mt, partners):
    """State-matched strategic effect: same states, policy trained with P vs with R; Shapley average over worlds."""
    rows = []
    for keys, g in mt.groupby(["training_id", "training_set", "training", "map", "grid_size", "num_agents",
                               "num_coins"]):
        for R, P in contrasts(partners):
            for m in ["p_own", "p_other", "p_stay", "steal_share"]:
                x = {(A, W): g[(g.trained_with == A) & (g.world == W)].set_index("seed")[m]
                     for A in (R, P) for W in (R, P)}
                if any(len(v) < 2 for v in x.values()):
                    continue
                sR, sP = sorted(x[(R, R)].index), sorted(x[(P, P)].index)
                xr = {W: x[(R, W)].loc[sR].values for W in (R, P)}
                xp = {W: x[(P, W)].loc[sP].values for W in (R, P)}

                def est(ir, ip):
                    return 0.5 * sum(xp[W][ip].mean(-1) - xr[W][ir].mean(-1) for W in (R, P))

                pt = est(np.arange(len(sR)), np.arange(len(sP)))
                bs = est(RNG.integers(0, len(sR), (NB, len(sR))), RNG.integers(0, len(sP), (NB, len(sP))))
                lo, hi = _ci(bs)
                rows.append({**dict(zip(["training_id", "training_set", "training", "map", "grid_size",
                                         "num_agents", "num_coins"], keys)), "contrast": f"{R}->{P}", "metric": m,
                             "strategic_policy_effect": pt, "lo": lo, "hi": hi, "p": _p(bs),
                             **{f"{m}_{a[:4]}_on_{w[:4]}": x[(a, w)].mean() for a in (R, P) for w in (R, P)}})
    return pd.DataFrame(rows)


def fig_fingerprints(coef, out, tag):
    """Revealed preference weights of every team policy (attitude | trained with), on two maps per training type."""
    for tr, ct in coef.groupby("training"):
        maps = ct.sort_values("num_agents")["map"].unique()
        maps = [maps[0], maps[-1]] if len(maps) > 1 else list(maps)
        fig, axes = plt.subplots(len(maps), 1, figsize=(13, 3.6 * len(maps)), squeeze=False)
        for ax, mp in zip(axes[:, 0], maps):
            c = ct[ct["map"] == mp]
            groups = list(c.groupby(["attitude", "trained_with"]))
            w = 0.8 / len(groups)
            for i, ((att, tw), g) in enumerate(groups):
                mu = g[PROBE_TERMS].mean()
                se = g[PROBE_TERMS].std() / np.sqrt(len(g))
                ax.bar(np.arange(len(PROBE_TERMS)) + i * w, mu, w, yerr=1.96 * se, capsize=2,
                       label=f"{att} (trained with {tw})")
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(np.arange(len(PROBE_TERMS)) + 0.4 - w / 2, PROBE_TERMS)
            ax.set_ylabel("effect on log P(own) / P(other)")
            ax.set_title(f"{mp} ({tr})")
            ax.grid(axis="y", alpha=0.3)
        axes[0, 0].legend(fontsize=7, ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"probe_fingerprints_{tag}_{tr}.png", dpi=120)
        plt.close(fig)


def fig_probe_shifts(sh, x, out, name, lines=None, xlabel=None):
    terms = ["colour_bias", "threat_own", "threat_other", "p_stay"]
    cons = sh.contrast.unique()
    fig, axes = plt.subplots(len(terms), len(cons), figsize=(4.6 * len(cons), 2.8 * len(terms)), squeeze=False)
    for i, t in enumerate(terms):
        for j, c in enumerate(cons):
            ax = axes[i, j]
            for k, (lab, g) in enumerate(sh[(sh.term == t) & (sh.contrast == c)].groupby(lines) if lines
                                         else [("", sh[(sh.term == t) & (sh.contrast == c)])]):
                g = g.sort_values(x)
                _err(ax, g[x], g["shift"], g.lo, g.hi, ms=3, label=str(lab) if lines else None)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"Δ {t}: {c}", fontsize=9); ax.grid(alpha=0.3)
            if i == len(terms) - 1:
                ax.set_xlabel(xlabel or x)
    if lines:
        axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Strategic change of revealed preferences (focal policy trained with new vs ref partner) — {name}")
    fig.tight_layout()
    fig.savefig(out / f"probe_shifts_{name}.png", dpi=120)
    plt.close(fig)


def exp2(icfg, out, root_out):
    spec = icfg["experiments"]["exp1"]
    cfg = load_experiment(spec["experiment"])
    partners = icfg["partners"]
    base = Path(cfg["data_root"]) / "influence" / spec["training_id"]
    coef = pd.read_csv(base / "probe_coefficients.csv")
    mt = pd.read_csv(base / "probe_matched.csv")
    sh = probe_shifts(coef, partners)
    sh.to_csv(out / "probe_shifts.csv", index=False)
    me = matched_effects(mt, partners)
    me.to_csv(out / "matched_effects.csv", index=False)
    fig_fingerprints(coef, out, "exp2")
    for fam, maps in cfg["families"].items():
        names = [m["name"] for m in maps]
        for tr in sh.training.unique():
            d = sh[(sh["map"].isin(names)) & (sh.training == tr)]
            if d["map"].nunique() > 1:
                fig_probe_shifts(d, "grid_size", out, f"{fam}_{tr}", xlabel=f"grid size ({fam} family)")
    # policy-level vs behaviour-level strategic effect (needs exp1 analysis)
    d1 = root_out / "exp1" / "decomposition.csv"
    if d1.exists():
        dec = pd.read_csv(d1)
        b = dec[dec.metric == "steal_frac"][["training_set", "map", "contrast", "strategic", "ecological"]]
        p = me[me.metric == "steal_share"][["training_set", "map", "contrast", "strategic_policy_effect", "lo", "hi"]]
        j = b.merge(p, on=["training_set", "map", "contrast"])
        j.to_csv(out / "policy_vs_behaviour.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for c, g in j.groupby("contrast"):
            ax.errorbar(g.strategic, g.strategic_policy_effect,
                        yerr=[g.strategic_policy_effect - g.lo, g.hi - g.strategic_policy_effect], fmt="o", ms=4,
                        capsize=2, label=c)
        lim = np.nanmax(np.abs(j[["strategic", "strategic_policy_effect"]].values)) * 1.1 + 1e-3
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("strategic effect on steal_frac (behaviour, exp 1)")
        ax.set_ylabel("strategic effect on P(other coin) share (same states)")
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "policy_vs_behaviour.png", dpi=120); plt.close(fig)
    k = sh[(sh.contrast.str.startswith(partners[0])) & (sh.term.isin(["colour_bias", "threat_own"]))]
    print("\nStrategic change of the focal team's preference weights (trained with new vs reference partner):")
    for _, r in k.sort_values(["training", "term", "contrast", "num_agents"]).iterrows():
        print(f"  {r.training:10s} {r['map']:24s} {r.term:12s} {r.contrast:30s} {r.with_ref:+.3f} -> "
              f"{r.with_new:+.3f}  shift {r['shift']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]")


def exp3(icfg, out):
    partners = icfg["partners"]
    s3, s1 = icfg["experiments"]["exp3"], icfg["experiments"]["exp1"]
    cfg3, cfg1 = load_experiment(s3["experiment"]), load_experiment(s1["experiment"])
    sm3 = seed_metrics(load_crossplay(cfg3["data_root"], s3["training_id"]))
    groups = {(m["grid_size"], m["num_agents"]) for m in cfg3["maps"].values()}
    sm1 = seed_metrics(load_crossplay(cfg1["data_root"], s1["training_id"]))
    sm1 = sm1[(sm1.training == "specialist") & sm1.apply(lambda r: (r.grid_size, r.num_agents) in groups, axis=1)]
    sm = pd.concat([sm1, sm3], ignore_index=True)
    sm.to_csv(out / "seed_metrics.csv", index=False)
    cells = cells_table(sm)
    cells.to_csv(out / "cells.csv", index=False)
    dec = decomposition_table(sm, partners)
    dec["group"] = dec.num_agents.astype(str) + " agents"
    dec.to_csv(out / "decomposition.csv", index=False)
    coadaptation_table(sm, partners).to_csv(out / "coadaptation.csv", index=False)
    fig_decomposition(dec, "coins_per_agent", out, "scarcity", lines="group", xlabel="coins per agent")
    fig_share(dec, "coins_per_agent", out, "scarcity", lines="group", xlabel="coins per agent")
    # probes vs scarcity, if available for both trainings
    files = [Path(c["data_root"]) / "influence" / s["training_id"] / "probe_coefficients.csv"
             for c, s in ((cfg1, s1), (cfg3, s3))]
    if all(f.exists() for f in files):
        coef = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        coef = coef[(coef.training == "specialist") &
                    coef.apply(lambda r: (r.grid_size, r.num_agents) in groups, axis=1)]
        sh = probe_shifts(coef, partners)
        sh["coins_per_agent"] = (sh.num_coins / sh.num_agents).round(3)
        sh["group"] = sh.num_agents.astype(str) + " agents"
        sh.to_csv(out / "probe_shifts.csv", index=False)
        fig_probe_shifts(sh, "coins_per_agent", out, "scarcity", lines="group", xlabel="coins per agent")
    print_key(dec)


def ols_fe(df, y, xs, fe):
    """OLS of y on xs after demeaning within fixed-effect groups; returns coefficients."""
    d = df[[y] + xs + fe].dropna()
    dm = d[[y] + xs] - d.groupby(fe)[[y] + xs].transform("mean")
    return np.linalg.lstsq(dm[xs].values, dm[y].values, rcond=None)[0]


def cluster_boot(df, fn, cluster="focal_seed"):
    ids = df[cluster].unique()
    groups = {i: df[df[cluster] == i] for i in ids}
    pt = fn(df)
    bs = np.array([fn(pd.concat([groups[i] for i in RNG.choice(ids, len(ids))])) for _ in range(500)])
    return pt, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def exp4(icfg, out, root_out):
    """Observational view vs ground truth."""
    partners = icfg["partners"]
    R = partners[0]
    frames = []
    for key in ("exp1", "exp3"):
        spec = icfg["experiments"][key]
        cfg = load_experiment(spec["experiment"])
        try:
            raw = load_crossplay(cfg["data_root"], spec["training_id"])
        except FileNotFoundError:
            print(f"  ({key} cross-play not found, skipped)")
            continue
        f, p = seed_metrics(raw, 0), seed_metrics(raw, 1)
        p = p.rename(columns={m: f"partner_{m}" for m in METRICS + ["opportunity_other"]})
        frames.append(f.merge(p, on=KEYS + ["focal_seed", "coins_per_agent"]))
    sm = pd.concat(frames, ignore_index=True)

    # (a) naive comparison of co-trained pairs vs the true decomposition
    rows = []
    for keys, g in sm.groupby(["training_id", "training_set", "training", "map", "num_agents", "num_coins"]):
        for Ra, P in contrasts(partners):
            for m in METRICS:
                co = {A: g[(g.pairing == "co") & (g.train_partner == A)][m].values for A in (Ra, P)}
                if min(len(v) for v in co.values()) < 2:
                    continue
                naive = np.nanmean(co[P]) - np.nanmean(co[Ra])
                bs = np.nanmean(co[P][RNG.integers(0, len(co[P]), (NB, len(co[P])))], 1) - \
                     np.nanmean(co[Ra][RNG.integers(0, len(co[Ra]), (NB, len(co[Ra])))], 1)
                d = decompose(g, Ra, P, m)
                lo, hi = _ci(bs)
                rows.append({**dict(zip(["training_id", "training_set", "training", "map", "num_agents", "num_coins"],
                                        keys)), "contrast": f"{Ra}->{P}", "metric": m, "observed": naive,
                             "observed_lo": lo, "observed_hi": hi, "strategic": d["strategic"] if d else np.nan,
                             "ecological": d["ecological"] if d else np.nan,
                             "coadaptation_gap": naive - (d["total"] if d else np.nan)})
    nv = pd.DataFrame(rows)
    nv["strategic_fraction_of_observed"] = nv.strategic / nv.observed
    nv.to_csv(out / "observed_vs_true.csv", index=False)

    # (b) peer-effect regressions with map fixed effects (unit = focal seed in a cell)
    regs = []
    for tr, g in sm.groupby("training"):
        for y in ["steal_frac", "other_rate", "own_rate"]:
            x = "partner_steal_frac"
            co = g[g.pairing == "co"]
            eco = g[(g.pairing == "cross") & (g.train_partner == R)]   # focal policy fixed, partner varies
            fe = ["training_id", "training_set", "map"]
            specs = {"observational (co-trained pairs)": (co, [x]),
                     "observational + availability control": (co, [x, "opportunity_other"]),
                     "ecological only (focal policy fixed)": (eco, [x])}
            for name, (d, xs) in specs.items():
                if len(d) < 5:
                    continue
                pt, lo, hi = cluster_boot(d, lambda dd: ols_fe(dd, y, xs, fe))
                regs.append({"training": tr, "y": y, "x": x, "spec": name, "beta": pt[0], "lo": lo[0], "hi": hi[0],
                             "n": len(d)})
    rg = pd.DataFrame(regs)
    rg.to_csv(out / "peer_regressions.csv", index=False)

    # figures
    for m in ["own_rate", "steal_frac", "pure_rate"]:
        d = nv[nv.metric == m]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for c, g in d.groupby("contrast"):
            ax.scatter(g.observed, g.strategic, s=18, label=c)
        lim = np.nanmax(np.abs(d[["observed", "strategic"]].values)) * 1.1 + 1e-3
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, label="observed = strategic")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"observed difference (co-trained pairs): {m}"); ax.set_ylabel("true strategic effect S")
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"observed_vs_strategic_{m}.png", dpi=120); plt.close(fig)
    if not rg.empty:
        fig, axes = plt.subplots(1, rg.training.nunique(), figsize=(6.5 * rg.training.nunique(), 4), squeeze=False)
        for ax, (tr, g) in zip(axes[0], rg.groupby("training")):
            specs = list(g.spec.unique())
            for i, (y, gy) in enumerate(g.groupby("y")):
                gy = gy.set_index("spec").loc[specs]
                xs = np.arange(len(specs)) + i * 0.25
                ax.errorbar(xs, gy.beta, yerr=[gy.beta - gy.lo, gy.hi - gy.beta], fmt="o", capsize=3, label=y)
            ax.set_xticks(np.arange(len(specs)) + 0.25, [s.replace(" (", "\n(") for s in specs], fontsize=7)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_ylabel("peer coefficient on partner's steal_frac"); ax.set_title(tr); ax.grid(alpha=0.3)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out / "peer_regressions.png", dpi=120); plt.close(fig)
    print("\nPeer-effect regressions (focal metric on partner's steal_frac, map fixed effects, 95% cluster CI):")
    for _, r in rg.iterrows():
        print(f"  {r.training:10s} y={r.y:10s} {r.spec:40s} beta {r.beta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]  n={r.n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", default="configs/influence.yaml")
    ap.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    a = ap.parse_args()
    icfg = _yaml(a.influence)
    cfg = load_experiment(icfg["experiments"]["exp1"]["experiment"])
    root_out = Path(cfg["data_root"]) / "influence" / "analysis"
    out = root_out / f"exp{a.exp}"
    out.mkdir(parents=True, exist_ok=True)
    {1: lambda: exp1(icfg, out), 2: lambda: exp2(icfg, out, root_out), 3: lambda: exp3(icfg, out),
     4: lambda: exp4(icfg, out, root_out)}[a.exp]()
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
