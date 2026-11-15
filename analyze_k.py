#!/usr/bin/env python3
"""Cross-k analysis of the theft-penalty sweep (docs/experiment_k.md).

    python analyze_k.py --influence configs/influence_k.yaml            # sources: every key under experiments:
    python analyze_k.py --influence configs/influence_k.yaml --sources k0,k2,k10

Inputs: <data_root>/influence/<training_id>/crossplay_games.csv of every source, and training_stats.csv of every
trained pair. Output: <data_root>/influence/analysis/k_sweep/

Definitions as in analyze_influence.py (b[A, B]: focal policy trained with A, playing B, cross-seed pairing;
E, S, I, T, strategic share). For every contrast R->P and metric, per map and pooled:
    pooled = equal-weight mean over maps of the per-map component (= map fixed effects in a balanced panel);
    pooled strategic share = |S_pooled| / (|S_pooled| + |E_pooled|).
Trend in k: OLS slope of the component on k (k in its natural units), plus each k minus the smallest k.
Teacher effect: b[cooperative, B] - b[B, B] for every play partner B != cooperative (pure_rate, coin_rate).

Statistics: one bootstrap over training-seed INDICES shared by every arm, map and cell (B = 2000 draws). This is
valid whether or not arms are independent; note that seed s has the same initial parameters in every arm
(PRNGKey(s)), and that generalist results on different maps share the same seeds (not independent replications).
p-values are two-sided bootstrap p-values, never 0 (reported "<1/B"); q-values are Benjamini-Hochberg over all
p-values of one output table.

Outputs:
    k_decomposition.csv        per (contrast, metric, k, map | pooled): S, E, I, T, share with CIs, p, q
    k_trends.csv               per (contrast, metric, component, map | pooled): slope on k and k - k_min
    k_trend_tests_S.csv        the trend in S for individualistic->competitive and individualistic->cooperative
    k_teacher.csv              teacher effect per (k, play partner, metric, map | pooled) and its slope on k
    k_training_diagnostics.csv last 10% of training: gradient norms, value loss, return statistics by pair and k
    figures: k_decomposition_<contrast>.png, k_strategic_share.png, k_S_by_map.png, k_teacher.png,
             k_training_diagnostics.png, k_training_curves.png
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_influence import METRICS, NB, RNG, _ci, _p, add_q, contrasts, load_crossplay, seed_metrics, write_note
from coopcoins.utils import _yaml, load_experiment, pair_name, training_dir

COMPONENTS = ["strategic", "ecological", "interaction", "total", "strategic_share"]
COL = {"ecological": "#2e8b57", "strategic": "#8e44ad", "total": "#333333"}
DIAG = ["actor_grad_norm", "critic_grad_norm", "vf_loss", "return_mean", "return_std", "entropy", "approx_kl"]


# ------------------------------------------------------------------ data
def load_all(icfg, sources):
    frames, cfgs = [], {}
    for src in sources:
        spec = icfg["experiments"][src]
        cfg = load_experiment(spec["experiment"])
        cfgs[src] = (cfg, spec["training_id"])
        frames.append(seed_metrics(load_crossplay(cfg["data_root"], spec["training_id"])))
    sm = pd.concat(frames, ignore_index=True)
    if sm.groupby("k").training_id.nunique().max() > 1:
        raise ValueError("two sources share the same k")
    return sm, cfgs


def cube(sm, metric, partners, ks, maps, seeds):
    """-> X (K, n_maps, P, P, n_seeds) for the cross pairing; NaN where missing."""
    P = len(partners)
    X = np.full((len(ks), len(maps), P, P, len(seeds)), np.nan)
    d = sm[sm.pairing == "cross"].set_index(["k", "map", "train_partner", "play_partner", "focal_seed"])[metric]
    for i, k in enumerate(ks):
        for j, mp in enumerate(maps):
            for a, A in enumerate(partners):
                for b, B in enumerate(partners):
                    try:
                        X[i, j, a, b] = d.loc[(k, mp, A, B)].reindex(seeds).to_numpy()
                    except KeyError:
                        pass
    return X


def components(M, a, b):
    """M (..., P, P) cell means; a = index of R, b = index of P -> dict of (...)."""
    RR, RP, PR, PP = M[..., a, a], M[..., a, b], M[..., b, a], M[..., b, b]
    E = 0.5 * ((RP - RR) + (PP - PR))
    S = 0.5 * ((PR - RR) + (PP - RP))
    I = (PP - PR) - (RP - RR)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.abs(S) / (np.abs(S) + np.abs(E))
    return {"strategic": S, "ecological": E, "interaction": I, "total": E + S, "strategic_share": share}


def pooled(comp):
    """Map fixed effects in a balanced panel = equal-weight mean over the map axis (-1)."""
    out = {c: np.nanmean(comp[c], -1) for c in ("strategic", "ecological", "interaction", "total")}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["strategic_share"] = np.abs(out["strategic"]) / (np.abs(out["strategic"]) + np.abs(out["ecological"]))
    return out


def slope(y, ks):
    """OLS slope of y (..., K) on k."""
    k = np.asarray(ks, float)
    kc = k - k.mean()
    return ((y - np.nanmean(y, -1, keepdims=True)) * kc).sum(-1) / (kc ** 2).sum()


# ------------------------------------------------------------------ analyses
def decomposition_vs_k(sm, partners, ks, maps, seeds):
    n = len(seeds)
    idx_b = RNG.integers(0, n, (NB, n))
    idx_0 = np.arange(n)
    dec_rows, trend_rows = [], []
    for metric in METRICS:
        X = cube(sm, metric, partners, ks, maps, seeds)                 # (K, Mp, P, P, n)
        M0 = np.nanmean(X, -1)                                          # (K, Mp, P, P)
        Mb = np.stack([np.nanmean(X[..., ib], -1) for ib in idx_b])     # (NB, K, Mp, P, P)
        for R, P in contrasts(partners):
            a, b = partners.index(R), partners.index(P)
            c0, cb = components(M0, a, b), components(Mb, a, b)         # (K, Mp), (NB, K, Mp)
            p0, pb = pooled(c0), pooled(cb)                             # (K,), (NB, K)
            scopes = [(mp, {c: c0[c][:, j] for c in c0}, {c: cb[c][:, :, j] for c in cb}) for j, mp in enumerate(maps)]
            scopes.append(("pooled", p0, pb))
            for mp, pt, bs in scopes:
                for i, k in enumerate(ks):
                    row = {"contrast": f"{R}->{P}", "metric": metric, "k": k, "map": mp, "n_seeds": n}
                    for c in COMPONENTS:
                        lo, hi = _ci(bs[c][:, i])
                        row.update({c: pt[c][i], f"{c}_lo": lo, f"{c}_hi": hi})
                        if c != "strategic_share":
                            row[f"{c}_p"] = _p(bs[c][:, i])
                    dec_rows.append(row)
                for c in COMPONENTS:
                    s0, sb = slope(pt[c], ks), slope(bs[c], ks)
                    lo, hi = _ci(sb)
                    tr = {"contrast": f"{R}->{P}", "metric": metric, "component": c, "map": mp,
                          "slope_per_unit_k": s0, "slope_lo": lo, "slope_hi": hi, "slope_p": _p(sb)}
                    for i, k in enumerate(ks[1:], 1):
                        d0, db = pt[c][i] - pt[c][0], bs[c][:, i] - bs[c][:, 0]
                        dlo, dhi = _ci(db)
                        tr.update({f"k{k:g}_minus_k{ks[0]:g}": d0, f"k{k:g}_minus_k{ks[0]:g}_lo": dlo,
                                   f"k{k:g}_minus_k{ks[0]:g}_hi": dhi, f"k{k:g}_minus_k{ks[0]:g}_p": _p(db)})
                    trend_rows.append(tr)
    return add_q(pd.DataFrame(dec_rows)), add_q(pd.DataFrame(trend_rows))


def teacher_vs_k(sm, partners, ks, maps, seeds, teacher="cooperative"):
    if teacher not in partners:
        return pd.DataFrame()
    n = len(seeds)
    idx_b = RNG.integers(0, n, (NB, n))
    t = partners.index(teacher)
    rows = []
    for metric in ("pure_rate", "coin_rate"):
        X = cube(sm, metric, partners, ks, maps, seeds)
        M0 = np.nanmean(X, -1)
        Mb = np.stack([np.nanmean(X[..., ib], -1) for ib in idx_b])
        for B in partners:
            if B == teacher:
                continue
            b = partners.index(B)
            e0, eb = M0[:, :, t, b] - M0[:, :, b, b], Mb[:, :, :, t, b] - Mb[:, :, :, b, b]   # (K, Mp), (NB, K, Mp)
            scopes = [(mp, e0[:, j], eb[:, :, j]) for j, mp in enumerate(maps)]
            scopes.append(("pooled", np.nanmean(e0, -1), np.nanmean(eb, -1)))
            for mp, pt, bs in scopes:
                for i, k in enumerate(ks):
                    lo, hi = _ci(bs[:, i])
                    rows.append({"metric": metric, "play_partner": B, "map": mp, "k": k,
                                 "teacher_effect": pt[i], "lo": lo, "hi": hi, "p": _p(bs[:, i]), "kind": "level"})
                s0, sb = slope(pt, ks), slope(bs, ks)
                lo, hi = _ci(sb)
                rows.append({"metric": metric, "play_partner": B, "map": mp, "k": np.nan, "teacher_effect": s0,
                             "lo": lo, "hi": hi, "p": _p(sb), "kind": "slope_per_unit_k"})
    return add_q(pd.DataFrame(rows))


def training_diagnostics(cfgs):
    rows, curves = [], []
    for src, (cfg, tid) in cfgs.items():
        for tset in cfg["training_sets"]:
            for pair in cfg["pairs"]:
                f = training_dir(cfg, tset["name"], pair, tid) / "training_stats.csv"
                if not f.exists():
                    print(f"  (missing {f})")
                    continue
                df = pd.read_csv(f)
                df["pair"], df["training_set"] = pair_name(pair), tset["name"]
                curves.append(df)
                last = df[df.episode > 0.9 * df.episode.max()]
                # optimiser quantities are per team (identical across maps in a row block): average over maps
                per_seed = last.groupby(["k", "pair", "training_set", "team", "attitude", "seed"])[DIAG].mean()
                for keys, g in per_seed.groupby(level=[0, 1, 2, 3, 4]):
                    r = dict(zip(["k", "pair", "training_set", "team", "attitude"], keys))
                    for c in DIAG:
                        v = g[c].to_numpy()
                        lo, hi = _ci(v[RNG.integers(0, len(v), (NB, len(v)))].mean(1))
                        r.update({c: v.mean(), f"{c}_lo": lo, f"{c}_hi": hi})
                    rows.append(r)
    return pd.DataFrame(rows), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


# ------------------------------------------------------------------ figures
def _err(ax, x, y, lo, hi, **kw):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    ax.errorbar(x, y, yerr=[np.abs(y - lo), np.abs(hi - y)], capsize=3, marker="o", **kw)


def fig_decomposition(dec, ks, out):
    pos = {k: i for i, k in enumerate(ks)}
    for con, dc in dec[dec["map"] == "pooled"].groupby("contrast"):
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
        for ax, m in zip(axes.flat, METRICS):
            d = dc[dc.metric == m].sort_values("k")
            x = d.k.map(pos).to_numpy(float)
            for j, c in enumerate(("ecological", "strategic", "total")):
                _err(ax, x + (j - 1) * 0.06, d[c], d[f"{c}_lo"], d[f"{c}_hi"], color=COL[c], ms=4, label=c)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(range(len(ks)), [f"k={k:g}" for k in ks])
            ax.set_title(m, fontsize=9); ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{con}: decomposition vs theft penalty k (pooled over maps, 95% CI over seeds)")
        fig.tight_layout(); fig.savefig(out / f"k_decomposition_{con.replace('->', '_to_')}.png", dpi=120)
        plt.close(fig)


def fig_share(dec, ks, out):
    pos = {k: i for i, k in enumerate(ks)}
    d = dec[(dec["map"] == "pooled")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, m in zip(axes.flat, METRICS):
        for i, (con, g) in enumerate(d[d.metric == m].groupby("contrast")):
            g = g.sort_values("k")
            _err(ax, g.k.map(pos) + (i - 1) * 0.06, g.strategic_share, g.strategic_share_lo, g.strategic_share_hi,
                 ms=4, label=con)
        ax.set_ylim(-0.02, 1.02); ax.set_xticks(range(len(ks)), [f"k={k:g}" for k in ks])
        ax.set_title(f"strategic share: {m}", fontsize=9); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "k_strategic_share.png", dpi=120); plt.close(fig)


def fig_s_by_map(dec, ks, focal, out):
    d = dec[(dec["map"] != "pooled") & dec.contrast.str.startswith(f"{focal}->")]
    cons = sorted(d.contrast.unique())
    mets = ["own_rate", "other_rate", "steal_frac", "pure_rate", "defence_rate", "contest_rate"]
    fig, axes = plt.subplots(len(cons), len(mets), figsize=(3.6 * len(mets), 3.2 * len(cons)), squeeze=False)
    cmap = plt.get_cmap("viridis", max(d["map"].nunique(), 2))
    order = d.drop_duplicates("map").sort_values("map", key=lambda s: s.str.extract(r"grid(\d+)")[0].astype(int))
    for i, con in enumerate(cons):
        for j, m in enumerate(mets):
            ax = axes[i, j]
            for c, mp in enumerate(order["map"]):
                g = d[(d.contrast == con) & (d.metric == m) & (d["map"] == mp)].sort_values("k")
                ax.plot(range(len(g)), g.strategic, marker="o", ms=3, color=cmap(c), label=mp)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks]); ax.set_xlabel("k")
            ax.set_title(f"S {m}: {con}", fontsize=8); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Strategic component S vs k, per map (same seeds on every map: not independent)")
    fig.tight_layout(); fig.savefig(out / "k_S_by_map.png", dpi=120); plt.close(fig)


def fig_teacher(te, ks, out):
    if te.empty:
        return
    pos = {k: i for i, k in enumerate(ks)}
    d = te[(te.kind == "level") & (te["map"] == "pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, ("pure_rate", "coin_rate")):
        for i, (B, g) in enumerate(d[d.metric == m].groupby("play_partner")):
            g = g.sort_values("k")
            _err(ax, g.k.map(pos) + (i - 0.5) * 0.06, g.teacher_effect, g.lo, g.hi, ms=4, label=f"plays {B}")
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks(range(len(ks)), [f"k={k:g}" for k in ks])
        ax.set_title(f"teacher effect on {m}: b[coop, B] - b[B, B]", fontsize=9); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "k_teacher.png", dpi=120); plt.close(fig)


def fig_diagnostics(diag, curves, out):
    if diag.empty:
        return
    cols = ["actor_grad_norm", "critic_grad_norm", "vf_loss", "return_mean", "return_std", "entropy"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ks = sorted(diag.k.unique())
    for ax, c in zip(axes.flat, cols):
        for i, ((pair, team), g) in enumerate(diag.groupby(["pair", "team"])):
            g = g.sort_values("k")
            _err(ax, np.arange(len(g)) + (i - 2.5) * 0.04, g[c], g[f"{c}_lo"], g[f"{c}_hi"], ms=3,
                 label=f"{pair} team {team}")
        ax.set_xticks(range(len(ks)), [f"k={k:g}" for k in ks]); ax.set_title(f"{c} (last 10%)"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "k_training_diagnostics.png", dpi=120); plt.close(fig)
    if curves.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    per = curves.groupby(["k", "pair", "team", "episode"])[cols].mean().reset_index()
    cmap = plt.get_cmap("plasma", len(ks) + 1)
    for ax, c in zip(axes.flat, cols):
        for (k, pair, team), g in per.groupby(["k", "pair", "team"]):
            ax.plot(g.episode, g[c], color=cmap(ks.index(k)), lw=0.8, alpha=0.8,
                    ls=["-", "--", ":"][sorted(per.pair.unique()).index(pair) % 3],
                    label=f"k={k:g} {pair} t{team}" if c == cols[0] else None)
        ax.set_title(c); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=5, frameon=False, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("training games per map")
    fig.tight_layout(); fig.savefig(out / "k_training_curves.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", default="configs/influence_k.yaml")
    ap.add_argument("--sources", default=None, help="comma list of experiment keys (default: all)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    icfg = _yaml(a.influence)
    sources = a.sources.split(",") if a.sources else list(icfg["experiments"])
    partners, focal = icfg["partners"], icfg["focal"]

    sm, cfgs = load_all(icfg, sources)
    cfg0 = next(iter(cfgs.values()))[0]
    out = Path(cfg0["data_root"]) / "influence" / "analysis" / "k_sweep"
    out.mkdir(parents=True, exist_ok=True)
    sm.to_csv(out / "k_seed_metrics.csv", index=False)

    ks = sorted(sm.k.unique())
    frames = []
    for tr, g in sm.groupby("training"):
        maps = sorted(g["map"].unique(), key=lambda m: (g[g["map"] == m].grid_size.iloc[0], m))
        seeds = sorted(g.focal_seed.unique())
        dec, trends = decomposition_vs_k(g, partners, ks, maps, seeds)
        te = teacher_vs_k(g, partners, ks, maps, seeds)
        for df in (dec, trends, te):
            if not df.empty:
                df.insert(0, "training", tr)
        frames.append((dec, trends, te))
    dec = pd.concat([f[0] for f in frames], ignore_index=True)
    trends = pd.concat([f[1] for f in frames], ignore_index=True)
    te = pd.concat([f[2] for f in frames], ignore_index=True)
    dec.to_csv(out / "k_decomposition.csv", index=False)
    trends.to_csv(out / "k_trends.csv", index=False)
    te.to_csv(out / "k_teacher.csv", index=False)

    # the pre-specified tests: trend in S for focal->competitive and focal->cooperative, BH within this table
    keep = [f"{focal}->competitive", f"{focal}->cooperative"]
    tS = trends[(trends.component == "strategic") & trends.contrast.isin(keep)].copy()
    tS = tS.drop(columns=[c for c in tS.columns if c.endswith("_q") or c.endswith("_report")])
    add_q(tS).to_csv(out / "k_trend_tests_S.csv", index=False)

    diag, curves = training_diagnostics(cfgs)
    diag.to_csv(out / "k_training_diagnostics.csv", index=False)

    fig_decomposition(dec, ks, out)
    fig_share(dec, ks, out)
    fig_s_by_map(dec, ks, focal, out)
    fig_teacher(te, ks, out)
    fig_diagnostics(diag, curves, out)
    write_note(out)

    print(f"\nTrend of the strategic component S with k (pooled over maps, slope per unit k, 95% CI, BH q):")
    t = add_q(tS)
    for _, r in t[t["map"] == "pooled"].sort_values(["contrast", "metric"]).iterrows():
        print(f"  {r.contrast:34s} {r.metric:13s} slope {r.slope_per_unit_k:+.5f} [{r.slope_lo:+.5f},"
              f"{r.slope_hi:+.5f}]  p {r.slope_p_report:>8s}  q {r.slope_q:.3g}")
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
