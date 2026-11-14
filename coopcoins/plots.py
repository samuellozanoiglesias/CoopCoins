"""Training curves and simulation report figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAM_COLORS = ["#d63031", "#095cba"]


def training_plot(csv_path, png_path, pair, episode_len, smooth=20):
    """training_stats.csv (long: seed, update, episode, map, team, metrics) -> grid of curves."""
    df = pd.read_csv(csv_path)
    metrics = [("own_coins", "own-team coins / agent / game"), ("other_coins", "other-team coins / agent / game"),
               ("pure_reward", "score / agent / game"), ("attitude_reward", "attitude reward / agent / game"),
               ("missed_other", "reachable enemy coins left / agent / game"), ("entropy", "policy entropy (all maps pooled)")]
    maps = list(dict.fromkeys(df["map"]))
    cmap = plt.get_cmap("viridis", max(len(maps), 2))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, (m, title) in zip(axes.flat, metrics):
        for mi, mp in enumerate(maps):
            for k in (0, 1):
                g = df[(df["map"] == mp) & (df.team == k)].sort_values(["seed", "episode"])
                sm = g.groupby("seed")[m].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
                stat = pd.DataFrame({"episode": g.episode, "v": sm}).groupby("episode")["v"]
                mu, sd = stat.mean(), stat.std().fillna(0)
                color = TEAM_COLORS[k] if len(maps) == 1 else cmap(mi)
                ax.plot(mu.index, mu, color=color, ls="-" if k == 0 else "--", lw=1.4,
                        label=(f"{['red', 'blue'][k]}: {pair[k]}" if len(maps) == 1 else (mp if k == 0 else None)))
                if len(maps) == 1:
                    ax.fill_between(mu.index, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("training games per map (episodes)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    style = "" if len(maps) == 1 else f" — solid: red ({pair[0]}), dashed: blue ({pair[1]})"
    fig.suptitle(f"{pair[0]} (red) vs {pair[1]} (blue), {df.seed.nunique()} seeds, {episode_len} steps/game{style}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def simulation_plot(team_df, png_path, pair, map_name):
    """team_df: one row per (train_seed, sim_seed, team), per-agent game totals."""
    metrics = ["own_coins", "other_coins", "missed_other", "pure_reward", "attitude_reward"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.6))
    per_seed = team_df.groupby(["team", "train_seed"])[metrics].mean().reset_index()
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, metrics):
        for k in (0, 1):
            v = per_seed[per_seed.team == k][m].values
            ax.bar(k, v.mean(), color=TEAM_COLORS[k], alpha=0.6)
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, color=TEAM_COLORS[k], s=12, zorder=3)
        ax.set_xticks([0, 1], [f"{pair[0]}\n(red)", f"{pair[1]}\n(blue)"], fontsize=8)
        ax.set_title(m.replace("_", " ") + " / agent / game", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{map_name}: bars = mean, dots = training seeds (each averaged over its games)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
