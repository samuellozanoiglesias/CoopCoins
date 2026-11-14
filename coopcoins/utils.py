"""Experiment config, map families, folder layout, checkpoints."""
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent


def _yaml(path):
    path = Path(path)
    with open(path if path.is_absolute() else CODE_ROOT / path) as f:
        return yaml.safe_load(f)


def map_name(grid_size, num_agents, num_coins=None):
    base = f"grid{grid_size}_agents{num_agents}"
    return base if num_coins in (None, num_agents) else f"{base}_coins{num_coins}"


def make_map(grid_size, num_agents, cfg, num_coins=None):
    """A map: grid size, agents and coins (default coins = agents). Game length and payoff are experiment-wide."""
    m = int(num_coins or num_agents)
    return {"name": map_name(grid_size, num_agents, m), "grid_size": int(grid_size), "num_agents": int(num_agents),
            "num_coins": m, "episode_len": int(cfg["episode_len"]), "payoff": cfg["payoff"],
            "density": round(num_agents / grid_size ** 2, 4), "ratio": round(grid_size / num_agents, 4),
            "coins_per_agent": round(m / num_agents, 4)}


def map_from_name(name, cfg):
    g, n, m = re.fullmatch(r"grid(\d+)_agents(\d+)(?:_coins(\d+))?", name).groups()
    return make_map(int(g), int(n), cfg, int(m) if m else None)


def family_maps(fam, cfg):
    """density:  num_agents = base_agents * (G / base_grid)^2, rounded to an even number
       ratio:    num_agents = base_agents * G / base_grid, must be an even integer
       explicit: maps: [[grid_size, num_agents, num_coins], ...]"""
    if fam["keep"] == "explicit":
        return [make_map(g, n, cfg, m) for g, n, m in fam["maps"]]
    out = []
    for g in fam["grid_sizes"]:
        if fam["keep"] == "density":
            n = 2 * int(round(fam["base_agents"] / 2 * (g / fam["base_grid"]) ** 2))
        elif fam["keep"] == "ratio":
            n = fam["base_agents"] * g / fam["base_grid"]
            assert n == int(n) and int(n) % 2 == 0, f"ratio family: grid {g} gives {n} agents (needs an even integer)"
        else:
            raise ValueError(fam["keep"])
        out.append(make_map(g, int(n), cfg))
    return out


def load_experiment(path):
    cfg = _yaml(path)
    cfg["name"] = cfg.get("name") or Path(path).stem
    cfg["families"] = {k: family_maps(f, cfg) for k, f in cfg["families"].items()}
    cfg["maps"] = {m["name"]: m for fam in cfg["families"].values() for m in fam}       # unique maps
    for m in cfg["maps"].values():
        m["families"] = [k for k, fam in cfg["families"].items() if any(x["name"] == m["name"] for x in fam)]
    sets = []
    if cfg["training"].get("generalist", True):
        sets.append({"name": "all_maps", "maps": list(cfg["maps"].values())})
    if cfg["training"].get("specialists", True):
        sets += [{"name": m["name"], "maps": [m]} for m in cfg["maps"].values()]
    cfg["training_sets"] = sets
    angles = _yaml(cfg["attitudes"])
    cfg["attitude_angles"] = angles
    cfg["attitudes"] = {k: [round(float(np.cos(np.radians(a))), 6), round(float(np.sin(np.radians(a))), 6)]
                        for k, a in angles.items()}
    if isinstance(cfg["seeds"], int):
        cfg["seeds"] = list(range(cfg["seeds"]))
    return cfg


def pair_name(pair):
    return f"{pair[0]}_{pair[1]}"


def training_dir(cfg, set_name, pair, training_id):
    """<data_root>/<training set: all_maps | map name>/<red_blue>/Training_<training_id>"""
    return Path(cfg["data_root"]) / set_name / pair_name(pair) / f"Training_{training_id}"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(path, params, **meta):
    """params: leading axis 2 (teams) -> {'team_0': numpy params, 'team_1': ..., **meta}."""
    import jax
    path.parent.mkdir(parents=True, exist_ok=True)
    teams = {f"team_{t}": jax.tree.map(lambda x: np.asarray(x[t]), params) for t in (0, 1)}
    with open(path, "wb") as f:
        pickle.dump({**teams, **meta}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stack_teams(ckpt):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(lambda a, b: jnp.stack([a, b]), ckpt["team_0"], ckpt["team_1"])
