#!/usr/bin/env python3
"""Tiny config reader for the k_*.sh experiments.

    python experiments/kcfg.py <influence.yaml> sources                  -> k0 k2 k10
    python experiments/kcfg.py <influence.yaml> <source> experiment|training_id|data_root|k
    python experiments/kcfg.py <influence.yaml> <source> trained         -> exit 0 iff every training dir has summary.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coopcoins.utils import _yaml, load_experiment, payoff_columns, training_dir  # noqa: E402

icfg = _yaml(sys.argv[1])
if sys.argv[2] == "sources":
    print(" ".join(icfg["experiments"]))
    sys.exit(0)
spec, field = icfg["experiments"][sys.argv[2]], sys.argv[3]
if field in spec:
    print(spec[field])
    sys.exit(0)
cfg = load_experiment(spec["experiment"])
if field == "data_root":
    print(cfg["data_root"])
elif field == "k":
    print(f"{payoff_columns(cfg['payoff'])['k']:g}")
elif field == "trained":
    dirs = [training_dir(cfg, t["name"], p, spec["training_id"]) for t in cfg["training_sets"] for p in cfg["pairs"]]
    sys.exit(0 if all((d / "summary.yaml").exists() for d in dirs) else 1)
else:
    sys.exit(f"unknown field {field}")
