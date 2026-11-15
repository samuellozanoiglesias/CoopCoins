"""Probe helpers: pseudo-replication ids and memory override."""
import numpy as np
import pandas as pd

from analyze_influence import dedup_pooled
from coopcoins.env import MEMORY, N_BASE
from coopcoins.influence import probe_grid, probe_state_set, with_memory


def test_probe_state_sets_and_identical_baseline_states():
    density = [(3, 2), (4, 4), (5, 6), (6, 8), (7, 10), (8, 14), (9, 18), (10, 22)]
    ids = [probe_state_set(g, n) for g, n in density]
    assert ids[1] == ids[2] and ids[3] == ids[4] and ids[5] == ids[6]
    assert len(set(ids)) == 5
    a, _, _ = probe_grid(4, 4)
    b, _, _ = probe_grid(5, 6)
    assert np.array_equal(a, b)                       # generalist sees identical states: pseudo-replication


def test_memory_override():
    c, m, _ = probe_grid(6, 8, memory=[0.1, 0.2, 0.3, 0.4])
    assert np.allclose(c[..., N_BASE:], [0.1, 0.2, 0.3, 0.4])
    base, _, _ = probe_grid(6, 8)
    assert np.array_equal(c[..., :N_BASE], base[..., :N_BASE]) and np.all(base[..., N_BASE:] == 0)
    mask = np.array([[True, False]] * len(c))
    c2 = with_memory(c, mask, np.zeros(len(MEMORY)) + 0.5)
    assert np.all(c2[:, 0, N_BASE:] == 0.5) and np.all(c2[:, 1, N_BASE:] == 0)


def test_dedup_pooled_drops_pseudo_replicates():
    rows = []
    for mp, pset, pseudo, v in [("grid4_agents4", "s4", False, 1.0), ("grid5_agents6", "s4", True, 1.0),
                                ("grid6_agents8", "s6", False, 3.0)]:
        for seed in (0, 1):
            rows.append({"training_id": "t", "training_set": "all_maps", "training": "generalist", "pair": "p",
                         "team": 0, "attitude": "a", "trained_with": "b", "seed": seed, "map": mp,
                         "memory_source": "zero", "probe_state_set": pset, "pseudo_replicate": pseudo,
                         "colour_bias": v, "distance": v, "threat_own": v, "threat_other": v, "target_own": v,
                         "target_other": v, "p_stay": 0.1})
    d = dedup_pooled(pd.DataFrame(rows))
    assert len(d) == 2 and np.allclose(d.colour_bias, 2.0) and np.all(d.n_state_sets == 2)
