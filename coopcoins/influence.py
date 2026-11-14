"""Tools for the social-influence study: cross-play, synthetic and state-matched policy probes.

Notation. The focal team (red, team 0) has the focal attitude (individualistic). b[A, B] is a behaviour of the
focal team when its policy was TRAINED with a partner of attitude A and it PLAYS with a partner of attitude B.
Every cell uses cross-seed pairing (focal seed s with partner seed s + shift), so no cell plays with the exact
partner it was trained with; the co-trained pairs (same run) are kept separately as a control.

For a reference partner R and a new partner P, the total influence b[P, P] - b[R, R] splits (Shapley) into
    ecological  E = 1/2 [(b[R, P] - b[R, R]) + (b[P, P] - b[P, R])]   same policy, different partner
    strategic   S = 1/2 [(b[P, R] - b[R, R]) + (b[P, P] - b[R, P])]   same partner, different policy
    interaction I = (b[P, P] - b[P, R]) - (b[R, P] - b[R, R])
with E + S = total exactly.
"""
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .env import FEATURES, NO_TEAMMATE, STEP_SCALE, make_env
from .networks import policy_logits
from .ppo import per_agent
from .simulation import make_simulator
from .utils import load_checkpoint, load_yaml, training_dir

F = {f: i for i, f in enumerate(FEATURES)}


# ------------------------------------------------------------------ loading trained policies
def last_episode(d):
    per_seed = [{int(re.findall(r"\d+", p.stem)[0]) for p in s.glob("episode_*.pkl")}
                for s in (Path(d) / "checkpoints").glob("seed_*")]
    return max(set.intersection(*per_seed))


def load_team(d, team, episode=None):
    """Parameters of one team for every training seed of a training dir -> (params (S, ...), seeds, episode)."""
    d = Path(d)
    episode = episode or last_episode(d)
    seed_dirs = sorted((d / "checkpoints").glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    ckpts = [load_checkpoint(s / f"episode_{episode}.pkl") for s in seed_dirs]
    params = jax.tree.map(lambda *x: jnp.stack(x), *[c[f"team_{team}"] for c in ckpts])
    return params, [int(c["seed"]) for c in ckpts], episode


def partner_dirs(cfg, set_name, focal, partners, training_id):
    """{partner attitude: training dir of the pair [focal, partner]} for one training set."""
    out = {}
    for p in partners:
        d = training_dir(cfg, set_name, [focal, p], training_id)
        if not (d / "checkpoints").exists():
            raise FileNotFoundError(f"missing training {d}")
        out[p] = d
    return out


def set_maps(d):
    return load_yaml(Path(d) / "config.yaml")["maps"]


def combine(focal, partner, shift):
    """(S, ...) x (S, ...) -> (S, 2, ...): team 0 = focal seed s, team 1 = partner seed (s + shift) % S."""
    return jax.tree.map(lambda a, b: jnp.stack([a, jnp.roll(b, -shift, axis=0)], axis=1), focal, partner)


def sim_keys(n):
    return jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), 7919))(jnp.arange(n))


_CACHE = {}


def env_tools(m):
    """Compiled env + simulator + state collector per map (cached)."""
    if m["name"] not in _CACHE:
        env = make_env(**m)
        _CACHE[m["name"]] = (env, make_simulator(env), make_collector(env, team=0))
    return _CACHE[m["name"]]


# ------------------------------------------------------------------ state-matched probes
def make_collector(env, team=0):
    """(params (R, 2, ...), keys (S,)) -> observations of the given team's agents at every step:
    {'coins': (R, S, T, n, P, F), 'mask': (R, S, T, n, P)}."""
    idx = np.where(env.agent_team == team)[0]

    def play(params, key):
        k0, key = jax.random.split(key)
        obs, state = env.reset(k0)

        def step(carry, _):
            state, obs, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            act = jax.random.categorical(k_act, per_agent(policy_logits, params, env.agent_team, obs))
            obs2, state2, *_ = env.step(k_env, state, act)
            return (state2, obs2, key), jax.tree.map(lambda x: x[idx], obs)

        _, traj = jax.lax.scan(step, (state, obs, key), None, env.episode_len)
        return traj

    return jax.jit(jax.vmap(jax.vmap(play, (None, 0)), (0, None)))


def state_pool(traj, size, rng):
    """Flatten collected observations to individual decision states and subsample `size` of them."""
    coins = np.asarray(traj["coins"]).reshape(-1, *traj["coins"].shape[-2:])
    mask = np.asarray(traj["mask"]).reshape(-1, traj["mask"].shape[-1])
    pick = rng.choice(len(coins), min(size, len(coins)), replace=False)
    return coins[pick], mask[pick]


@jax.jit
def _probs(params, coins, mask):
    """params (S, ...), coins (K, P, F), mask (K, P) -> action probabilities (S, K, P + 1)."""
    f = lambda p: jax.vmap(lambda c, m: jax.nn.softmax(policy_logits(p, {"coins": c, "mask": m})))(coins, mask)
    return jax.vmap(f)(params)


def choice_probs(params, coins, mask, batch=2048):
    """-> p_own, p_other, p_stay, each (S, K): probability of heading to an own-team coin, an other-team coin,
    or staying."""
    outs = []
    for i in range(0, len(coins), batch):
        c, m = jnp.asarray(coins[i:i + batch]), jnp.asarray(mask[i:i + batch])
        pr = np.asarray(_probs(params, c, m))
        own = (np.asarray(c[..., F["own_team_coin"]]) > 0.5) & np.asarray(m)
        other = (~own) & np.asarray(m)
        outs.append(((pr[..., :-1] * own[None]).sum(-1), (pr[..., :-1] * other[None]).sum(-1), pr[..., -1]))
    return tuple(np.concatenate([o[k] for o in outs], axis=1) for k in range(3))


# ------------------------------------------------------------------ synthetic probes
def max_dist(grid_size):
    return 2 * (grid_size // 2)


def probe_grid(grid_size, num_agents):
    """Synthetic decision states with exactly two coins: coin 0 = own team's, coin 1 = other team's.
    Factors: my distance to each coin, nearest opponent's distance to each coin, whether an opponent chose each
    coin last step. Teammates are neutral (absent with 2 agents; farthest possible and not targeting otherwise).
    Returns coins (K, 2, F), mask (K, 2), factors (dict of (K,) arrays)."""
    md = max_dist(grid_size)
    my_levels = list(range(1, min(4, md) + 1))
    opp_levels = sorted({x for x in (1, 2, 3, md) if 1 <= x <= md})
    has_tm = num_agents > 2
    tm_dist = md if has_tm else NO_TEAMMATE
    rows = []
    for d_own in my_levels:
        for d_other in my_levels:
            for o_own in opp_levels:
                for o_other in opp_levels:
                    if num_agents == 2 and abs(o_own - o_other) > d_own + d_other:
                        continue  # one opponent: its two distances must be compatible with the coins' positions
                    for t_own in (0, 1):
                        for t_other in (0, 1):
                            rows.append((d_own, d_other, o_own, o_other, t_own, t_other))
    f = np.array(rows, dtype=np.float32)
    K = len(f)
    coins = np.zeros((K, 2, len(FEATURES)), np.float32)
    coins[:, 0, F["own_team_coin"]] = 1.0
    for c, (dc, oc, tc) in enumerate([(0, 2, 4), (1, 3, 5)]):
        coins[:, c, F["my_dist"]] = f[:, dc] * STEP_SCALE
        coins[:, c, F["opponent_dist"]] = f[:, oc] * STEP_SCALE
        coins[:, c, F["opponents_targeting"]] = f[:, tc]
        coins[:, c, F["teammate_dist"]] = tm_dist * STEP_SCALE
        coins[:, c, F["has_teammates"]] = float(has_tm)
    factors = {"d_own": f[:, 0], "d_other": f[:, 1], "opp_own": f[:, 2], "opp_other": f[:, 3],
               "tgt_own": f[:, 4], "tgt_other": f[:, 5]}
    return coins, np.ones((K, 2), bool), factors


PROBE_TERMS = ["colour_bias", "distance", "threat_own", "threat_other", "target_own", "target_other"]


def probe_design(factors):
    """Regressors for log P(own coin) - log P(other coin):
    colour_bias   intercept: preference for own over other coin when everything else is equal
    distance      d_other - d_own (steps): how much being closer to the own coin matters
    threat_own    1 if an opponent reaches my own coin no later than me  (> 0 = defending my coin)
    threat_other  1 if an opponent reaches the other coin no later than me (< 0 = avoiding contests)
    target_own    an opponent chose my coin last step
    target_other  an opponent chose the other coin last step"""
    return np.column_stack([np.ones_like(factors["d_own"]), factors["d_other"] - factors["d_own"],
                            (factors["opp_own"] <= factors["d_own"]).astype(np.float32),
                            (factors["opp_other"] <= factors["d_other"]).astype(np.float32),
                            factors["tgt_own"], factors["tgt_other"]])


def probe_policy(params, grid_size, num_agents):
    """-> per seed regression coefficients (S, 6), mean P(stay) (S,), and per state probabilities."""
    coins, mask, factors = probe_grid(grid_size, num_agents)
    pr = np.asarray(_probs(params, jnp.asarray(coins), jnp.asarray(mask)))  # (S, K, 3)
    lo = np.log(pr[..., 0] + 1e-9) - np.log(pr[..., 1] + 1e-9)
    X = probe_design(factors)
    beta = np.linalg.lstsq(X, lo.T, rcond=None)[0].T                        # (S, 6)
    return beta, pr[..., 2].mean(1), pr, factors

