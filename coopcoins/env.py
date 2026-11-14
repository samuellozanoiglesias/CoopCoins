"""CoopCoins with coin-targeting actions: the same agents can play on ANY map.

Agents (team i % 2: 0 = red, 1 = blue) and coins (team c % 2) live on a G x G torus.
Each step every agent picks a coin (or "stay"); the environment moves it one cell along the
shortest toroidal path to that coin (same deterministic rule for everyone: the longer axis first).

Observation of agent i = one feature row per coin (raw step counts, identical meaning on every map):
    own_team_coin       1 if the coin belongs to my team
    my_dist             my distance to the coin (toroidal Manhattan, in steps)
    teammate_dist       distance of my nearest teammate to the coin (NO_TEAMMATE if I have none)
    opponent_dist       distance of the nearest opponent to the coin
    teammates_targeting how many teammates chose this coin last step
    opponents_targeting how many opponents chose this coin last step
    has_teammates       1 if my team has more than one agent
Rows are padded to `max_coins` with a mask, so batches can mix maps.
Actions: 0..max_coins-1 = go to that coin, max_coins = stay.

Pure rewards per step: own-team coin +own to the taker; other-team coin +other to the taker and
`penalty / team_size` to every member of the robbed team (2 agents: exactly +1 / -2).
Attitude (SVO) reward, in-group always cooperative:
    r_i = alpha_t * mean(pure of my team) + beta_t * mean(pure of the other team)
"""
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

STEP_SCALE = 0.1      # distances enter the network as steps * 0.1 (same constant on every map)
NO_TEAMMATE = 20.0    # distance used for "teammate" when the team has a single agent
FEATURES = ["own_team_coin", "my_dist", "teammate_dist", "opponent_dist",
            "teammates_targeting", "opponents_targeting", "has_teammates"]
N_FEATURES = len(FEATURES)
STATS = ["own_coins", "other_coins", "missed_own", "missed_other"]
_ADJ = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]], dtype=jnp.int32)


class State(NamedTuple):
    agent_pos: jnp.ndarray  # (N, 2) [row, col]
    coin_pos: jnp.ndarray   # (M, 2)
    target: jnp.ndarray     # (N,) coin chosen last step, -1 = stay / none


class Env(NamedTuple):
    reset: Callable
    step: Callable
    attitude_reward: Callable
    name: str
    grid_size: int
    num_agents: int
    num_coins: int
    max_coins: int
    episode_len: int
    agent_team: np.ndarray
    coin_team: np.ndarray


def make_env(grid_size, num_agents, num_coins=None, episode_len=150, payoff=None, max_coins=None, name=None, **_):
    payoff = payoff or {"own": 1.0, "other": 1.0, "penalty": -2.0}
    G, N = int(grid_size), int(num_agents)
    M = int(num_coins or N)
    P = int(max_coins or M)
    assert N % 2 == 0 and M % 2 == 0, "two equal teams: num_agents and num_coins must be even"
    assert G * G >= N + 2 * M, "grid too small for agents + coins (+ respawn room)"
    assert P >= M
    n_cells, team_size = G * G, N // 2
    agent_team, coin_team = np.arange(N) % 2, np.arange(M) % 2
    a_team, c_team = jnp.asarray(agent_team), jnp.asarray(coin_team)
    same = agent_team[:, None] == agent_team[None]
    tm_mask = jnp.asarray(same & ~np.eye(N, dtype=bool))   # teammates (not me)
    opp_mask = jnp.asarray(~same)
    mine = jnp.asarray(agent_team[:, None] == coin_team[None])  # (N, M)
    p_own, p_other, p_pen = float(payoff["own"]), float(payoff["other"]), float(payoff["penalty"])

    def cell(p):
        return p[..., 0] * G + p[..., 1]

    def free_cells(key, occupied, k):
        """k distinct random free cells (Gumbel top-k)."""
        g = jnp.where(occupied, -jnp.inf, jax.random.gumbel(key, (n_cells,)))
        c = jax.lax.top_k(g, k)[1]
        return jnp.stack([c // G, c % G], -1).astype(jnp.int32)

    def dist(a, b):
        d = jnp.abs(a - b)
        return jnp.minimum(d, G - d).sum(-1)

    def get_obs(s):
        D = dist(s.agent_pos[:, None], s.coin_pos[None]).astype(jnp.float32)          # (N, M)
        if team_size > 1:
            d_tm = jnp.min(jnp.where(tm_mask[:, :, None], D[None], jnp.inf), axis=1)
        else:
            d_tm = jnp.full((N, M), NO_TEAMMATE)
        d_opp = jnp.min(jnp.where(opp_mask[:, :, None], D[None], jnp.inf), axis=1)
        tgt = jax.nn.one_hot(s.target, M)                                               # -1 -> zeros
        feats = jnp.stack([mine.astype(jnp.float32), D * STEP_SCALE, d_tm * STEP_SCALE, d_opp * STEP_SCALE,
                           tm_mask.astype(jnp.float32) @ tgt, opp_mask.astype(jnp.float32) @ tgt,
                           jnp.full((N, M), float(team_size > 1))], -1)                 # (N, M, F)
        feats = jnp.pad(feats, ((0, 0), (0, P - M), (0, 0)))
        mask = jnp.broadcast_to(jnp.arange(P) < M, (N, P))
        return {"coins": feats, "mask": mask}

    def reset(key):
        k1, k2 = jax.random.split(key)
        agent_pos = jax.random.randint(k1, (N, 2), 0, G, dtype=jnp.int32)
        coin_pos = free_cells(k2, jnp.zeros(n_cells, bool).at[cell(agent_pos)].set(True), M)
        s = State(agent_pos, coin_pos, jnp.full(N, -1, jnp.int32))
        return get_obs(s), s

    def step(key, s, actions):
        """actions (N,) in [0, max_coins]; >= num_coins means stay.
        Returns obs, state, pure reward (N,), info {stat: (N,)}, taken (M,)."""
        stay = actions >= M
        target = jnp.clip(actions, 0, M - 1)
        d = (s.coin_pos[target] - s.agent_pos + G // 2) % G - G // 2                    # signed shortest displacement
        row_first = jnp.abs(d[:, 0]) >= jnp.abs(d[:, 1])
        move = jnp.where(row_first[:, None], jnp.stack([jnp.sign(d[:, 0]), jnp.zeros(N, jnp.int32)], -1),
                         jnp.stack([jnp.zeros(N, jnp.int32), jnp.sign(d[:, 1])], -1))
        new_pos = (s.agent_pos + jnp.where(stay[:, None], 0, move)) % G

        on = jnp.all(new_pos[:, None] == s.coin_pos[None], -1)                         # (N, M)
        own, other = (on & mine).sum(1), (on & ~mine).sum(1)
        hits = jnp.stack([(on & ~mine & (c_team == t)).sum() for t in (0, 1)])         # coins lost per team
        pure = p_own * own + p_other * other + p_pen * hits[a_team] / team_size

        adj = (s.agent_pos[:, None] + _ADJ[None]) % G                                    # cells reachable in 1 step
        reach = jnp.any(jnp.all(adj[:, :, None] == s.coin_pos[None, None], -1), 1)       # (N, M)
        info = {"own_coins": own, "other_coins": other,
                "missed_own": (reach & mine & ~on).sum(1), "missed_other": (reach & ~mine & ~on).sum(1)}

        taken = jnp.any(on, axis=0)
        occ = jnp.zeros(n_cells, bool).at[cell(new_pos)].set(True).at[cell(s.coin_pos)].set(True)
        coin_pos = jnp.where(taken[:, None], free_cells(key, occ, M), s.coin_pos)
        # remember targets (for the "targeting" features), forgetting coins that were just taken
        s = State(new_pos, coin_pos, jnp.where(stay | taken[target], -1, target).astype(jnp.int32))
        return get_obs(s), s, pure, info, taken

    team_onehot = jax.nn.one_hot(a_team, 2)

    def attitude_reward(pure, coef):
        """pure (..., N); coef (2, 2): row t = [alpha, beta] of team t."""
        team_mean = pure @ team_onehot / team_size                                      # (..., 2)
        return coef[a_team, 0] * team_mean[..., a_team] + coef[a_team, 1] * team_mean[..., 1 - a_team]

    return Env(reset, step, attitude_reward, name or f"grid{G}_agents{N}", G, N, M, P, int(episode_len),
               agent_team, coin_team)
