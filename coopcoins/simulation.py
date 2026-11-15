"""Games with trained team policies (no learning), same length as in training.

Per-agent game totals (SIM_STATS): pure / attitude reward, the env STATS, and policy-level counts computed on every
decision state of the game:
    defence_opps / defence_choices  own coins that an opponent targeted last step and that I reach no later than
                                    the nearest opponent (my_dist <= opponent_dist) / how many of them I chose
    contest_opps / contest_choices  the same for other-team coins (targeted by their owners, the opponents)
MEM_STATS are per-step MEANS (not totals) of the memory features the agent observed when deciding.
"""
import jax
import jax.numpy as jnp

from .env import FEATURES, MEMORY, N_BASE, STATS
from .networks import policy_logits
from .ppo import per_agent

_F = {f: i for i, f in enumerate(FEATURES)}
CHOICE_STATS = ["defence_opps", "defence_choices", "contest_opps", "contest_choices"]
MEM_STATS = list(MEMORY)
SIM_STATS = ["pure_reward", "attitude_reward", *STATS, *CHOICE_STATS, *MEM_STATS]


def choice_counts(obs, act):
    """obs {'coins': (..., P, F), 'mask': (..., P)}, act (...) -> (4, ...) defence/contest opportunities, choices."""
    c, mask = obs["coins"], obs["mask"]
    own = (c[..., _F["own_team_coin"]] > 0.5) & mask
    enemy = (~own) & mask
    hot = (c[..., _F["opponents_targeting"]] > 0.5) & (c[..., _F["my_dist"]] <= c[..., _F["opponent_dist"]])
    chose = jax.nn.one_hot(act, c.shape[-2] + 1)[..., :-1] > 0.5
    d, e = own & hot, enemy & hot
    return jnp.stack([d.sum(-1), (d & chose).sum(-1), e.sum(-1), (e & chose).sum(-1)]).astype(jnp.float32)


def make_player(env, deterministic=False, record=False):
    """play(params (2 teams, ...), coef (2, 2), key) -> per-agent totals (len(SIM_STATS), N) or (state0, trajectory)."""
    def play(params, coef, key):
        k0, key = jax.random.split(key)
        obs, state = env.reset(k0)

        def step(carry, _):
            state, obs, key, acc = carry
            key, k_act, k_env = jax.random.split(key, 3)
            logits = per_agent(policy_logits, params, env.agent_team, obs)
            act = jnp.argmax(logits, -1) if deterministic else jax.random.categorical(k_act, logits)
            mem = obs["coins"][:, 0, N_BASE:].T                                   # (4, N) memory seen at decision
            obs2, state2, pure, info, taken = env.step(k_env, state, act)
            row = jnp.concatenate([jnp.stack([pure, env.attitude_reward(pure, coef),
                                              *[info[k].astype(jnp.float32) for k in STATS]]),
                                   choice_counts(obs, act), mem / env.episode_len])
            return (state2, obs2, key, acc + row), ((state2, act, pure, info) if record else None)

        acc0 = jnp.zeros((len(SIM_STATS), env.num_agents))
        (_, _, _, acc), traj = jax.lax.scan(step, (state, obs, key, acc0), None, env.episode_len)
        return (state, traj) if record else acc
    return play


def make_simulator(env, deterministic=False):
    """(params (R, 2, ...), coef, keys (S,)) -> (R, S, stats, N); same keys as make_recorder."""
    return jax.jit(jax.vmap(jax.vmap(make_player(env, deterministic), (None, None, 0)), (0, None, None)))


def make_recorder(env, deterministic=False):
    return jax.jit(make_player(env, deterministic, record=True))
