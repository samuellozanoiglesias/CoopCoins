"""Games with trained team policies (no learning), same length as in training."""
import jax
import jax.numpy as jnp

from .env import STATS
from .networks import policy_logits
from .ppo import per_agent

SIM_STATS = ["pure_reward", "attitude_reward", *STATS]


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
            obs, state2, pure, info, taken = env.step(k_env, state, act)
            row = jnp.stack([pure, env.attitude_reward(pure, coef), *[info[k].astype(jnp.float32) for k in STATS]])
            return (state2, obs, key, acc + row), ((state2, act, pure, info) if record else None)

        acc0 = jnp.zeros((len(SIM_STATS), env.num_agents))
        (_, _, _, acc), traj = jax.lax.scan(step, (state, obs, key, acc0), None, env.episode_len)
        return (state, traj) if record else acc
    return play


def make_simulator(env, deterministic=False):
    """(params (R, 2, ...), coef, keys (S,)) -> (R, S, stats, N); same keys as make_recorder."""
    return jax.jit(jax.vmap(jax.vmap(make_player(env, deterministic), (None, None, 0)), (0, None, None)))


def make_recorder(env, deterministic=False):
    return jax.jit(make_player(env, deterministic, record=True))
