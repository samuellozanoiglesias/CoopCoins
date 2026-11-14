"""PPO with one policy per TEAM (shared by all team members), fully in JAX.

A training set is one or more maps. One update = `num_envs` games of `episode_len` steps on EVERY
map of the set, then one PPO step per team on all the data of its members. With several maps
this trains a single "generalist" pair of team policies that plays all of them.
Independent runs (attitude pair x seed) are vmapped into one compiled program.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax

from .networks import init_policy, policy_logits, value


class Runner(NamedTuple):
    params: dict      # leading axis 2 = team
    opt_state: tuple
    key: jnp.ndarray
    coef: jnp.ndarray  # (2, 2): [alpha, beta] per team


def per_agent(fn, params, agent_team, obs):
    """Apply fn(team params, single-agent obs) to obs with leaves (..., N, ...): agent axis = -3 for coins."""
    pa = jax.tree.map(lambda p: p[agent_team], params)                      # (N, ...)
    lead = obs["mask"].ndim - 2                                             # batch dims before N
    f = lambda p, o: fn(p, o)
    for _ in range(lead):
        f = jax.vmap(f, (None, 0))
    return jax.vmap(f, (0, lead), lead)(pa, obs)


def make_trainer(envs, cfg):
    E = cfg["num_envs"]
    opt = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]), optax.adam(cfg["lr"], eps=1e-5))
    g, lam = cfg["gamma"], cfg["gae_lambda"]
    n_samples = sum(E * env.episode_len * env.num_agents // 2 for env in envs)   # per team per update
    n_mb = cfg["num_minibatches"]
    mb = n_samples // n_mb

    def init(key, coef):
        k_net, key = jax.random.split(key)
        params = jax.vmap(lambda k: init_policy(k, cfg["hidden"]))(jax.random.split(k_net, 2))
        return Runner(params, jax.vmap(opt.init)(params), key, coef)

    def play(env, params, coef, key):
        T, N = env.episode_len, env.num_agents
        k_reset, key = jax.random.split(key)
        obs, state = jax.vmap(env.reset)(jax.random.split(k_reset, E))

        def step(carry, _):
            state, obs, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            logits = per_agent(policy_logits, params, env.agent_team, obs)    # (E, N, P+1)
            act = jax.random.categorical(k_act, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), act[..., None], -1)[..., 0]
            obs2, state2, pure, info, _ = jax.vmap(env.step)(jax.random.split(k_env, E), state, act)
            return (state2, obs2, key), (obs, act, logp, pure, info)

        (_, last_obs, _), (obs, act, logp, pure, info) = jax.lax.scan(step, (state, obs, key), None, T)
        all_obs = jax.tree.map(lambda a, b: jnp.concatenate([a, b[None]]), obs, last_obs)
        vals = per_agent(value, params, env.agent_team, all_obs)               # (T+1, E, N)
        rew = env.attitude_reward(pure, coef)
        # GAE; the game end is a time limit -> bootstrap from V(last obs)
        def f(adv_next, x):
            r, v, v_next = x
            adv = r + g * v_next - v + g * lam * adv_next
            return adv, adv
        _, adv = jax.lax.scan(f, jnp.zeros_like(vals[0]), (rew, vals[:-1], vals[1:]), reverse=True)
        ret = adv + vals[:-1]
        # (T, E, N, ...) -> (2 teams, T*E*N/2, ...): agent i = 2*j + team
        to_team = lambda x: jnp.moveaxis(x.reshape(*x.shape[:2], N // 2, 2, *x.shape[3:]), 3, 0).reshape(2, -1, *x.shape[3:])
        batch = jax.tree.map(to_team, (obs, act, logp, adv, ret))
        team_mean = lambda x: to_team(x.sum(0)[None]).mean(1)  # per-game totals, mean over games and members
        metrics = {"pure_reward": team_mean(pure), "attitude_reward": team_mean(rew),
                   **{k: team_mean(v.astype(jnp.float32)) for k, v in info.items()}}
        return batch, metrics

    def loss_fn(p, obs, act, logp_old, adv, ret):
        logp_all = jax.nn.log_softmax(jax.vmap(policy_logits, (None, 0))(p, obs))
        logp = jnp.take_along_axis(logp_all, act[:, None], -1)[:, 0]
        ratio = jnp.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv).mean()
        vf = jnp.square(jax.vmap(value, (None, 0))(p, obs) - ret).mean()
        ent = -(jnp.where(logp_all > -1e8, jnp.exp(logp_all) * logp_all, 0.0)).sum(-1).mean()
        kl = ((ratio - 1) - (logp - logp_old)).mean()
        return pg + cfg["vf_coef"] * vf - cfg["ent_coef"] * ent, jnp.stack([pg, vf, ent, kl])

    grad_fn = jax.vmap(jax.grad(loss_fn, has_aux=True))  # over teams

    def update(params, opt_state, batch, key):
        def epoch(carry, k):
            perm = jax.random.permutation(k, n_samples)[: n_mb * mb]
            mbs = jax.tree.map(lambda x: x[:, perm].reshape(2, n_mb, mb, *x.shape[2:]).swapaxes(0, 1), batch)

            def mb_step(carry, m):
                params, opt_state = carry
                grads, aux = grad_fn(params, *m)
                upd, opt_state = jax.vmap(opt.update)(grads, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            return jax.lax.scan(mb_step, carry, mbs)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, cfg["update_epochs"]))
        return params, opt_state, aux.mean((0, 1))  # (2, 4)

    def train_update(r: Runner, _):
        keys = jax.random.split(r.key, len(envs) + 2)
        batches, metrics = zip(*[play(env, r.params, r.coef, k) for env, k in zip(envs, keys[2:])])
        batch = jax.tree.map(lambda *x: jnp.concatenate(x, axis=1), *batches)
        params, opt_state, aux = update(r.params, r.opt_state, batch, keys[1])
        losses = {"pg_loss": aux[:, 0], "vf_loss": aux[:, 1], "entropy": aux[:, 2], "approx_kl": aux[:, 3]}
        return Runner(params, opt_state, keys[0], r.coef), (metrics, losses)  # metrics: one dict per map

    def train_chunk(r: Runner, n_updates: int):
        return jax.lax.scan(train_update, r, None, n_updates)

    init_v = jax.jit(jax.vmap(init))
    chunk_v = jax.jit(lambda r, n: jax.vmap(lambda ri: train_chunk(ri, n))(r), static_argnums=1)
    return init_v, chunk_v
