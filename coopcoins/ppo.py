"""PPO with one policy per TEAM (shared by all team members), fully in JAX.

A training set is one or more maps. One update = `num_envs` games of `episode_len` steps on EVERY
map of the set, then one PPO step per team on all the data of its members. With several maps
this trains a single "generalist" pair of team policies that plays all of them.
Independent runs (attitude pair x seed) are vmapped into one compiled program.

Gradient clipping: the "actor" and "critic" subtrees of a team's parameters are each clipped by their own
global norm (train.clip.max_grad_norm_actor / max_grad_norm_critic), then Adam.
Return normalisation: running mean / variance of returns per team and per run (in Runner). The critic predicts
normalised values; they are de-normalised (v * std + mean) before GAE, the statistics are updated from each
update's batch of returns, and the critic is trained on (ret - mean) / std with the updated statistics.
std = sqrt(var + eps); initial mean 0, var 1 (std 1), count 0.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax

from .networks import init_policy, policy_logits, value


class RetStats(NamedTuple):
    mean: jnp.ndarray   # (2,) per team
    var: jnp.ndarray    # (2,)
    count: jnp.ndarray  # (2,)


class Runner(NamedTuple):
    params: dict      # leading axis 2 = team
    opt_state: tuple
    key: jnp.ndarray
    coef: jnp.ndarray  # (2, 2): [alpha, beta] per team
    ret_stats: RetStats


def init_ret_stats(init_std=1.0):
    # explicit float32 (not weak-typed): the runner returned by an update must have the same types as the initial
    # one, or the first chunk is compiled twice
    f32 = jnp.float32
    return RetStats(jnp.zeros(2, f32), jnp.full(2, float(init_std) ** 2, f32), jnp.zeros(2, f32))


def ret_std(st, eps):
    return jnp.sqrt(st.var + eps)


def update_ret_stats(st, x):
    """Combine running stats with a batch x (2 teams, n) (parallel / Chan update; exact with count 0)."""
    bm, bv, bc = x.mean(1), x.var(1), jnp.float32(x.shape[1])
    tot = st.count + bc
    delta = bm - st.mean
    mean = st.mean + delta * bc / tot
    m2 = st.var * st.count + bv * bc + jnp.square(delta) * st.count * bc / tot
    return RetStats(mean, m2 / tot, tot)


def global_norm(tree):
    """L2 norm over all leaves: optax.tree.norm where available (optax.global_norm is deprecated)."""
    tree_mod = getattr(optax, "tree", None)
    return tree_mod.norm(tree) if hasattr(tree_mod, "norm") else optax.global_norm(tree)


def labels(params):
    """Optimizer partition: every leaf under 'actor' -> 'actor', under 'critic' -> 'critic'."""
    return {k: jax.tree.map(lambda _: k, v) for k, v in params.items()}


def make_optimizer(cfg, with_adam=True):
    """Per-subtree global-norm clipping, then Adam (elementwise, so equivalent to one Adam over all params).
    with_adam=False returns the clipping stage alone (tests)."""
    c = cfg["clip"]
    branch = lambda mx: optax.chain(optax.clip_by_global_norm(mx),
                                    optax.adam(cfg["lr"], eps=1e-5) if with_adam else optax.identity())
    return optax.multi_transform({"actor": branch(c["max_grad_norm_actor"]),
                                  "critic": branch(c["max_grad_norm_critic"])}, labels)


def per_agent(fn, params, agent_team, obs):
    """Apply fn(team params, single-agent obs) to obs with leaves (..., N, ...): agent axis = -3 for coins."""
    pa = jax.tree.map(lambda p: p[agent_team], params)                      # (N, ...)
    lead = obs["mask"].ndim - 2                                             # batch dims before N
    f = lambda p, o: fn(p, o)
    for _ in range(lead):
        f = jax.vmap(f, (None, 0))
    return jax.vmap(f, (0, lead), lead)(pa, obs)


def make_trainer(envs, cfg):
    assert "max_grad_norm" not in cfg, "max_grad_norm was replaced by clip: {max_grad_norm_actor, max_grad_norm_critic}"
    E = cfg["num_envs"]
    opt = make_optimizer(cfg)
    rn = cfg["return_norm"]
    eps = float(rn["eps"])
    g, lam = cfg["gamma"], cfg["gae_lambda"]
    n_samples = sum(E * env.episode_len * env.num_agents // 2 for env in envs)   # per team per update
    n_mb = cfg["num_minibatches"]
    mb = n_samples // n_mb

    def init(key, coef):
        k_net, key = jax.random.split(key)
        params = jax.vmap(lambda k: init_policy(k, cfg["hidden"]))(jax.random.split(k_net, 2))
        return Runner(params, jax.vmap(opt.init)(params), key, coef, init_ret_stats(rn["init_std"]))

    def play(env, params, coef, key, stats):
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
        vals_n = per_agent(value, params, env.agent_team, all_obs)             # (T+1, E, N) normalised
        vals = vals_n * ret_std(stats, eps)[env.agent_team] + stats.mean[env.agent_team]  # de-normalised
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
                norms = jax.vmap(lambda gr: jnp.stack([global_norm(gr["actor"]),
                                                       global_norm(gr["critic"])]))(grads)  # pre-clip, (2, 2)
                aux = jnp.concatenate([aux, norms], -1)
                upd, opt_state = jax.vmap(opt.update)(grads, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            return jax.lax.scan(mb_step, carry, mbs)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, cfg["update_epochs"]))
        return params, opt_state, aux.mean((0, 1))  # (2, 6)

    def train_update(r: Runner, _):
        keys = jax.random.split(r.key, len(envs) + 2)
        batches, metrics = zip(*[play(env, r.params, r.coef, k, r.ret_stats) for env, k in zip(envs, keys[2:])])
        obs, act, logp, adv, ret = jax.tree.map(lambda *x: jnp.concatenate(x, axis=1), *batches)
        stats = update_ret_stats(r.ret_stats, ret)                     # update from this batch, then normalise
        target = (ret - stats.mean[:, None]) / ret_std(stats, eps)[:, None]
        params, opt_state, aux = update(r.params, r.opt_state, (obs, act, logp, adv, target), keys[1])
        losses = {"pg_loss": aux[:, 0], "vf_loss": aux[:, 1], "entropy": aux[:, 2], "approx_kl": aux[:, 3],
                  "actor_grad_norm": aux[:, 4], "critic_grad_norm": aux[:, 5],
                  "return_mean": stats.mean, "return_std": ret_std(stats, eps)}
        return Runner(params, opt_state, keys[0], r.coef, stats), (metrics, losses)  # metrics: one dict per map

    def train_chunk(r: Runner, n_updates: int):
        return jax.lax.scan(train_update, r, None, n_updates)

    init_v = jax.jit(jax.vmap(init))
    chunk_v = jax.jit(lambda r, n: jax.vmap(lambda ri: train_chunk(ri, n))(r), static_argnums=1)
    return init_v, chunk_v
