"""A7 tests: separate gradient clipping and return normalisation (ppo.py)."""
import jax
import jax.numpy as jnp
import numpy as np

from coopcoins.networks import init_policy
from coopcoins.ppo import global_norm, init_ret_stats, make_optimizer, ret_std, update_ret_stats

CFG = {"lr": 1e-3, "clip": {"max_grad_norm_actor": 0.5, "max_grad_norm_critic": 0.5}}


def scaled_grads(actor_norm, critic_norm, key=0):
    g = init_policy(jax.random.PRNGKey(key), 8)
    g = jax.tree.map(lambda x: jnp.ones_like(x), g)
    sa, sc = global_norm(g["actor"]), global_norm(g["critic"])
    return {"actor": jax.tree.map(lambda x: x * actor_norm / sa, g["actor"]),
            "critic": jax.tree.map(lambda x: x * critic_norm / sc, g["critic"])}


def test_actor_and_critic_clipped_independently():
    clip = make_optimizer(CFG, with_adam=False)
    g = scaled_grads(actor_norm=0.2, critic_norm=50.0)
    out, _ = clip.update(g, clip.init(g), g)
    assert np.isclose(global_norm(out["actor"]), 0.2, rtol=1e-5)     # untouched (joint clipping would shrink it)
    assert np.isclose(global_norm(out["critic"]), 0.5, rtol=1e-5)
    g = scaled_grads(actor_norm=7.0, critic_norm=0.1)
    out, _ = clip.update(g, clip.init(g), g)
    assert np.isclose(global_norm(out["actor"]), 0.5, rtol=1e-5)
    assert np.isclose(global_norm(out["critic"]), 0.1, rtol=1e-5)
    # different thresholds per subtree
    clip2 = make_optimizer({**CFG, "clip": {"max_grad_norm_actor": 1.0, "max_grad_norm_critic": 3.0}}, False)
    g = scaled_grads(10.0, 10.0)
    out, _ = clip2.update(g, clip2.init(g), g)
    assert np.isclose(global_norm(out["actor"]), 1.0, rtol=1e-5)
    assert np.isclose(global_norm(out["critic"]), 3.0, rtol=1e-5)


def test_clipping_is_per_team_under_vmap():
    clip = make_optimizer(CFG, with_adam=False)
    g = jax.tree.map(lambda a, b: jnp.stack([a, b]), scaled_grads(9.0, 0.1), scaled_grads(0.1, 9.0))
    st = jax.vmap(clip.init)(g)
    out, _ = jax.vmap(clip.update)(g, st, g)
    norms = jax.vmap(lambda x: jnp.stack([global_norm(x["actor"]), global_norm(x["critic"])]))(out)
    assert np.allclose(np.asarray(norms), [[0.5, 0.1], [0.1, 0.5]], rtol=1e-5)


def test_full_optimizer_runs_per_team():
    opt = make_optimizer(CFG)
    p = jax.vmap(lambda k: init_policy(k, 8))(jax.random.split(jax.random.PRNGKey(0), 2))
    st = jax.vmap(opt.init)(p)
    upd, _ = jax.vmap(opt.update)(p, st, p)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(upd))


def test_return_stats_match_numpy():
    rng = np.random.default_rng(0)
    x1 = rng.normal(3.0, 2.0, (2, 500)).astype(np.float32)
    x2 = rng.normal(-1.0, 5.0, (2, 300)).astype(np.float32)
    st0 = init_ret_stats(1.0)
    assert np.allclose(np.asarray(ret_std(st0, 1e-8)), 1.0)
    st1 = update_ret_stats(st0, jnp.asarray(x1))
    assert np.allclose(np.asarray(st1.mean), x1.mean(1), rtol=1e-5)
    assert np.allclose(np.asarray(st1.var), x1.var(1), rtol=1e-4)
    st2 = update_ret_stats(st1, jnp.asarray(x2))
    allx = np.concatenate([x1, x2], 1)
    assert np.allclose(np.asarray(st2.mean), allx.mean(1), rtol=1e-4, atol=1e-5)
    assert np.allclose(np.asarray(st2.var), allx.var(1), rtol=1e-4)
    assert np.allclose(np.asarray(ret_std(st2, 1e-8)), np.sqrt(allx.var(1) + 1e-8), rtol=1e-4)
    assert np.allclose(np.asarray(st2.count), 800)


def test_runner_types_stable_across_updates():
    """The runner after an update has the same shapes / dtypes / weak types as the initial one (no recompilation)."""
    from coopcoins.env import make_env
    from coopcoins.ppo import make_trainer
    from coopcoins.utils import load_experiment
    cfg = load_experiment("configs/experiments/payoff_k2.yaml")
    tc = dict(cfg["train"], num_minibatches=3)
    init_v, chunk_v = make_trainer([make_env(**cfg["maps"]["grid3_agents2"], max_coins=2)], tc)
    r0 = init_v(jnp.stack([jax.random.PRNGKey(0)]), jnp.array([[cfg["attitudes"]["individualistic"]] * 2]))
    r1, _ = chunk_v(r0, 1)
    sig = lambda r: [(x.shape, x.dtype, x.weak_type) for x in jax.tree.leaves(r)]
    assert sig(r0) == sig(r1)
