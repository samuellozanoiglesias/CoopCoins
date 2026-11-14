"""Set network over coins: works for any number of coins (and so any map).

Each coin row is encoded by a shared MLP; a masked mean+max pool gives the context.
Actor: score per coin = MLP([coin code, context]); "stay" score = MLP(context).
Critic: separate encoder, value = MLP(context).
"""
import jax
import jax.numpy as jnp

from .env import N_FEATURES

_ortho = jax.nn.initializers.orthogonal


def _init(key, sizes, out_scale=None):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = out_scale if (out_scale is not None and i == len(sizes) - 2) else jnp.sqrt(2.0)
        layers.append({"w": _ortho(scale)(k, (a, b)), "b": jnp.zeros(b)})
    return layers


def _mlp(layers, x, last_act=False):
    for i, l in enumerate(layers):
        x = x @ l["w"] + l["b"]
        if i < len(layers) - 1 or last_act:
            x = jnp.tanh(x)
    return x


def init_policy(key, hidden=64):
    k = jax.random.split(key, 5)
    h = hidden
    return {"actor": {"enc": _init(k[0], [N_FEATURES, h, h]), "coin": _init(k[1], [3 * h, h, 1], 0.01),
                      "stay": _init(k[2], [2 * h, h, 1], 0.01)},
            "critic": {"enc": _init(k[3], [N_FEATURES, h, h]), "head": _init(k[4], [2 * h, h, 1], 1.0)}}


def _encode(enc, coins, mask):
    z = _mlp(enc, coins, last_act=True)                                   # (P, h)
    m = mask[:, None]
    mean = (z * m).sum(0) / jnp.maximum(m.sum(), 1)
    mx = jnp.where(m, z, -1.0).max(0)                                      # tanh codes are >= -1
    return z, jnp.concatenate([mean, mx])


def policy_logits(p, obs):
    """obs: {'coins': (P, F), 'mask': (P,)} -> logits (P + 1,), last = stay."""
    z, ctx = _encode(p["actor"]["enc"], obs["coins"], obs["mask"])
    s = _mlp(p["actor"]["coin"], jnp.concatenate([z, jnp.broadcast_to(ctx, (z.shape[0], ctx.shape[0]))], -1))[:, 0]
    s = jnp.where(obs["mask"], s, -1e9)
    return jnp.concatenate([s, _mlp(p["actor"]["stay"], ctx)])


def value(p, obs):
    _, ctx = _encode(p["critic"]["enc"], obs["coins"], obs["mask"])
    return _mlp(p["critic"]["head"], ctx)[0]
