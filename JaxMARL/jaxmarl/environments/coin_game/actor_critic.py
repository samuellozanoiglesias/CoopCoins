import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax 
from typing import NamedTuple

# === NETWORK DEFINITION ===
class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        a = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        a = act_fn(a)
        a = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a)
        a = act_fn(a)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        pi = distrax.Categorical(logits=logits)

        # Critic
        c = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        c = act_fn(c)
        c = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(c)
        c = act_fn(c)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)

        return pi, jnp.squeeze(value, axis=-1)
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray