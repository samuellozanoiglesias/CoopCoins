import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

# === NETWORK DEFINITION ===
class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    def __init__(self, obs_shape, n_actions, key, hidden_sizes=(256, 256, 128)):
        key1, key2 = jr.split(key)
        flat_obs_dim = int(jnp.prod(jnp.array(obs_shape)))

        self.actor = eqx.nn.MLP(flat_obs_dim, n_actions, hidden_sizes, key=key1)
        self.critic = eqx.nn.MLP(flat_obs_dim, 1, hidden_sizes, key=key2)

    def __call__(self, obs):
        x = jnp.ravel(obs)  # Flatten (e.g., grid or spatial obs)
        logits = self.actor(x)
        value = self.critic(x).squeeze()
        return logits, value