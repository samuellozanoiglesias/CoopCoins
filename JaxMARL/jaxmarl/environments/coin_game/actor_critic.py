import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

class CustomMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, hidden_sizes, key):
        keys = jr.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + list(hidden_sizes) + [out_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = eqx.nn.Linear(sizes[i], sizes[i+1], key=keys[i])
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x

# === NETWORK DEFINITION ===
class ActorCritic(eqx.Module):
    actor: CustomMLP
    critic: CustomMLP

    def __init__(self, obs_shape, n_actions, key, hidden_sizes=(256, 256, 128)):
        key1, key2 = jr.split(key)
        flat_obs_dim = int(jnp.prod(jnp.array(obs_shape)))

        self.actor = CustomMLP(flat_obs_dim, n_actions, hidden_sizes, key1)
        self.critic = CustomMLP(flat_obs_dim, 1, hidden_sizes, key2)

    def __call__(self, obs):
        x = jnp.ravel(obs)
        logits = self.actor(x)
        value = self.critic(x).squeeze()
        return logits, value