import jax #pip install jax
import jax.numpy as jnp
import optax
import csv
import os
import pickle
import jaxmarl
from jaxmarl import make
from jaxmarl.environments.coin_game.actor_critic import ActorCritic

def make_train(config):
    # Crear entornos
    keys = jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
    envs = [make("coin_game", 
        num_inner_steps = config["NUM_INNER_STEPS"],
        num_outer_steps = config["NUM_OUTER_STEPS"],
        cnn = False,
        egocentric = True,
        shared_rewards = config["SHARED_REWARDS"],
        payoff_matrix = config["PAYOFF_MATRIX"]
        ) for _ in range(config["NUM_ENVS"])]
    states = [env.reset(k)[1] for env, k in zip(envs, keys)]
    obs = [env.reset(k)[0] for env, k in zip(envs, keys)]

    # === INIT NETWORKS, OPTIMIZERS ===
    params = {}
    opt_state = {}
    models = {}
    optimizers = {}

    example_env = envs[0]
    action_dim = example_env.action_space("agent_0").n

    for i in range(config["NUM_AGENTS"]):
        agent = f"agent_{i}"
        model = ActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros(example_env.observation_space().shape)[None, ...]
        rng = jax.random.PRNGKey(42 + i)
        variables = model.init(rng, dummy_obs)
        params[agent] = variables
        models[agent] = model
        optimizers[agent] = optax.adam(config["LR"])
        opt_state[agent] = optimizers[agent].init(variables)

    # === ACTION FUNCTION ===
    def select_action(model, params, obs, key):
        pi, value = model.apply(params, obs)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, value, log_prob
    
    # === LOSS FUNCTION ===
    def loss_fn(params, model, obs, action, advantage, old_log_prob, returns):
        pi, value = model.apply(params, obs)
        new_log_prob = pi.log_prob(action)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 0.8, 1.2)
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        critic_loss = (returns - value) ** 2
        loss = actor_loss + 0.5 * critic_loss
        return loss.mean()
    
    rewards = {}

    # === TRAINING LOOP ===
    for epoch in range(config["NUM_EPOCHS"]):
        rewards[epoch] = {}

        for env_idx, env in enumerate(envs):
            state = states[env_idx]
            obs_env = obs[env_idx]
            key = jax.random.PRNGKey(epoch * 100 + env_idx)

            actions = {}
            values = {}
            log_probs = {}

            for i in enumerate(env.agents):
                agent = f"agent_{i[0]}"
                obs_agent = jnp.array(obs_env[agent])[None, ...]
                key, subkey = jax.random.split(key)
                action, value, log_prob = select_action(models[agent], params[agent], obs_agent, subkey)
                actions[agent] = action
                values[agent] = value
                log_probs[agent] = log_prob

            obs_next, state_next, reward, done, info = env.step(key, state, actions)

            # PPO Step (1-step advantage)
            for agent in env.agents:
                obs_agent = jnp.array(obs_env[agent])[None, ...]
                rew = reward[agent]
                advantage = rew - values[agent]
                returns = rew

                def grad_loss(p):
                    return loss_fn(p, models[agent], obs_agent, actions[agent], advantage, log_probs[agent], returns)

                grads = jax.grad(grad_loss)(params[agent])
                updates, opt_state[agent] = optimizers[agent].update(grads, opt_state[agent])
                params[agent] = optax.apply_updates(params[agent], updates)
                rewards[epoch][agent] = rew.item()

            states[env_idx] = state_next
            obs[env_idx] = obs_next

        if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"\nEpoch {epoch}:")
            for agent in reward:
                print(f"  Reward of {agent} = {rewards[epoch][agent]:.2f}")