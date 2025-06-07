import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax import lax
import optax
import csv
import os
import pickle
from datetime import datetime
from jaxmarl import make
from jaxmarl.environments.coin_game.actor_critic import ActorCritic

def actor_critic_fn(obs_shape, n_actions, key):
    model = ActorCritic(obs_shape, n_actions, key)
    return model, lambda params, obs: model(obs)

def make_train(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # Crear entornos
    keys = jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
    envs = [make("coin_game", 
        num_inner_steps = config["NUM_INNER_STEPS"],
        num_outer_steps = config["NUM_EPOCHS"],
        cnn = False,
        egocentric = False,
        payoff_matrix = config["PAYOFF_MATRIX"],
        grid_size = config["GRID_SIZE"],
        reward_coef = config["REWARD_COEF"]
        ) for _ in range(config["NUM_ENVS"])]
    
    env = envs[0]
    agents = env.possible_agents
    obs_shape = env.observation_space(agents[0]).shape
    n_actions = env.action_space(agents[0]).n

    model_params, model_apply, opt_state, optimizer = {}, {}, {}, {}
    for i, agent in enumerate(agents):
        key = random.PRNGKey(42 + i)
        model, apply_fn = actor_critic_fn(obs_shape, n_actions, key)
        model_params[agent] = model.params
        model_apply[agent] = apply_fn
        optimizer[agent] = optax.adam(config["LR"])
        opt_state[agent] = optimizer[agent].init(model_params[agent])

    # === ACTION FUNCTION ===
    @jit
    def select_action(params, obs, key, apply_fn):
        logits, value = apply_fn(params, obs)
        probs = jax.nn.softmax(logits)
        action = jax.random.categorical(key, logits)
        log_prob = jnp.log(probs[action])
        return action, log_prob, value
    
    # === LOSS FUNCTION ===
    @jit
    def compute_loss(params, obs, actions, advantages, returns, apply_fn):
        logits, values = apply_fn(params, obs)
        log_probs = jax.nn.log_softmax(logits)
        chosen_log_probs = jnp.sum(log_probs * jax.nn.one_hot(actions, logits.shape[-1]), axis=-1)
        policy_loss = -jnp.mean(chosen_log_probs * advantages)
        value_loss = jnp.mean((returns - values) ** 2)
        entropy_loss = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    
    # === COMPUTE ADVANTAGE ===
    @jit
    def compute_gae(rewards, values, gamma):
        next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
        deltas = rewards + gamma * next_values - values

        def scan_fn(carry, delta):
            gae = delta + gamma * carry
            return gae, gae

        _, gaes = lax.scan(scan_fn, 0.0, deltas[::-1])
        gaes = gaes[::-1]
        returns = gaes + values
        return returns, gaes
    
    # === TRAINING UPDATE ===
    @jit
    def train_minibatches(params, opt_state, obs_batch, action_batch, advantage, return_batch, key, apply_fn, optimizer, minibatch_size, epochs):
        num_minibatches = obs_batch.shape[0] // minibatch_size

        def epoch_step(carry, _):
            params, opt_state, key = carry
            key, subkey = random.split(key)
            perm = random.permutation(subkey, obs_batch.shape[0])

            def minibatch_step(carry, i):
                params, opt_state = carry
                idx = perm[i * minibatch_size:(i + 1) * minibatch_size]
                batch_obs = obs_batch[idx]
                batch_actions = action_batch[idx]
                batch_adv = advantage[idx]
                batch_returns = return_batch[idx]

                loss, grads = value_and_grad(compute_loss)(params, batch_obs, batch_actions, batch_adv, batch_returns, apply_fn)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss

            (params, opt_state), _ = lax.scan(minibatch_step, (params, opt_state), jnp.arange(num_minibatches))
            return (params, opt_state, key), None

        (params, opt_state, _), _ = lax.scan(epoch_step, (params, opt_state, key), None, length=epochs)
        return params, opt_state
    
    # === LOGGING INIT ===
    states = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
    observations = [env.observe(state) for env, state in zip(envs, states)]
    csv_path = os.path.join(path, "training_stats.csv")
    write_header = not os.path.exists(csv_path)

    # === TRAINING LOOP ===
    for epoch in range(config["NUM_EPOCHS"]):
        for env_idx, env in enumerate(envs):
            state = states[env_idx]
            obs_env = observations[env_idx]
            done = False
            key = random.PRNGKey(epoch * 100 + env_idx)

            trajectory = {agent: {"obs": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
                          for agent in env.agents}

            while not done:
                actions, values, log_probs = {}, {}, {}

                for i, agent in enumerate(env.agents):
                    obs_agent = jnp.array(obs_env[agent])[None, ...]
                    key, subkey = random.split(key)
                    action, log_prob, value = select_action(model_params[agent], obs_agent, subkey, model_apply[agent])
                    actions[agent] = int(action)
                    trajectory[agent]["obs"].append(obs_agent)
                    trajectory[agent]["actions"].append(action)
                    trajectory[agent]["log_probs"].append(log_prob)
                    trajectory[agent]["values"].append(value)

                obs_next, state_next, reward, dones, infos = env.step(key, state, actions)

                for agent in env.agents:
                    trajectory[agent]["rewards"].append(reward[agent])

                done = dones["__all__"]
                state = state_next
                obs_env = obs_next

            for agent in env.agents:
                obs_batch = jnp.concatenate(trajectory[agent]["obs"], axis=0)
                action_batch = jnp.array(trajectory[agent]["actions"])
                value_batch = jnp.array(trajectory[agent]["values"]).squeeze()
                reward_batch = jnp.array(trajectory[agent]["rewards"])

                returns, advantage = compute_gae(reward_batch, value_batch, config["GAMMA"])
                advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

                model_params[agent], opt_state[agent] = train_minibatches(
                    model_params[agent],
                    opt_state[agent],
                    obs_batch,
                    action_batch,
                    advantage,
                    returns,
                    key,
                    model_apply[agent],
                    optimizer[agent],
                    config["MINIBATCH_SIZE"],
                    config["MINIBATCH_EPOCHS"]
                )

            stats = infos
            row = {
                "epoch": epoch,
                "env": env_idx,
            }

            for agent in env.agents:
                row.update({
                    f"reward_{agent}": float(infos[agent]["cumulated_modified_reward"]),
                    f"pure_reward_{agent}": float(infos[agent]["cumulated_pure_reward"]),
                })
                a_stats = infos[agent]["cumulated_action_stats"]
                row.update({
                    f"own_coin_{agent}": int(a_stats[0]),
                    f"other_coin_{agent}": int(a_stats[1]),
                    f"reject_own_{agent}": int(a_stats[2]),
                    f"reject_other_{agent}": int(a_stats[3]),
                    f"no_coin_{agent}": int(a_stats[4]),
                })

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

        if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        if epoch % config["SAVE_EVERY_N_EPOCHS"] == 0:
            with open(os.path.join(path, f"params_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(model_params, f)

    return model_params, current_date