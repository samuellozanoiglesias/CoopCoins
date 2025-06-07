import jax #pip install jax
import jax.numpy as jnp
import optax
import csv
import os
import pickle
import jaxmarl
from datetime import datetime
from jaxmarl import make
from jaxmarl.environments.coin_game.actor_critic import ActorCritic

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
    states = [env.reset(k)[1] for env, k in zip(envs, keys)]
    obs = [env.reset(k)[0] for env, k in zip(envs, keys)]

    # === INIT NETWORKS, OPTIMIZERS ===
    params, opt_state, models, optimizers = {}, {}, {}, {}
    example_env = envs[0]
    action_dim = example_env.action_space("agent_0").n
    entropy_coef = config.get("ENTROPY_COEF", 0.01)
    gamma = config.get("GAMMA", 0.99)
    clip_epsilon = config.get("CLIP_EPSILON", 0.2)

    for i in range(config["NUM_AGENTS"]):
        agent = f"agent_{i}"
        model = ActorCritic(action_dim=action_dim)
        dummy_obs = jnp.zeros(example_env.observation_space().shape)[None, ...]
        rng = jax.random.PRNGKey(42 + i)
        variables = model.init(rng, dummy_obs)
        params[agent] = variables
        models[agent] = model
        optimizers[agent] = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        opt_state[agent] = optimizers[agent].init(variables)

    # === ACTION FUNCTION ===
    def select_action(model, params, obs, key):
        pi, value = model.apply(params, obs)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, value, log_prob
    
    # === LOSS FUNCTION ===
    def loss_fn(params, model, obs, action, advantage, old_log_prob, returns, entropy_coef, clip_epsilon):
        pi, value = model.apply(params, obs)
        new_log_prob = pi.log_prob(action)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        critic_loss = (returns - value) ** 2
        entropy_bonus = pi.entropy()
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_bonus
        return loss.mean()
    
    # === LOGGING INIT ===
    rewards, pure_rewards, action_stats_total = {}, {}, {}
    csv_path = os.path.join(path, "training_stats.csv")
    write_header = not os.path.exists(csv_path)

    # === TRAINING LOOP ===
    for epoch in range(config["NUM_EPOCHS"]):
        rewards[epoch], pure_rewards[epoch], action_stats_total[epoch] = {}, {}, {}

        for env_idx, env in enumerate(envs):
            rewards[epoch][env_idx], pure_rewards[epoch][env_idx], action_stats_total[epoch][env_idx] = {}, {}, {}
            state = states[env_idx]
            obs_env = obs[env_idx]
            key = jax.random.PRNGKey(epoch * 100 + env_idx)

            if config["TRAINING_TYPE"] == 'AC_Step':
                done = False

                while not done:
                    actions, values, log_probs = {}, {}, {}
                    for i in enumerate(env.agents):
                        agent = f"agent_{i[0]}"
                        obs_agent = jnp.array(obs_env[agent])[None, ...]
                        key, subkey = jax.random.split(key)
                        action, value, log_prob = select_action(models[agent], params[agent], obs_agent, subkey)
                        actions[agent] = action
                        values[agent] = value
                        log_probs[agent] = log_prob

                    obs_next, state_next, reward, dones, infos = env.step(key, state, actions)
                    
                    # PPO Step (1-step advantage)
                    for agent in env.agents:
                        obs_agent = jnp.array(obs_env[agent])[None, ...]
                        rew = reward[agent]

                        next_obs_agent = jnp.array(obs_next[agent])[None, ...]
                        _, next_value = models[agent].apply(params[agent], next_obs_agent)
                        advantage = rew + gamma * next_value - values[agent]
                        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
                        returns = rew + gamma * next_value

                        def grad_loss(p):
                            return loss_fn(p, models[agent], obs_agent, actions[agent], advantage, log_probs[agent], returns, entropy_coef, clip_epsilon)

                        grads = jax.grad(grad_loss)(params[agent])
                        updates, opt_state[agent] = optimizers[agent].update(grads, opt_state[agent])
                        params[agent] = optax.apply_updates(params[agent], updates)

                    done = dones["__all__"]
                    state = state_next
                    obs_env = obs_next

                for agent in env.agents:
                    rewards[epoch][env_idx][agent] = infos[agent]["cumulated_modified_reward"].item()
                    pure_rewards[epoch][env_idx][agent] = infos[agent]["cumulated_pure_reward"].item()
                    action_stats_total[epoch][env_idx][agent] = infos[agent]["cumulated_action_stats"]

                states[env_idx] = state
                obs[env_idx] = obs_env

            elif config["TRAINING_TYPE"] == 'AC_Minibatches':
                done = False
                trajectory = {
                    agent: {"obs": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
                    for agent in env.agents
                }

                while not done:
                    actions, values, log_probs = {}, {}, {}

                    for i in enumerate(env.agents):
                        agent = f"agent_{i[0]}"
                        obs_agent = jnp.array(obs_env[agent])[None, ...]
                        key, subkey = jax.random.split(key)
                        action, value, log_prob = select_action(models[agent], params[agent], obs_agent, subkey)
                        actions[agent] = action
                        values[agent] = value
                        log_probs[agent] = log_prob

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
                    action_batch = jnp.stack(trajectory[agent]["actions"])
                    log_prob_batch = jnp.stack(trajectory[agent]["log_probs"])
                    value_batch = jnp.stack(trajectory[agent]["values"]).squeeze()
                    reward_batch = jnp.array(trajectory[agent]["rewards"])

                    returns = jnp.cumsum(reward_batch[::-1])[::-1]
                    advantage = returns - value_batch
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    minibatch_size = config["MINIBATCH_SIZE"]
                    num_minibatches = len(obs_batch) // minibatch_size

                    for _ in range(config["MINIBATCH_EPOCHS"]):
                        perm = jax.random.permutation(key, len(obs_batch))
                        for i in range(num_minibatches):
                            idx = perm[i * minibatch_size:(i + 1) * minibatch_size]
                            def grad_loss(p):
                                return loss_fn(
                                    p, models[agent],
                                    obs_batch[idx],
                                    action_batch[idx],
                                    advantage[idx],
                                    log_prob_batch[idx],
                                    returns[idx],
                                    entropy_coef,
                                    clip_epsilon
                                )
                            grads = jax.grad(grad_loss)(params[agent])
                            updates, opt_state[agent] = optimizers[agent].update(grads, opt_state[agent])
                            params[agent] = optax.apply_updates(params[agent], updates)

                    rewards[epoch][env_idx][agent] = infos[agent]["cumulated_modified_reward"].item()
                    pure_rewards[epoch][env_idx][agent] = infos[agent]["cumulated_pure_reward"].item()
                    action_stats_total[epoch][env_idx][agent] = infos[agent]["cumulated_action_stats"]

                states[env_idx] = state
                obs[env_idx] = obs_env

            elif config["TRAINING_TYPE"] == 'AC_Epoch':
                done = False
                trajectory = {
                    agent: {
                        "obs": [],
                        "actions": [],
                        "log_probs": [],
                        "rewards": [],
                        "values": [],
                    }
                    for agent in env.agents
                }

                while not done:
                    actions, values, log_probs = {}, {}, {}

                    for i in enumerate(env.agents):
                        agent = f"agent_{i[0]}"
                        obs_agent = jnp.array(obs_env[agent])[None, ...]
                        key, subkey = jax.random.split(key)
                        action, value, log_prob = select_action(models[agent], params[agent], obs_agent, subkey)
                        actions[agent] = action
                        values[agent] = value
                        log_probs[agent] = log_prob

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


                # PPO update per agent after full episode
                for agent in env.agents:
                    obs_batch = jnp.concatenate(trajectory[agent]["obs"], axis=0)
                    action_batch = jnp.stack(trajectory[agent]["actions"])
                    log_prob_batch = jnp.stack(trajectory[agent]["log_probs"])
                    value_batch = jnp.stack(trajectory[agent]["values"]).squeeze()
                    reward_batch = jnp.array(trajectory[agent]["rewards"])

                    # Simple return and advantage (no bootstrapping)
                    returns = jnp.cumsum(reward_batch[::-1])[::-1]
                    advantage = returns - value_batch
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    def grad_loss(p):
                        return loss_fn(p, models[agent], obs_batch, action_batch, advantage, log_prob_batch, returns, entropy_coef, clip_epsilon)

                    grads = jax.grad(grad_loss)(params[agent])
                    updates, opt_state[agent] = optimizers[agent].update(grads, opt_state[agent])
                    params[agent] = optax.apply_updates(params[agent], updates)

                    rewards[epoch][env_idx][agent] = infos[agent]["cumulated_modified_reward"].item()
                    pure_rewards[epoch][env_idx][agent] = infos[agent]["cumulated_pure_reward"].item()
                    action_stats_total[epoch][env_idx][agent] = infos[agent]["cumulated_action_stats"]

                states[env_idx] = state
                obs[env_idx] = obs_env
            
            else:
                print('ERROR EN EL TRAINING_TYPE')
                break
        
            row = {
                "epoch": epoch,
                "env": env_idx,
                "reward_agent_0": rewards[epoch][env_idx]["agent_0"],
                "reward_agent_1": rewards[epoch][env_idx]["agent_1"],
                "pure_reward_agent_0": float(pure_rewards[epoch][env_idx]["agent_0"]),
                "pure_reward_agent_1": float(pure_rewards[epoch][env_idx]["agent_1"]),
                "pure_reward_total": float(pure_rewards[epoch][env_idx]["agent_0"]) + float(pure_rewards[epoch][env_idx]["agent_1"]),
            }

            for agent in ["agent_0", "agent_1"]:
                stats = action_stats_total[epoch][env_idx][agent]
                suffix = f"_{agent}"
                row.update({
                    "own_coin_collected" + suffix: int(stats[0]),
                    "other_coin_collected" + suffix: int(stats[1]),
                    "rejected_own" + suffix: int(stats[2]),
                    "rejected_other" + suffix: int(stats[3]),
                    "no_coin_visible" + suffix: int(stats[4]),
                })

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
                print(f"\nEpoch {epoch}:")
                for agent in reward:
                    print(f"  Pure reward of {agent} in env {env_idx} = {pure_rewards[epoch][env_idx][agent]:.2f}")

        if epoch % config["SAVE_EVERY_N_EPOCHS"] == 0:
            with open(os.path.join(path, f"params_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(params, f)

    return params, current_date