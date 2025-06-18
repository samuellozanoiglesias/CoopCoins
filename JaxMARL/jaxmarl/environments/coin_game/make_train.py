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
import equinox as eqx

def actor_critic_fn(obs_shape, n_actions, key):
    model = ActorCritic(obs_shape, n_actions, key)
    return model

def make_train(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path
    master_key = jax.random.PRNGKey(0)

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # Crear entornos
    keys_envs = jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
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
    agents = env.agents
    obs_shape = env.observation_space().shape
    n_actions = env.action_space(agents[0]).n

    models, models_params, opt_state, optimizer = {}, {}, {}, {}
    keys_agents = jax.random.split(jax.random.PRNGKey(0), config["NUM_AGENTS"])
    # === OPTIMIZER ===
    lr_schedule = optax.linear_schedule(
        init_value=config["LR"],
        end_value=config["LR"] * 0.1,
        transition_steps=config["NUM_EPOCHS"]
    )
    for i, agent in enumerate(agents):
        key = keys_agents[i]
        model = actor_critic_fn(obs_shape, n_actions, key)
        models[agent] = model
        models_params[agent] = eqx.filter(model, eqx.is_array)
        optimizer[agent] = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(lr_schedule)
        )
        opt_state[agent] = optimizer[agent].init(models_params[agent])

    # === ACTION FUNCTION ===
    @jit
    def select_action(model, obs, key):
        logits, value = model(obs)
        probs = jax.nn.softmax(logits)
        action = jax.random.categorical(key, logits)
        log_prob = jnp.log(probs[action])
        return action, log_prob, value
    
    # === LOSS FUNCTION ===
    @jit
    def compute_loss(model, obs, actions, advantages, returns, value_coef, entropy_coef, clip_eps, old_log_probs):
        logits, values = model(obs)
        log_probs = jax.nn.log_softmax(logits)
        chosen_log_probs = jnp.sum(log_probs * jax.nn.one_hot(actions, logits.shape[-1]), axis=-1)
        
        # PPO clipping
        ratio = jnp.exp(chosen_log_probs - old_log_probs)
        policy_loss1 = -ratio * advantages
        policy_loss2 = -jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = jnp.mean(jnp.maximum(policy_loss1, policy_loss2))
        
        # Value function clipping
        value_pred_clipped = values + jnp.clip(values - values, -clip_eps, clip_eps)
        value_losses = jnp.square(returns - values)
        value_losses_clipped = jnp.square(returns - value_pred_clipped)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
        
        entropy_loss = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
        return policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
    
    # === COMPUTE ADVANTAGE ===
    @jit
    def compute_gae(rewards, values, gamma, gae_lambda):
        next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
        deltas = rewards + gamma * next_values - values

        def scan_fn(carry, delta):
            gae = delta + gamma * gae_lambda *carry
            return gae, gae

        _, gaes = lax.scan(scan_fn, 0.0, deltas[::-1])
        gaes = gaes[::-1]
        returns = gaes + values
        return returns, gaes
    
    # === TRAINING UPDATE ===
    def train_minibatches(model, opt_state, obs_batch, action_batch, advantage, return_batch, key, optimizer, minibatch_size, epochs, clip_eps, old_log_probs):
        num_minibatches = obs_batch.shape[0] // minibatch_size

        def compute_loss_batch(model, obs, actions, adv, returns):
            def single_loss(m, o, a, ad, r):
                return compute_loss(m, o, a, ad, r, config["VF_COEF"], config["ENT_COEF"], clip_eps, old_log_probs)
            losses = jax.vmap(single_loss, in_axes=(None, 0, 0, 0, 0))(model, obs, actions, adv, returns)
            return losses.mean()

        def epoch_step(carry, _):
            model, opt_state, key = carry
            key, subkey = random.split(key)
            perm = random.permutation(subkey, obs_batch.shape[0])

            def minibatch_step(carry, i):
                model, opt_state = carry
                start = i * minibatch_size
                idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))
                batch_obs = obs_batch[idx]
                batch_actions = action_batch[idx]
                batch_adv = advantage[idx]
                batch_returns = return_batch[idx]

                loss, grads = value_and_grad(compute_loss_batch)(model, batch_obs, batch_actions, batch_adv, batch_returns)
                # Compute gradient norm for monitoring
                grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
                updates, opt_state = optimizer.update(grads, opt_state)
                model = optax.apply_updates(model, updates)
                return (model, opt_state), (loss, grad_norm)

            (model, opt_state), (losses, grad_norms) = lax.scan(minibatch_step, (model, opt_state), jnp.arange(num_minibatches))
            return (model, opt_state, key), (losses, grad_norms)

        (model, opt_state, _), (losses, grad_norms) = lax.scan(epoch_step, (model, opt_state, key), None, length=epochs)
        return model, opt_state, (losses, grad_norms)

    # Create a JIT-compiled version with static optimizer, epochs, and minibatch_size
    train_minibatches_jit = jit(train_minibatches, static_argnums=(7, 8, 9), static_argnames=('optimizer', 'minibatch_size', 'epochs'))
    
    # === LOGGING INIT ===
    states = [env.reset(k)[1] for env, k in zip(envs, keys_envs)]
    observations = [env.reset(k)[0] for env, k in zip(envs, keys_envs)]
    csv_path = os.path.join(path, "training_stats.csv")
    write_header = not os.path.exists(csv_path)

    # === TRAINING LOOP ===
    for epoch in range(config["NUM_EPOCHS"]):
        master_key, subkey = jax.random.split(master_key)
        keys_envs = jax.random.split(subkey, config["NUM_ENVS"])

        for env_idx, env in enumerate(envs):
            state = states[env_idx]
            obs_env = observations[env_idx]
            done = False
            key = keys_envs[env_idx]

            trajectory = {agent: {"obs": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
                          for agent in env.agents}

            while not done:
                key, *subkeys = jax.random.split(master_key, config["NUM_AGENTS"] + 1)
                master_key = key
                actions, values, log_probs = {}, {}, {}

                for i, agent in enumerate(env.agents):
                    obs_agent = jnp.array(obs_env[agent])[None, ...]
                    subkey = subkeys[i]
                    action, log_prob, value = select_action(models[agent], obs_agent, subkey)
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

                returns, advantage = compute_gae(reward_batch, value_batch, config["GAMMA"], config["GAE_LAMBDA"])
                advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

                models[agent], opt_state[agent], (losses, grad_norms) = train_minibatches_jit(
                    models[agent],
                    opt_state[agent],
                    obs_batch,
                    action_batch,
                    advantage,
                    returns,
                    key,
                    optimizer=optimizer[agent],
                    minibatch_size=config["MINIBATCH_SIZE"],
                    epochs=config["NUM_UPDATES_PER_MINIBATCH"],
                    clip_eps=config["CLIP_EPS"],
                    old_log_probs=jnp.array(trajectory[agent]["log_probs"])
                )

                # Add loss and gradient monitoring to stats
                stats = infos
                stats[agent]["mean_loss"] = float(jnp.mean(losses))
                stats[agent]["mean_grad_norm"] = float(jnp.mean(grad_norms))

            row = {
                "epoch": epoch,
                "env": env_idx,
            }

            for agent in env.agents:
                row.update({
                    f"reward_{agent}": float(infos[agent]["cumulated_modified_reward"]),
                    f"pure_reward_{agent}": float(infos[agent]["cumulated_pure_reward"]),
                    f"mean_loss_{agent}": float(stats[agent]["mean_loss"]),
                    f"mean_grad_norm_{agent}": float(stats[agent]["mean_grad_norm"]),
                })
                a_stats = infos[agent]["cumulated_action_stats"]
                row.update({
                    f"own_coin_collected_{agent}": int(a_stats[0]),
                    f"other_coin_collected_{agent}": int(a_stats[1]),
                    f"reject_own_coin_{agent}": int(a_stats[2]),
                    f"reject_other_coin_{agent}": int(a_stats[3]),
                    f"no_coin_visible_{agent}": int(a_stats[4]),
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
            with open(os.path.join(path, f"model_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(models, f)

    return models, current_date