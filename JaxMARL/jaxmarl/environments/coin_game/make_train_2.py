import os
import jax
import jax.numpy as jnp
from jax import lax
import optax
from gymnax.wrappers.purerl import LogWrapper
from flax.training.train_state import TrainState
from jaxmarl import make
from jaxmarl.environments.coin_game.actor_critic import ActorCritic, Transition
from jaxmarl.environments.coin_game.utils import create_training_directory, save_config, save_params, log_training_stats

def reshape_info(info):
    def reshape_fn(x):
        if isinstance(x, dict):
            return jax.tree.map(lambda y: y.reshape(-1), x)
        # For (8, 2) shaped arrays, reshape to (16,)
        elif len(x.shape) == 2 and x.shape[1] == 2:
            return x.reshape(-1)
        # For (8,) shaped arrays
        else:
            # Repeat each element twice since we have 2 agents
            return jnp.repeat(x, 2)
    return jax.tree.map(reshape_fn, info)

def split_info(info):
    def split_fn_agent_0(x):
        if isinstance(x, dict):
            return jax.tree.map(lambda y: y[:, 0], x)
        # For (8, 2) shaped arrays, reshape to (16,)
        elif len(x.shape) == 2 and x.shape[1] == 2:
            return x[:, 0]
        # For (8,) shaped arrays
        else:
            # Repeat each element twice since we have 2 agents
            return x

    def split_fn_agent_1(x):
        if isinstance(x, dict):
            return jax.tree.map(lambda y: y[:, 1], x)
        # For (8, 2) shaped arrays, reshape to (16,)
        elif len(x.shape) == 2 and x.shape[1] == 2:
            return x[:, 1]
        # For (8,) shaped arrays
        else:
            # Repeat each element twice since we have 2 agents
            return x

    return jax.tree.map(split_fn_agent_0, info), jax.tree.map(split_fn_agent_1, info)

def merge_info(info_1, info_2):
    def merge_fn(x_1, x_2):
        if isinstance(x_1, dict):
            # If the values are dictionaries, recursively merge them
            return jax.tree.map(merge_fn, x_1, x_2)
        else:
            # For array values, stack them along a new axis (axis=1)
            return jnp.stack([x_1, x_2], axis=1)

    # Use tree map to merge corresponding elements from both dictionaries
    return jax.tree.map(merge_fn, info_1, info_2)

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])

    env = LogWrapper(env)

    def train(seed, rng, initial_dir, csv_file_path):
        # INIT NETWORK
        network_1 = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        network_2 = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        _rng_1, _rng_2 = jax.random.split(_rng)

        init_x = jnp.zeros(env.observation_space().shape)
        init_x = init_x.flatten()
        network_1_params = network_1.init(_rng_1, init_x)
        network_2_params = network_2.init(_rng_2, init_x)
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state_1 = TrainState.create(
            apply_fn=network_1.apply,
            params=network_1_params,
            tx=tx,
        )

        train_state_2 = TrainState.create(
            apply_fn=network_2.apply,
            params=network_2_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        @jax.jit
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state_1, train_state_2, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                _rng_1, _rng_2 = jax.random.split(_rng)

                # obs_batch = batchify(last_obs, env.agents, config["NUM_AGENTS"])
                pi_1, value_1 = network_1.apply(train_state_1.params, jnp.array(last_obs[env.agents[0]]).reshape((int(config["NUM_AGENTS"]/2), -1)))
                pi_2, value_2 = network_2.apply(train_state_2.params, jnp.array(last_obs[env.agents[1]]).reshape((int(config["NUM_AGENTS"]/2), -1)))

                action_1 = pi_1.sample(seed=_rng_1)
                log_prob_1 = pi_1.log_prob(action_1)

                action_2 = pi_2.sample(seed=_rng_2)
                log_prob_2 = pi_2.log_prob(action_2)

                action = jnp.concatenate([action_1, action_2], axis=0)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = {k:v.flatten() for k,v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                info_1, info_2 = split_info(info)

                transition_1 = Transition(
                    jnp.array(done[env.agents[0]]).reshape((int(config["NUM_AGENTS"]/2), -1)).squeeze(),
                    action_1,
                    value_1,
                    jnp.array(reward[env.agents[0]]).reshape((int(config["NUM_AGENTS"]/2), -1)).squeeze(),
                    log_prob_1,
                    jnp.array(last_obs[env.agents[0]]).reshape((int(config["NUM_AGENTS"]/2), -1)),
                    info_1
                )
                transition_2 = Transition(
                    jnp.array(done[env.agents[1]]).reshape((int(config["NUM_AGENTS"]/2), -1)).squeeze(),
                    action_2,
                    value_2,
                    jnp.array(reward[env.agents[1]]).reshape((int(config["NUM_AGENTS"]/2), -1)).squeeze(),
                    log_prob_2,
                    jnp.array(last_obs[env.agents[1]]).reshape((int(config["NUM_AGENTS"]/2), -1)),
                    info_2
                )
                runner_state = (train_state_1, train_state_2, env_state, obsv, rng)
                return runner_state, (transition_1, transition_2)

            # This line performs the _env_step function repeatedly for NUM_STEPS. jax.lax.scan is a JAX function that allows you to apply a function repeatedly over a sequence of data, which is used here to simulate multiple environment interactions (collecting trajectories for NUM_STEPS).
            runner_state, (traj_batch_1, traj_batch_2) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            # After collecting the trajectory, this part calculates the advantages of each transition using GAE. The last observation (last_obs) is passed through the network to get the value (last_val) for the last state.
            train_state_1, train_state_2, env_state, last_obs, rng = runner_state
            # last_obs_batch = batchify(last_obs, env.agents, config["NUM_AGENTS"])

            _, last_val_1 = network_1.apply(train_state_1.params, last_obs[env.agents[0]].reshape((int(config["NUM_AGENTS"]/2), -1)))
            _, last_val_2 = network_2.apply(train_state_2.params, last_obs[env.agents[1]].reshape((int(config["NUM_AGENTS"]/2), -1)))

            # This function implements the GAE algorithm to compute advantages and targets (the value targets).
            # delta is computed using the Bellman equation for each transition.
            # gae is calculated iteratively in reverse order of the trajectory, allowing for more stable learning by incorporating multiple future rewards.
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages_1, targets_1 = _calculate_gae(traj_batch_1, last_val_1)
            advantages_2, targets_2 = _calculate_gae(traj_batch_2, last_val_2)

            # UPDATE NETWORK 1
            def _update_epoch_agent_1(update_state, unused):

                # This function updates the model by applying the computed gradients and losses.
                # Loss Calculation: The loss function has two parts:
                # Value loss: The loss for the value function is the mean squared error between predicted values and the target values (calculated using GAE).
                # Actor loss: The loss for the policy (actor) is based on the surrogate objective from the PPO (Proximal Policy Optimization) algorithm, involving the ratio between the current and previous probabilities of actions.
                # Entropy loss: A term to encourage exploration by adding entropy to the objective function.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network_1.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # This code shuffles the data (traj_batch, advantages, targets) into minibatches for efficient training. The training data is reshaped, and the order is randomized to reduce bias during training.
                train_state, traj_batch, advantages, targets, rng, agent_num = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * int(config["NUM_AGENTS"]/2)
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                # This line performs the minibatch updates using the scan function to apply the _update_minbatch function across all minibatches.
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng, agent_num)
                return update_state, total_loss
            
            # UPDATE NETWORK 1
            def _update_epoch_agent_2(update_state, unused):

                # This function updates the model by applying the computed gradients and losses.
                # Loss Calculation: The loss function has two parts:
                # Value loss: The loss for the value function is the mean squared error between predicted values and the target values (calculated using GAE).
                # Actor loss: The loss for the policy (actor) is based on the surrogate objective from the PPO (Proximal Policy Optimization) algorithm, involving the ratio between the current and previous probabilities of actions.
                # Entropy loss: A term to encourage exploration by adding entropy to the objective function.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network_2.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # This code shuffles the data (traj_batch, advantages, targets) into minibatches for efficient training. The training data is reshaped, and the order is randomized to reduce bias during training.
                train_state, traj_batch, advantages, targets, rng, agent_num = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * int(config["NUM_AGENTS"]/2)
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                # This line performs the minibatch updates using the scan function to apply the _update_minbatch function across all minibatches.
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng, agent_num)
                return update_state, total_loss

            # This line performs the epoch updates using the scan function to apply the _update_epoch function across all epochs.
            rng, _rng = jax.random.split(rng)
            _rng_1, _rng_2 = jax.random.split(_rng)
     
            update_state_1 = (train_state_1, traj_batch_1, advantages_1, targets_1, _rng_1, 0)
            update_state_2 = (train_state_2, traj_batch_2, advantages_2, targets_2, _rng_2, 1)
            update_state_1, loss_info_1 = jax.lax.scan(
                _update_epoch_agent_1, update_state_1, None, config["UPDATE_EPOCHS"]
            )
            update_state_2, loss_info_2 = jax.lax.scan(
                _update_epoch_agent_2, update_state_2, None, config["UPDATE_EPOCHS"]
            )

            # After all epochs of training are completed, the train_state (the updated model) is returned along with metrics (e.g., information about rewards, losses, etc.), and the random number generator (rng) is updated.
            train_state_1 = update_state_1[0]
            train_state_2 = update_state_2[0]
            metric = merge_info(traj_batch_1.info, traj_batch_2.info)
            metric = reshape_info(metric) # remove if you want info separated by agent
            #rng = update_state_1[-2] # reduces chances of _rng_1 sequence repeating, but not strictly necessary

            runner_state = (train_state_1, train_state_2, env_state, last_obs, rng)
            return runner_state, metric

        # This line splits the rng (random number generator) into two separate random number generators: rng and _rng. This is done so that each part of the code can use a different random stream.
        rng, _rng = jax.random.split(rng)

        # Here, the runner_state is initialized as a tuple that includes:
        # train_state: The current state of the model (including parameters and optimization state).
        # env_state: The current state of the environment (e.g., positions, internal states of agents).
        # obsv: The current batch of observations (e.g., the states that the agents are observing from the environment).
        # _rng: The random number generator that will be used for further random operations in the loop.
        runner_state = (train_state_1, train_state_2, env_state, obsv, _rng)

        # This line performs an update loop using jax.lax.scan. Here’s how it works:
        # jax.lax.scan is a JAX function that allows you to loop over some operation while maintaining the state between iterations, making it ideal for iterative processes like training loops.
        # The _update_step function is applied iteratively, where each iteration updates the state of the runner (i.e., the model and environment) and records the metrics (e.g., loss, performance metrics).
        # runner_state: This contains all the information required for each update step (including train_state, env_state, etc.).
        # None: The second argument is None because the loop doesn’t need any additional data passed each time; it's just evolving the state.
        # config["NUM_UPDATES"]: This specifies how many times the _update_step function will be applied. Essentially, this determines how many updates (iterations) will be made to the model.
        # runner_state, metric = jax.lax.scan(
        #    _update_step, runner_state, None, config["NUM_UPDATES"]
        # )

        metrics = {}
        save_intervals = max(1, config["NUM_UPDATES"] // config["NUM_SAVES"])
        save_dir = f'{initial_dir}Seed_{seed}/'

        for step in range(config["NUM_UPDATES"]):
            runner_state, metric = _update_step(runner_state, step)

            if step % save_intervals == 0 or step == config["NUM_UPDATES"] - 1:
                save_model(runner_state, save_dir, f"trained_model_{step}.pkl")
            
            returns = metric["returned_episode_returns"]
            mean_returns = returns.mean(axis=-1).reshape(-1)

            # Save mean_returns in CSV file
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, step, mean_returns])

            if step % 100 == 0:
                print(f"Seed {seed}, step {step} completed")

        # After executing the loop, runner_state and metric will be updated:
        # runner_state: The final state of the model, environment, and random number generator after all the updates.
        # metric: A collection of metrics generated during the updates (e.g., losses, rewards, or performance indicators).
        return {"runner_state": runner_state, "metrics": jax.tree.map(lambda x: jnp.stack(x), metrics)}

    return train