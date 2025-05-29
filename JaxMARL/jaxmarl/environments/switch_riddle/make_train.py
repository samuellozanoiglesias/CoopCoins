import imageio.v3 as iio
import jax #pip install jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import csv
import os
import pickle
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax #pip install distrax
import jaxmarl
import hydra #pip install hydra-core
from omegaconf import OmegaConf
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.switch_riddle.actor_critic import ActorCritic, Transition
from functools import partial
import jax.lax as lax
from PIL import Image


def save_model(runner_state, save_dir, file):
    # Asegura que el directorio de guardado exista
    os.makedirs(save_dir, exist_ok=True)

    # Construye la ruta completa al archivo
    filename = os.path.join(save_dir, file)

    # Extrae los train_states (asumiendo que runner_state[0] es un dict de train_states por agente)
    train_states = runner_state[0]

    # Extraer los parámetros de cada train_state y guardarlos en un diccionario
    params_dict = {f'params_agent_{k}': ts.params for k, ts in train_states.items()}

    # Guarda solo los parámetros en un archivo pickle
    with open(filename, 'wb') as f:
        pickle.dump(params_dict, f)

    print(f"Model parameters saved to {filename}")

def choose_tx(config):
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    
    return tx

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

def split_info_by_agent(info_dict, agents):
    # info_dict: {'returned_episode': (NUM_ENVS, NUM_AGENTS), ...}
    # agents: ['agent_0', 'agent_1', ..., 'agent_N']
    per_agent_info = {}

    for i, agent in enumerate(agents):
        per_agent_info[agent] = {
            k: v[:, i]  # Selecciona la columna i (i-ésimo agente) → shape: (NUM_ENVS,)
            for k, v in info_dict.items()
        }

    return per_agent_info

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(seed, rng, initial_dir, csv_file_path):

        # INIT NETWORK
        networks = {}
        network_params = {}
        train_states = {}

        # Splits the random number generator rng into two separate RNGs (rng and _rng), so they can be used independently.
        rng, _rng = jax.random.split(rng)
        agent_rngs = jax.random.split(_rng, len(env.agents))

        # This serves as a sample input to initialize the network.
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape).flatten()

        # Creates an instance of the ActorCritic model.
        # The ActorCritic class is initialized with:
        # env.action_space().n: The number of possible actions in the environment.
        # config["ACTIVATION"]: The type of activation function to use (e.g., ReLU or Tanh).
        
        for i, agent in enumerate(env.agents):
            # Initializes the parameters of the network (network_params) by passing the random key _rng and the input init_x (observation example).
            action_space = env.action_space(agent)
            networks[agent] = ActorCritic(action_space.n, activation=config["ACTIVATION"])
            network_params[agent] = networks[agent].init(agent_rngs[i], init_x)

        # This block initializes the optimizer (tx) that will be used to update the network parameters.
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        # Creates a TrainState object that holds the model's parameters (network_params), the optimizer (tx), and the function to apply the model (network.apply).
        # This will be used for model training, including parameter updates.
        for agent in env.agents:
            train_states[agent] = TrainState.create(
                apply_fn=networks[agent].apply,
                params=network_params[agent],
                tx=tx,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        @jax.jit
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES

            # Action Selection: It uses the current policy (pi) to sample an action based on the current observations.
            # Environment Step: It then steps the environment using the selected actions, receiving observations, rewards, done flags, and additional info.
            # Transition Recording: A transition is created, which includes the action taken, value estimate, rewards, and log probabilities for the action. This transition is used to calculate losses later.

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                agent_rngs = jax.random.split(_rng, len(env.agents))

                action_dict = {}
                value_dict = {}
                log_prob_dict = {}
                transition_dict = {}

                for i, agent in enumerate(env.agents):
                    obs_agent = jnp.array(last_obs[agent]).reshape((int(config["NUM_ACTORS"] / env.num_agents), -1))
                    pi, value = networks[agent].apply(train_states[agent].params, obs_agent)

                    action = pi.sample(seed=agent_rngs[i])
                    log_prob = pi.log_prob(action)

                    action_dict[agent] = action
                    value_dict[agent] = value
                    log_prob_dict[agent] = log_prob

                env_act = unbatchify(jnp.concatenate([action_dict[a] for a in env.agents], axis=0), env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info_per_agent = split_info_by_agent(info, env.agents)

                for agent in env.agents:
                    print(agent)
                    transition_dict[agent] = Transition(
                        jnp.array(done[agent]).reshape((int(config["NUM_ACTORS"] / env.num_agents), -1)).squeeze(),
                        action_dict[agent],
                        value_dict[agent],
                        jnp.array(reward[agent]).reshape((int(config["NUM_ACTORS"] / env.num_agents), -1)).squeeze(),
                        log_prob_dict[agent],
                        jnp.array(last_obs[agent]).reshape((int(config["NUM_ACTORS"] / env.num_agents), -1)),
                        info_per_agent[agent]  # Aquí pasamos la info filtrada para este agente
                    )
                
                runner_state = (train_states, env_state, obsv, rng)
                return runner_state, transition_dict

            # This line performs the _env_step function repeatedly for NUM_STEPS. jax.lax.scan is a JAX function that allows you to apply a function repeatedly over a sequence of data, which is used here to simulate multiple environment interactions (collecting trajectories for NUM_STEPS).
            runner_state, traj_batch_dict = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            # After collecting the trajectory, this part calculates the advantages of each transition using GAE. The last observation (last_obs) is passed through the network to get the value (last_val) for the last state.
            train_states, env_state, last_obs, rng = runner_state
            # last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

            last_values = {}
            for agent in env.agents:
                obs_agent = jnp.array(last_obs[agent]).reshape((int(config["NUM_ACTORS"] / env.num_agents), -1))
                _, last_val = networks[agent].apply(train_states[agent].params, obs_agent)
                last_values[agent] = last_val

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

            advantages_dict = {}
            targets_dict = {}

            for agent in env.agents:
                advantages, targets = _calculate_gae(traj_batch_dict[agent], last_values[agent])
                advantages_dict[agent] = advantages
                targets_dict[agent] = targets

            # UPDATE NETWORK            
            def _update_epoch_agent(update_state, unused):
                # This function updates the model by applying the computed gradients and losses.
                # Loss Calculation: The loss function has two parts:
                # Value loss: The loss for the value function is the mean squared error between predicted values and the target values (calculated using GAE).
                # Actor loss: The loss for the policy (actor) is based on the surrogate objective from the PPO (Proximal Policy Optimization) algorithm, involving the ratio between the current and previous probabilities of actions.
                # Entropy loss: A term to encourage exploration by adding entropy to the objective function.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = networks[agent].apply(params, traj_batch.obs)
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
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
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
                    batch_size == config["NUM_STEPS"] * int(config["NUM_ACTORS"]/2)
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
            agent_rngs = jax.random.split(_rng, env.num_agents)
     
            update_states = []
            for i, agent in enumerate(env.agents):
                update_states.append((
                    train_states[agent], 
                    traj_batch_dict[agent], 
                    advantages_dict[agent], 
                    targets_dict[agent], 
                    agent_rngs[i], 
                    agent  # nombre o índice del agente
                ))

            def scan_epoch_over_agent(update_state, unused):
                return _update_epoch_agent(update_state, unused)


            final_update_states = []
            loss_infos = []
            for update_state in update_states:
                update_state, loss_info = jax.lax.scan(
                    scan_epoch_over_agent, update_state, None, config["UPDATE_EPOCHS"]
                )
                final_update_states.append(update_state)
                loss_infos.append(loss_info)

            # After all epochs of training are completed, the train_state (the updated model) is returned along with metrics (e.g., information about rewards, losses, etc.), and the random number generator (rng) is updated.
            for i, agent in enumerate(env.agents):
                train_states[agent] = final_update_states[i][0]
            metric = merge_info(*[traj_batch_dict[agent].info for agent in env.agents])
            metric = reshape_info(metric) # remove if you want info separated by agent
            #rng = update_state_1[-2] # reduces chances of _rng_1 sequence repeating, but not strictly necessary

            runner_state = (train_states[env.agents[0]], train_states[env.agents[1]], env_state, last_obs, rng)
            return runner_state, metric

        # This line splits the rng (random number generator) into two separate random number generators: rng and _rng. This is done so that each part of the code can use a different random stream.
        rng, _rng = jax.random.split(rng)

        # Here, the runner_state is initialized as a tuple that includes:
        # train_state: The current state of the model (including parameters and optimization state).
        # env_state: The current state of the environment (e.g., positions, internal states of agents).
        # obsv: The current batch of observations (e.g., the states that the agents are observing from the environment).
        # _rng: The random number generator that will be used for further random operations in the loop.
        runner_state = (train_states, env_state, obsv, _rng)

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