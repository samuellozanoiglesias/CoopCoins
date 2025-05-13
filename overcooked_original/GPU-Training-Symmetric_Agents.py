import os
#os.environ["JAX_PLATFORMS"] = "cpu"  # Forces JAX to run on CPU

import jax #pip install jax
print(f"JAX is using: {jax.devices()[0]}")
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax #pip install distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl import make
import hydra #pip install hydra-core
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

### Added to modify the structure of the definitions
from functools import partial

import jax.lax as lax
import pickle
from datetime import datetime
from PIL import Image
import time
from textwrap import dedent

# Save the model parameters only
def save_model(runner_state, save_dir, file):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the full file path
    filename = os.path.join(save_dir, file)

    # Extract train_state (first element of the tuple)
    train_state = runner_state[0]

    # Save only the model parameters
    with open(filename, 'wb') as f:
        pickle.dump({'params': train_state.params}, f)

    print(f"Model parameters saved to {filename}")

# Load the model parameters
def load_model(load_dir, file, network):
    filename = os.path.join(load_dir, file)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")

    with open(filename, "rb") as f:
        saved_data = pickle.load(f)

    # Ensure the loaded model has the correct parameter structure
    if "params" not in saved_data:
        raise ValueError("Invalid saved model file: missing 'params'.")

    model_params = saved_data["params"]
    print(f"Model parameters loaded from {filename}")

    return model_params

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

def find_closest_checkpoint(fixed_step, load_dir):
    # List all files in the directory
    files = os.listdir(load_dir)
    
    # Filter files that match the pattern 'trained_model_{step}.pkl'
    checkpoint_files = [f for f in files if f.startswith("trained_model_") and f.endswith(".pkl")]
    
    # Extract the step numbers from the filenames
    steps = []
    for file in checkpoint_files:
        step_str = file.split('_')[-1].split('.')[0]  # Extract the step number
        try:
            step = int(step_str)
            steps.append(step)
        except ValueError:
            continue  # Skip files that don't have a valid step number
    
    if len(steps) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {load_dir}")
    
    # Find the closest step to the fixed step
    closest_step = min(steps, key=lambda x: abs(x - fixed_step))
    closest_filename = f"trained_model_{closest_step}.pkl"
    
    print(f"Found checkpoint: {closest_filename} for step {closest_step}")
    
    # Load the model (assuming load_model function is available)
    loaded_params = load_model(load_dir, closest_filename, network)
    
    return loaded_params, closest_step

import imageio.v3 as iio

def custom_animate(state_seq, agent_view_size, filename="animation.gif"):
    """Animate a GIF give a state sequence and save if to file."""
    import imageio
    TILE_PIXELS = 32

    padding = agent_view_size - 2  # show

    def get_frame(state):
            grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the state
            frame = OvercookedVisualizer._render_grid(
                    grid,
                    tile_size=TILE_PIXELS,
                    highlight_mask=None,
                    agent_dir_idx=state.agent_dir_idx,
                    agent_inv=state.agent_inv
            )
            return frame

    frame_seq =[get_frame(state) for state in state_seq]
    
    imageio.mimsave(filename, frame_seq, 'GIF', duration=2)
    
def gif_to_mp4(gif_filename):
    """Convert a GIF to MP4 using imageio and ffmpeg."""
    mp4_filename = gif_filename.replace(".gif", ".mp4")
    gif = iio.imread(gif_filename)
    fps = 2  # Set desired FPS

    iio.imwrite(mp4_filename, gif, extension=".mp4", fps=fps)
    print(f"MP4 saved to {mp4_filename}")

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu

        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def reshape_info(info):
    def reshape_fn(x):
        if isinstance(x, dict):
            return jax.tree.map(lambda y: y.reshape(-1), x)
        # For (16, 2) shaped arrays, reshape to (32,)
        elif len(x.shape) == 2 and x.shape[1] == 2:
            return x.reshape(-1)
        # For (16,) shaped arrays
        else:
            # Repeat each element twice since we have 2 agents
            return jnp.repeat(x, 2)
    return jax.tree.map(reshape_fn, info)

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_config(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    print(f'LR: {config["LR"]}')
    print(f'NUM_ENVS: {config["NUM_ENVS"]}')
    print(f'NUM_STEPS: {config["NUM_STEPS"]}')
    print(f'TOTAL_TIMESTEPS: {config["TOTAL_TIMESTEPS"]}')
    print(f'UPDATE_EPOCHS: {config["UPDATE_EPOCHS"]}')
    print(f'NUM_MINIBATCHES: {config["NUM_MINIBATCHES"]}')

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    print(f'NUM_ACTORS: {config["NUM_ACTORS"]}')

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    print(f'NUM_UPDATES: {config["NUM_UPDATES"]}')

    config["NUM_SAVES"] = int(config["NUM_UPDATES"] // config["SAVE_EVERY_N_EPOCHS"])
    print(f'NUM_SAVES: {config["NUM_SAVES"]}')

    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    print(f'MINIBATCH_SIZE: {config["MINIBATCH_SIZE"]}')
    return config


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    env = LogWrapper(env)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(seed, rng, save_dir):

        # INIT NETWORK

        # Creates an instance of the ActorCritic model.
        # The ActorCritic class is initialized with:
        # env.action_space().n: The number of possible actions in the environment.
        # config["ACTIVATION"]: The type of activation function to use (e.g., ReLU or Tanh).
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        # Splits the random number generator rng into two separate RNGs (rng and _rng), so they can be used independently.
        rng, _rng = jax.random.split(rng)

        # Creates a zero-initialized array of the same shape as the observation space of the environment (env.observation_space().shape), then flattens it into a 1D array (init_x).
        # This serves as a sample input to initialize the network.
        init_x = jnp.zeros(env.observation_space().shape)
        init_x = init_x.flatten()

        # Initializes the parameters of the network (network_params) by passing the random key _rng and the input init_x (observation example).
        # This is necessary to set up the weights and biases of the neural network layers.
        network_params = network.init(_rng, init_x)

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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        # Splits the RNG (rng) into two new RNGs: one (_rng) used for resetting the environment, and another (reset_rng) for each environment if you're running multiple environments in parallel (config["NUM_ENVS"]).
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        # Initializes the environment(s). jax.vmap is used to vectorize the env.reset function, so it can reset config["NUM_ENVS"] environments in parallel.
        # reset_rng: The random keys for resetting each environment are passed in here.
        # obsv: The initial observations returned by the environment(s).
        # env_state: The initial state of the environment(s).
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        @jax.jit
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES

            # Action Selection: It uses the current policy (pi) to sample an action based on the current observations.
            # Environment Step: It then steps the environment using the selected actions, receiving observations, rewards, done flags, and additional info.
            # Transition Recording: A transition is created, which includes the action taken, value estimate, rewards, and log probabilities for the action. This transition is used to calculate losses later.

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = {k:v.flatten() for k,v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                info = reshape_info(info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info

                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            
            # This line performs the _env_step function repeatedly for NUM_STEPS. jax.lax.scan is a JAX function that allows you to apply a function repeatedly over a sequence of data, which is used here to simulate multiple environment interactions (collecting trajectories for NUM_STEPS).
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            # After collecting the trajectory, this part calculates the advantages of each transition using GAE. The last observation (last_obs) is passed through the network to get the value (last_val) for the last state.
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)
            
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                # This function updates the model by applying the computed gradients and losses.
                # Loss Calculation: The loss function has two parts:
                # Value loss: The loss for the value function is the mean squared error between predicted values and the target values (calculated using GAE).
                # Actor loss: The loss for the policy (actor) is based on the surrogate objective from the PPO (Proximal Policy Optimization) algorithm, involving the ratio between the current and previous probabilities of actions.
                # Entropy loss: A term to encourage exploration by adding entropy to the objective function.
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # This line performs the epoch updates using the scan function to apply the _update_epoch function across all epochs.
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            # After all epochs of training are completed, the train_state (the updated model) is returned along with metrics (e.g., information about rewards, losses, etc.), and the random number generator (rng) is updated.
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        # This line splits the rng (random number generator) into two separate random number generators: rng and _rng. This is done so that each part of the code can use a different random stream. 
        rng, _rng = jax.random.split(rng)

        # Here, the runner_state is initialized as a tuple that includes:
        # train_state: The current state of the model (including parameters and optimization state).
        # env_state: The current state of the environment (e.g., positions, internal states of agents).
        # obsv: The current batch of observations (e.g., the states that the agents are observing from the environment).
        # _rng: The random number generator that will be used for further random operations in the loop.
        runner_state = (train_state, env_state, obsv, _rng)

        # This line performs an update loop using jax.lax.scan. Here’s how it works:
        # jax.lax.scan is a JAX function that allows you to loop over some operation while maintaining the state between iterations, making it ideal for iterative processes like training loops.
        # The _update_step function is applied iteratively, where each iteration updates the state of the runner (i.e., the model and environment) and records the metrics (e.g., loss, performance metrics).
        # runner_state: This contains all the information required for each update step (including train_state, env_state, etc.).
        # None: The second argument is None because the loop doesn’t need any additional data passed each time; it's just evolving the state.
        # config["NUM_UPDATES"]: This specifies how many times the _update_step function will be applied. Essentially, this determines how many updates (iterations) will be made to the model.
        #runner_state, metric = jax.lax.scan(
        #    _update_step, runner_state, None, config["NUM_UPDATES"]
        #)

        metrics = {}
        save_intervals = max(1, config["NUM_UPDATES"] // config["NUM_SAVES"])

        for step in range(config["NUM_UPDATES"]):
            runner_state, metric = _update_step(runner_state, step)

            if step % save_intervals == 0 or step == config["NUM_UPDATES"] - 1:
                save_model(runner_state, save_dir, f"trained_model_{step}.pkl")

            # Store each metric separately in the dictionary
            if not metrics:  # Initialize keys on the first iteration
                metrics = {key: [] for key in metric}

            for key in metric:
                metrics[key].append(metric[key])  # Append new values for each key

            if step % 100 == 0:
                print(f"Seed {seed}, step {step} completed")

        # After executing the loop, runner_state and metric will be updated:
        # runner_state: The final state of the model, environment, and random number generator after all the updates.
        # metric: A collection of metrics generated during the updates (e.g., losses, rewards, or performance indicators).
        return {"runner_state": runner_state, "metrics": jax.tree.map(lambda x: jnp.stack(x), metrics)}

    return train

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
#layout_name = "forced_coord"

# Or make your own! (P=pot, O=onion, A=agent, B=plates, X=deliver)
layout_name = input("Introduce layout_name: ")

if layout_name == "custom_1":
    custom_layout_grid = dedent(
    """
    WWBWW
    WA AW
    P   P
    W   W
    WOXOW
    """
    ).strip()
    custom_layout = layout_grid_to_dict(custom_layout_grid)

elif layout_name == "custom_2":
    custom_layout_grid = dedent(
    """
    WWBWW
    WA AW
    P   W
    P   W
    WOXOW
    """
    ).strip()
    custom_layout = layout_grid_to_dict(custom_layout_grid)

elif layout_name == "custom_3":
    custom_layout_grid = dedent(
    """
    WBWWBW
    P    P
    W AA W
    O    O
    WXWWXW
    """
    ).strip()
    custom_layout = layout_grid_to_dict(custom_layout_grid)

elif layout_name == "custom_4":
    custom_layout_grid = dedent(
    """
    WBWXW
    P W O
    WAWAW
    P W O
    WBWXW
    """
    ).strip()
    custom_layout = layout_grid_to_dict(custom_layout_grid)

elif layout_name == "custom_5":
    custom_layout_grid = dedent(
    """
    WPWPW
    W B W
    WAWAW
    W O W
    WXWXW
    """
    ).strip()
    custom_layout = layout_grid_to_dict(custom_layout_grid)

else:
    layout_name = layout_name

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
initial_dir = f'/data/samuel_lozano/hfsp_collective_learning/data_JaxMARL/Symmetric_Agents/{layout_name}/Checkpoints_{current_datetime}/'

print(layout_name)
print(current_datetime)

# set hyperparameters:
config = {
    "NUM_ACTIONS": 6,
    "LR": 5e-2,
    "NUM_ENVS": 8,
    "NUM_STEPS": 3000,
    "TOTAL_TIMESTEPS": 5e9,
    "UPDATE_EPOCHS": 1000,
    "NUM_MINIBATCHES": 100,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.99,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "overcooked",
    "ENV_KWARGS": {
      "layout" : custom_layout #write custom_layout without quotation marks if you want to use the custom layout
    },
    "ANNEAL_LR": True,
    "SEED": 0,
    "SAVE_EVERY_N_EPOCHS": 50,
    "NUM_SEEDS": 3
}

#Comment the following line if you want to use the custom_layout
#config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]

config = make_config(config)

rng = jax.random.PRNGKey(config["SEED"])
rngs = jax.random.split(rng, config["NUM_SEEDS"])

# Training loop
for seed in range(config["NUM_SEEDS"]):
    save_dir = f'{initial_dir}Seed_{seed}/'
    train_jit = make_train(config)
    result = train_jit(seed, rngs[seed], save_dir)
    np.savez(os.path.join(initial_dir, f"plot_data_seed_{seed}.npz"), 
             returns=result["metrics"]["returned_episode_returns"])

    print(f"Data saved in {initial_dir}/plot_data_seed_{seed}.npz")
