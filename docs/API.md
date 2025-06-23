# CoopCoins API Documentation

This document describes the API for the CoopCoins environment and related components.

## Environment API

### CoinGame

The main Coin Game environment class.

```python
class CoinGame(MultiAgentEnv):
    def __init__(
        self,
        num_inner_steps: int = 10,
        num_outer_steps: int = 10,
        cnn: bool = False,
        egocentric: bool = False,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        grid_size: int = 3,
        reward_coef=[[1,0],[1,0]]
    ):
```

#### Parameters

- **num_inner_steps** (int): Number of steps per episode (default: 10)
- **num_outer_steps** (int): Number of episodes per outer episode (default: 10)
- **cnn** (bool): Whether to use CNN observations (default: False)
- **egocentric** (bool): Whether to use egocentric observations (default: False)
- **payoff_matrix** (list): Reward structure matrix (default: [[1, 1, -2], [1, 1, -2]])
- **grid_size** (int): Size of the grid (default: 3)
- **reward_coef** (list): Cooperative attitude coefficients (default: [[1,0],[1,0]])

#### Methods

##### `reset(key: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState]`

Reset the environment to initial state.

**Parameters:**
- **key** (jnp.ndarray): Random key for initialization

**Returns:**
- **obs** (dict): Initial observations for each agent
- **state** (EnvState): Initial environment state

##### `step(key: jnp.ndarray, state: EnvState, actions: Tuple[int, int]) -> Tuple[jnp.ndarray, EnvState, dict, dict, dict]`

Take a step in the environment.

**Parameters:**
- **key** (jnp.ndarray): Random key for stochasticity
- **state** (EnvState): Current environment state
- **actions** (Tuple[int, int]): Actions for each agent

**Returns:**
- **obs** (dict): Observations for each agent
- **next_state** (EnvState): Next environment state
- **rewards** (dict): Rewards for each agent
- **dones** (dict): Done flags for each agent
- **infos** (dict): Additional information

#### Properties

- **name** (str): Environment name ("CoinGame-v1")
- **num_actions** (int): Number of possible actions (5)
- **action_space** (spaces.Discrete): Action space
- **observation_space** (spaces.Box): Observation space

### CoinGameRLLibEnv

RLlib-compatible wrapper for the Coin Game.

```python
class CoinGameRLLibEnv(MultiAgentEnv):
    def __init__(
        self,
        num_inner_steps: int = 10,
        num_outer_steps: int = 10,
        cnn: bool = False,
        egocentric: bool = False,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        grid_size: int = 3,
        reward_coef=[[1,0],[1,0]],
        path: str = "episode_log.csv",
        env_idx: int = 0
    ):
```

#### Additional Parameters

- **path** (str): Path for logging episode data (default: "episode_log.csv")
- **env_idx** (int): Environment index for parallel training (default: 0)

## Training API

### make_train

Main training function for the Coin Game.

```python
def make_train(config: dict) -> Tuple[Any, str]:
```

#### Parameters

- **config** (dict): Training configuration dictionary

#### Configuration Keys

- **NUM_ENVS** (int): Number of parallel environments
- **NUM_INNER_STEPS** (int): Steps per episode
- **NUM_EPOCHS** (int): Total training epochs
- **NUM_AGENTS** (int): Number of agents (always 2)
- **SHOW_EVERY_N_EPOCHS** (int): How often to show progress
- **SAVE_EVERY_N_EPOCHS** (int): How often to save checkpoints
- **LR** (float): Learning rate
- **PAYOFF_MATRIX** (list): Reward structure matrix
- **GRID_SIZE** (int): Grid size
- **REWARD_COEF** (list): Cooperative attitude coefficients
- **SAVE_DIR** (str): Directory to save results
- **GAMMA** (float): Discount factor (default: 0.9)
- **GAE_LAMBDA** (float): GAE lambda parameter (default: 0.95)
- **ENT_COEF** (float): Entropy coefficient (default: 0.15)
- **CLIP_EPS** (float): PPO clip parameter (default: 0.1)
- **VF_COEF** (float): Value function coefficient (default: 0.7)
- **MAX_GRAD_NORM** (float): Gradient clipping (default: 0.5)
- **MINIBATCH_SIZE** (int): Minibatch size
- **NUM_UPDATES_PER_MINIBATCH** (int): Updates per minibatch
- **DEVICE** (list): JAX devices to use

#### Returns

- **params** (Any): Trained model parameters
- **current_date** (str): Timestamp of training completion

## Analysis API

### Training Analysis

The analysis module provides functions for analyzing training results.

```python
def analyze_training_results(results_dir: str) -> pd.DataFrame:
    """Analyze training results from a directory."""
    pass

def plot_training_progress(df: pd.DataFrame, save_path: str = None):
    """Plot training progress metrics."""
    pass

def compare_attitudes(results_dirs: List[str], save_path: str = None):
    """Compare results across different cooperative attitudes."""
    pass
```

### Visualization API

```python
def visualize_episode(
    trainer, 
    config, 
    num_episodes: int = 1,
    save_gif: bool = True,
    output_dir: str = "visualizations"
):
    """Visualize episodes using a trained model."""
    pass

def generate_behavior_plots(
    results_dir: str,
    output_dir: str = "plots"
):
    """Generate behavioral analysis plots."""
    pass
```

## Configuration Format

### Attitude Configuration Files

Text files containing reward coefficients:

```
alpha_1 beta_1
alpha_2 beta_2
```

Where:
- `alpha_i`: Weight for agent i's own reward
- `beta_i`: Weight for the other agent's reward

### Training Configuration Files

YAML files containing training parameters:

```yaml
training:
  num_envs: 4
  num_inner_steps: 300
  num_epochs: 100000
  # ... other training parameters

hyperparameters:
  learning_rate: 0.001
  gamma: 0.9
  # ... other hyperparameters

environment:
  grid_size: 3
  payoff_matrix: [[1, 1, -2], [1, 1, -2]]
  reward_coef: [[1.0, 0.0], [1.0, 0.0]]
  dilemma: false

paths:
  save_dir: "./results"
  log_dir: "./logs"
```

## Usage Examples

### Basic Environment Usage

```python
import jax
from jaxmarl.environments.coin_game.coin_game import CoinGame

# Create environment
env = CoinGame(
    grid_size=3,
    reward_coef=[[1.0, 0.0], [1.0, 0.0]],  # Selfish agents
    num_inner_steps=10
)

# Reset environment
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Take actions
actions = (0, 0)  # Both agents move right
obs, state, rewards, dones, infos = env.step(key, state, actions)
```

### Training Example

```python
from jaxmarl.environments.coin_game.make_train import make_train

config = {
    "NUM_ENVS": 4,
    "NUM_INNER_STEPS": 300,
    "NUM_EPOCHS": 100000,
    "LR": 0.001,
    "GRID_SIZE": 3,
    "REWARD_COEF": [[1.0, 0.0], [1.0, 0.0]],
    "SAVE_DIR": "./results"
}

params, timestamp = make_train(config)
```

### Analysis Example

```python
import pandas as pd
from coin_game.analysis import analyze_training_results

# Analyze results
df = analyze_training_results("./results")

# Plot progress
plot_training_progress(df, save_path="./plots/training_progress.png")
``` 