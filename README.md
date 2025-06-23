# CoopCoins 🪙

A simulated environment adapted from the Coin Game for training, analyzing, and visualizing Multi-Agent Reinforcement Learning (MARL) models with different cooperative characteristics.

## Overview

CoopCoins is a multi-agent grid-world environment that simulates social dilemmas similar to the Iterated Prisoner's Dilemma (IPD) but with high-dimensional dynamic states. The environment is designed to study cooperative behavior, social dilemmas, and opponent shaping in multi-agent systems.

## Game Mechanics

The environment consists of two agents (red and blue) moving on a grid (default 3x3) to collect coins of their respective colors:

- **Own Coin Collection**: +1 reward for collecting your own color coin
- **Other Coin Collection**: -2 penalty for the other agent when you collect their coin
- **Social Dilemma**: If both agents play greedily, the expected reward for both is 0
- **Cooperation Opportunity**: Agents can learn to cooperate by avoiding each other's coins

## Key Features

### 🎯 **Flexible Reward Structure**
- Configurable reward coefficients for different cooperative attitudes
- Support for both prisoner's dilemma and non-dilemma scenarios
- Customizable payoff matrices

### 🧠 **Multiple Environment Variants**
- **Standard Coin Game**: Full social dilemma implementation
- **RLlib Integration**: Compatible with Ray RLlib framework

### 📊 **Comprehensive Analysis Tools**
- Training statistics tracking and visualization
- Behavioral analysis of agent strategies
- Performance comparison across different configurations
- Automated result aggregation and plotting

### 🎨 **Visualization Capabilities**
- Real-time episode visualization
- GIF generation of agent interactions
- Behavioral pattern analysis
- Training progress monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coopcoins.git
cd coopcoins

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```python
from coin_game.training import make_train

# Configure training parameters
config = {
    "NUM_ENVS": 4,
    "NUM_INNER_STEPS": 300,
    "NUM_EPOCHS": 100000,
    "LR": 0.001,
    "GRID_SIZE": 3,
    "REWARD_COEF": [[1.0, 0.0], [1.0, 0.0]],  # Selfish agents
    "PAYOFF_MATRIX": [[1, 0, 0], [1, 0, 0]]
}

# Start training
params, current_date = make_train(config)
```

### Using Pre-configured Attitudes

```bash
# Train with different cooperative attitudes
python coin_game/launch_training.py inputs/inputs_0_0.txt 0 0.001 3    # Selfish
python coin_game/launch_training.py inputs/inputs_45_45.txt 0 0.001 3  # Cooperative
python coin_game/launch_training.py inputs/inputs_90_0.txt 0 0.001 3   # Altruistic
```

### Visualization

```python
# Visualize trained models
python coin_game/example_visualization.py /path/to/checkpoint --episodes 3
```

## Configuration

### Reward Coefficients

The `REWARD_COEF` parameter controls agent attitudes:

```python
REWARD_COEF = [[alpha_1, beta_1], [alpha_2, beta_2]]
```

Where:
- `alpha_i`: Weight for agent i's own reward
- `beta_i`: Weight for the other agent's reward

Common configurations:
- `[[1, 0], [1, 0]]`: Selfish agents (default)
- `[[0.7, 0.3], [0.7, 0.3]]`: Cooperative agents
- `[[0.5, 0.5], [0.5, 0.5]]`: Altruistic agents

### Environment Parameters

- `GRID_SIZE`: Size of the grid (default: 3)
- `NUM_INNER_STEPS`: Steps per episode (default: 10)
- `NUM_EPOCHS`: Total training epochs
- `PAYOFF_MATRIX`: Reward structure for different actions

## Analysis and Visualization

### Training Analysis

```python
# Run comprehensive analysis
python coin_game/analysis.ipynb
```

This generates:
- Training progress plots
- Behavioral analysis
- Performance comparisons
- Statistical summaries

### Model Visualization

```python
from coin_game.visualize_rllib_models import visualize_episode

# Visualize agent behavior
visualize_episode(
    trainer, 
    config, 
    num_episodes=1,
    save_gif=True,
    output_dir="visualizations"
)
```

## Project Structure

```
coopcoins/
├── coin_game/                 # Main environment and training code
│   ├── coin_game.py          # Standard Coin Game implementation
│   ├── coin_game_rllib_env.py # RLlib integration
│   ├── training.py           # Training script
│   ├── analysis.ipynb        # Analysis notebook
│   ├── visualization.ipynb   # Visualization notebook
│   ├── inputs/               # Pre-configured attitude files
│   └── logs/                 # Training logs
├── JaxMARL/                  # JAX-based MARL framework
│   └── jaxmarl/             # Core MARL implementation
└── README.md                # This file
```

## Research Applications

CoopCoins is particularly useful for studying:

- **Social Dilemmas**: How agents learn to cooperate or defect
- **Opponent Shaping**: Learning to influence other agents' behavior
- **Emergent Cooperation**: Spontaneous development of cooperative strategies
- **Multi-Agent Learning**: Coordination and competition dynamics

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Citation

If you use CoopCoins in your research, please cite:

```bibtex
@misc{coopcoins2024,
  title={CoopCoins: A Multi-Agent Environment for Studying Cooperative Behavior},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/coopcoins}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original Coin Game from [Lerer & Peysakhovich (2017)](https://arxiv.org/abs/1707.01068)
- Built on the [JaxMARL](https://github.com/flairox/jax-marl) framework
- Inspired by research on opponent shaping and social dilemmas

---

**Happy cooperating! 🤝**
