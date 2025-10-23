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

These instructions assume a Unix-like system (Linux, macOS). The project requires Python 3.10. Two common workflows are shown below: using conda to create an environment, or using a plain virtualenv with pip.

Option A — (recommended) Using conda

```bash
# Create and activate a conda environment with Python 3.10
conda create -n coopcoins python=3.10 -y
conda activate coopcoins

# Install pip (if not already present) and other tooling
conda install pip -y

# From the repository root
git clone https://github.com/samuellozanoiglesias/CoopCoins.git
cd CoopCoins

# Install runtime dependencies
pip install -r requirements.txt

# Install local editable package JaxMARL (required by CoopCoins)
# This will install the package in editable/development mode so changes in JaxMARL/ are available
pip install -e ./JaxMARL
```

Option B — Using virtualenv and pip

```bash
# Create a virtualenv (Python 3.10 must be available as `python3.10`)
python3.10 -m venv .venv
source .venv/bin/activate

# From the repository root
git clone https://github.com/samuellozanoiglesias/CoopCoins.git
cd CoopCoins

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install local editable package JaxMARL
pip install -e ./JaxMARL
```

Notes
- The `pip install -e ./JaxMARL` step installs the local JaxMARL package found in the `JaxMARL/` directory. This is required because CoopCoins depends on that package and the repository ships a copy for convenience and research reproducibility.
- If you prefer not to install the editable package, you can install JaxMARL from PyPI if a published version exists, but features in the shipped `JaxMARL/` folder may be required for some experiments.
- If you run into binary dependency issues (for example with JAX), consult the packages' documentation for platform-specific wheels and GPU support.

## Quick Start

### Generating attitude configuration files (scripts/generate_attitudes.py)

The repository ships a helper script at `scripts/generate_attitudes.py` that creates attitude configuration files (plain text) used by the training scripts. The script supports two modes:

- Angles mode: generate pairwise combinations of single-agent attitudes computed from angles (default).
- Predefined mode: generate pairwise combinations of named predefined attitudes only (no angles).

Files created by the script contain two lines, one per agent, with the agent's reward coefficients in the format:

alpha beta

So a file contains two lines (agent1 then agent2). Filenames are produced as `<attitudeA>_<attitudeB>.txt` (for example `angle_0_angle_45.txt` or `individualistic_cooperative.txt`).

Usage notes

- Default angles: `0,45,90,135,180,225,270,315`.
- Default output directory: when running from the repository root use `--output-dir configs/attitudes` (the script's default is set so running from `scripts/` writes to `../configs/attitudes`).

Examples

Generate all angle combinations (default):

```bash
# from repo root
python scripts/generate_attitudes.py --angles 0,45,315 --output-dir configs/attitudes

# or run in background and log output
# nohup python scripts/generate_attitudes.py --angles 0,45,315 --output-dir configs/attitudes > out_generate_attitudes.log 2>&1 &
```

Generate only specific predefined attitudes (no angles). Pass a comma-separated list of names or the special keyword `all` to include every predefined attitude:

```bash
# generate pairwise combinations only for the three named attitudes
python scripts/generate_attitudes.py --predefined individualistic,cooperative,competitive --output-dir configs/attitudes

# all predefined
python scripts/generate_attitudes.py --predefined all --output-dir configs/attitudes

# background example
# nohup python scripts/generate_attitudes.py --predefined individualistic,cooperative,competitive --output-dir configs/attitudes > out_generate_attitudes.log 2>&1 &
```

Available predefined names

```
individualistic
cooperative
altruistic
sacrificial
martyrial
destructive
spiteful
competitive
```

How these files are used

Pass a generated attitude file to the training entry point (example):

```bash
python scripts/training.py configs/attitudes/individualistic_cooperative.txt 0 0.001 3
```

If you need a different format or a custom sweep, you can edit `scripts/generate_attitudes.py` to change the naming or coefficient generation.

### Generating trained agents with different attitudes (scripts/training.py)

Launches a series of training experiments for every attitude file in a directory. It builds a command that calls `coin_game/trainer.py` for each attitude and writes logs to a per-experiment file. Useful flags include `--configs` to point to the attitudes directory, `--dilemma` to toggle the game variant, `--lr` and `--grid-size` to set hyperparameters, and `--dry-run` to preview commands.

Usage example:

```bash
# Dry-run to see which experiments would be launched
python scripts/training.py --configs configs/attitudes --dry-run

# Launch a full batch using RLlib, grid size 3, learning rate 1e-4
nohup python scripts/training.py --configs configs/attitudes --rllib Yes --dilemma 0 --lr 0.0001 --grid-size 3 --seed 42 > out_batch_training.log 2>&1 &
```

Notes
- `scripts/training.py` expects `coin_game/trainer.py` to be callable from the repository root; adjust paths if you run it from a different working directory.
- Logs are written to `logs/training` by default; use `--log-dir` to change it.

Using `scripts/training.py` (detailed)
------------------------------------

What it does
- Scans the `--configs` directory for `*.txt` attitude files.
- For each file it launches a training process by calling `python ../coin_game/trainer.py <attitude_file> <rllib> <dilemma> <lr> <grid_size> <cluster> <seed>`.
- Captures stdout/stderr for each run into a log file under `--log-dir`.

Important flags
- `--configs`: directory containing attitude `*.txt` files to iterate over (default `../configs/attitudes`).
- `--rllib`: `Yes` or `No` (default `Yes`) — whether to run the RLlib training path.
- `--dilemma`: `0` or `1` (default `0`) — set which game variant to run (0 = no dilemma, 1 = prisoner's dilemma).
- `--lr`: learning rate float (default `0.001`).
- `--grid-size`: grid size int (default `3`).
- `--cluster`: optional cluster name from `['brigit', 'cuenca', 'local']` — script passes this to the trainer as a string argument.
- `--seed`: optional integer seed used to label logs (default `0`).
- `--log-dir`: where to write per-experiment logs (default `logs/training`).
- `--dry-run`: do not execute training commands; just list what would run.

Log naming
- Each experiment log is named like: `<attitude_name>_r<rllib>_d<dilemma>_lr<lr>_gs<grid_size>_s<seed>.log`.
  Example: `individualistic_cooperative_rYes_d0_lr0.001_gs3_s42.log`.

Running tips
- If you want to run experiments in background on a remote machine use `nohup` and redirect output. The script itself writes per-experiment logs, but `nohup` output will capture the launcher's stdout.
- Use `--dry-run` first to ensure the correct attitude files are detected and commands look correct.
- Ensure `coin_game/trainer.py` is callable from the path the launcher uses; the launcher shells out to `python ../coin_game/trainer.py` so run it from `scripts/` or adjust the path.

Example workflows

1) Dry-run to verify files and commands:

```bash
python scripts/training.py --configs configs/attitudes --dry-run
```

2) Launch experiments (background):

```bash
nohup python scripts/training.py --configs configs/attitudes --rllib Yes --dilemma 0 --lr 0.0001 --grid-size 3 --seed 42 --log-dir logs/batch_training > out_batch_training.log 2>&1 &
```

3) Run a targeted experiment manually (single attitude file):

```bash
python scripts/training.py --configs configs/attitudes --dry-run
# then run a single train command manually if you prefer fine control:
python coin_game/trainer.py configs/attitudes/individualistic_cooperative.txt Yes 0 0.0001 3 local 42
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
- `[[1, 0], [1, 0]]`: Individualistic agents (default)
- `[[0.70, 0.7071], [0.7071, 0.7071]]`: Cooperative agents
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
CoopCoins/
├── README.md                 # Main project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── configs/                  # Configuration files (attitudes, training presets)
│   └── attitudes/
├── coin_game/                # Main package: env, trainer, launchers, visualizers
│   ├── launch_training.py
│   ├── launch_specialized_training.py
│   ├── comparing_policies.ipynb
│   ├── trainer.py
│   ├── analysis.ipynb
│   ├── visualization.ipynb
│   └── visualize_rllib_models.py
├── JaxMARL/                  # Local copy of JaxMARL used by the project
├── scripts/                  # Utility scripts: generate, batch-run, visualize
│   ├── generate_attitudes.py
│   ├── training.py
│   └── example_visualization.py
├── docs/                     # Documentation (md files)
├── examples/                 # Example scripts and notebooks
└── tests/                    # Unit / integration tests
```

## Research Applications

CoopCoins is particularly useful for studying:

- **Social Dilemmas**: How agents learn to cooperate or defect
- **Opponent Shaping**: Learning to influence other agents' behavior
- **Emergent Cooperation**: Spontaneous development of cooperative strategies
- **Multi-Agent Learning**: Coordination and competition dynamics

## Citation

If you use CoopCoins in your research, please cite:

```bibtex
@misc{coopcoins2024,
  title={CoopCoins: A Multi-Agent Environment for Studying Collective Behavior},
  author={Samuel Lozano},
  year={2024},
  url={https://github.com/samuellozanoiglesias/CoopCoins}
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
