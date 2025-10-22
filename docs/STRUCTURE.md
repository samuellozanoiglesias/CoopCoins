# Repository Structure

This document explains the organization and purpose of each file and directory in the CoopCoins repository.

## Root Directory

```
coopcoins/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore patterns
├── docs/                    # Documentation directory
├── src/                     # Source code directory
├── examples/                # Example scripts and notebooks
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
└── tests/                   # Test files
```

## Source Code (`src/`)

### Core Environment (`src/environments/`)

- **`coin_game.py`** - Standard Coin Game environment implementation
  - Full social dilemma with configurable reward coefficients
  - Supports both CNN and flattened observations
  - Includes egocentric and absolute positioning modes

- **`coin_game_rllib.py`** - RLlib-compatible environment wrapper
  - Integration with Ray RLlib framework
  - Compatible with RLlib's multi-agent training algorithms

### Training Modules (`src/training/`)

- **`trainer.py`** - Core training implementation
  - PPO-based multi-agent training
  - Configurable hyperparameters
  - Progress tracking and checkpointing

- **`config.py`** - Training configuration management
  - Default parameter sets
  - Configuration validation
  - Experiment setup utilities

### Analysis Tools (`src/analysis/`)

- **`metrics.py`** - Performance metrics calculation
  - Cooperation/defection rates
  - Reward analysis
  - Behavioral pattern detection

- **`visualization.py`** - Visualization utilities
  - Training progress plots
  - Agent behavior visualization
  - GIF generation for episodes

## Examples (`examples/`)

### Training Examples

- **`basic_training.py`** - Simple training example
  - Minimal setup for getting started
  - Basic configuration demonstration

# Repository structure (aligned with this repo)

This file describes the actual top-level layout of the CoopCoins repository and what each major folder contains. The repository uses `coin_game/` as the main package (not `src/`).

Top-level layout

```
CoopCoins/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/                # Attitude & training configuration files
│   └── attitudes/
├── coin_game/              # Main environment and training code (package)
├── JaxMARL/                # Local copy of JaxMARL used by the project
├── scripts/                # Utility scripts (generate, batch, visualize)
├── docs/                   # Documentation markdown files
├── examples/               # Example scripts / notebooks
└── tests/                  # Unit / integration tests
```

Key folders

- `coin_game/`: main package containing the environment implementations, training entrypoints (e.g., `trainer.py`, `launch_training.py`), and utilities. Use this package to run or import training and visualization functions.
- `configs/`: contains `attitudes/` (text files with reward coefficients) and `training_configs/` (yaml files for experiment presets).
- `scripts/`: helper scripts such as `generate_attitudes.py`, `training.py` (batch launcher), and `example_visualization.py`.
- `JaxMARL/`: local dependency shipped in-tree; install with `pip install -e ./JaxMARL` for development.
- `docs/`: user-facing documentation (this file, API references, training guides).

Scripts and utilities

- `scripts/generate_attitudes.py`: create attitude `.txt` files (angles or predefined names).
- `scripts/training.py`: batch launcher that iterates attitude files and calls `coin_game/trainer.py` for each.
- `scripts/example_visualization.py`: simple script to load an RLlib checkpoint and produce GIF visualizations.

Configuration files

- Attitude files (in `configs/attitudes/`) are text files with two lines: `alpha beta` for each agent.
- Training presets are YAML files in `configs/training_configs/` (e.g., `quick.yaml`, `standard.yaml`).

Usage examples

1. Generate attitudes:

```bash
python scripts/generate_attitudes.py --angles 0,45,90
```

2. Dry-run the batch launcher:

```bash
python scripts/training.py --configs configs/attitudes --dry-run
```

3. Visualize a checkpoint:

```bash
python scripts/example_visualization.py /path/to/checkpoint --episodes 3
```