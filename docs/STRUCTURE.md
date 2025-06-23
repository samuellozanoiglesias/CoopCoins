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

- **`attitude_experiments.py`** - Cooperative attitude experiments
  - Systematic testing of different reward coefficients
  - Comparison between selfish, cooperative, and altruistic agents

### Analysis Examples

- **`training_analysis.ipynb`** - Comprehensive training analysis
  - Performance comparison across configurations
  - Behavioral pattern analysis
  - Statistical summaries

- **`visualization_demo.ipynb`** - Visualization examples
  - Episode replay generation
  - Agent behavior analysis
  - Training progress visualization

## Configuration (`configs/`)

### Reward Configurations

- **`attitudes/`** - Pre-configured cooperative attitudes
  - `selfish.txt` - [1.0, 0.0] for both agents
  - `cooperative.txt` - [0.7, 0.3] for both agents
  - `altruistic.txt` - [0.5, 0.5] for both agents
  - `mixed.txt` - Different attitudes for each agent

### Training Configurations

- **`training_configs/`** - Training parameter sets
  - `quick.yaml` - Fast training for development
  - `standard.yaml` - Standard training parameters
  - `comprehensive.yaml` - Full-scale experiments

## Scripts (`scripts/`)

### Utility Scripts

- **`generate_attitudes.py`** - Generate attitude configuration files
  - Creates input files for different cooperative attitudes
  - Supports systematic parameter sweeps

- **`batch_training.py`** - Batch training launcher
  - Runs multiple experiments with different configurations
  - Automated experiment management

- **`analyze_results.py`** - Results analysis script
  - Aggregates training results
  - Generates comparison plots
  - Creates summary reports

## Documentation (`docs/`)

- **`STRUCTURE.md`** - This file, explaining repository organization
- **`API.md`** - API documentation for the environment
- **`TRAINING.md`** - Detailed training guide
- **`ANALYSIS.md`** - Analysis and visualization guide
- **`EXAMPLES.md`** - Example usage and tutorials

## File Naming Conventions

### Training Files
- `training_*.py` - Training scripts
- `train_*.py` - Training utilities
- `*_trainer.py` - Trainer classes

### Environment Files
- `*_env.py` - Environment implementations
- `*_wrapper.py` - Environment wrappers
- `*_interface.py` - Environment interfaces

### Configuration Files
- `*.yaml` - YAML configuration files
- `*.json` - JSON configuration files
- `*.txt` - Simple text configurations

### Analysis Files
- `*_analysis.py` - Analysis scripts
- `*_metrics.py` - Metrics calculation
- `*_viz.py` - Visualization utilities

## Key Configuration Parameters

### Environment Parameters
- `GRID_SIZE`: Size of the grid (default: 3)
- `NUM_INNER_STEPS`: Steps per episode (default: 10)
- `REWARD_COEF`: Cooperative attitude coefficients
- `PAYOFF_MATRIX`: Reward structure matrix

### Training Parameters
- `NUM_ENVS`: Number of parallel environments
- `NUM_EPOCHS`: Total training epochs
- `LR`: Learning rate
- `BATCH_SIZE`: Training batch size

### Analysis Parameters
- `METRICS_WINDOW`: Window for calculating metrics
- `SAVE_INTERVAL`: How often to save checkpoints
- `LOG_LEVEL`: Logging verbosity

## Usage Patterns

### Quick Start
1. Use `examples/basic_training.py` for simple experiments
2. Modify `configs/training_configs/quick.yaml` for parameters
3. Run analysis with `examples/training_analysis.ipynb`

### Systematic Experiments
1. Generate configurations with `scripts/generate_attitudes.py`
2. Run batch training with `scripts/batch_training.py`
3. Analyze results with `scripts/analyze_results.py`

### Custom Research
1. Extend environment in `src/environments/`
2. Add new metrics in `src/analysis/metrics.py`
3. Create custom visualizations in `src/analysis/visualization.py` 