# CoopCoins Documentation

Welcome to the CoopCoins documentation! This guide will help you navigate the documentation and get started with the project.

## Quick Navigation

### 🚀 Getting Started
- **[README.md](../README.md)** - Main project overview and quick start guide
- **[Installation Guide](#installation)** - How to set up CoopCoins
- **[Basic Examples](#basic-examples)** - Simple examples to get you started

### 📚 Core Documentation
- **[API Documentation](API.md)** - Complete API reference
- **[Training Guide](TRAINING.md)** - Comprehensive training documentation
- **[Repository Structure](STRUCTURE.md)** - File organization and purpose
- **[Migration Guide](MIGRATION.md)** - How to migrate to the new structure

### 🛠️ Development
- **[Configuration Guide](#configuration)** - How to configure experiments
- **[Analysis Guide](#analysis)** - How to analyze results
- **[Troubleshooting](#troubleshooting)** - Common issues and solutions

## Installation

### Prerequisites
- Python 3.8+
- JAX (with appropriate backend)
- Ray RLlib (optional, for advanced training)

### Quick Install
```bash
git clone https://github.com/yourusername/coopcoins.git
cd coopcoins
pip install -r requirements.txt
```

### Verify Installation
```bash
python examples/basic_training.py
```

## Basic Examples

### 1. Simple Training
```python
from examples.basic_training import main
main()
```

### 2. Attitude Experiments
```python
from examples.attitude_experiments import main
main()
```

### 3. Generate Configurations
```bash
python scripts/generate_attitudes.py --predefined
```

## Configuration

### Environment Configuration

The CoopCoins environment can be configured through several parameters:

#### Basic Parameters
- `grid_size`: Size of the grid (default: 3)
- `num_inner_steps`: Steps per episode (default: 10)
- `reward_coef`: Cooperative attitude coefficients

#### Advanced Parameters
- `payoff_matrix`: Reward structure matrix
- `cnn`: Whether to use CNN observations
- `egocentric`: Whether to use egocentric observations

### Training Configuration

Training parameters are specified in YAML files:

```yaml
training:
  num_envs: 4
  num_epochs: 100000
  learning_rate: 0.001

environment:
  grid_size: 3
  reward_coef: [[1.0, 0.0], [1.0, 0.0]]
```

### Attitude Configuration

Attitude files specify cooperative behavior:

```
1.000000 0.000000  # Agent 1: selfish
1.000000 0.000000  # Agent 2: selfish
```

## Analysis

### Training Analysis

Analyze training results using the provided notebooks:

```bash
jupyter notebook examples/training_analysis.ipynb
```

### Visualization

Generate visualizations of agent behavior:

```python
from src.analysis.visualization import visualize_episode
visualize_episode(trainer, config, num_episodes=3)
```

### Metrics

Key metrics tracked during training:
- Episode rewards
- Cooperation/defection rates
- Action distributions
- Behavioral patterns

## Project Structure

```
coopcoins/
├── README.md                 # Main documentation
├── requirements.txt          # Dependencies
├── src/                     # Source code
│   ├── environments/        # Environment implementations
│   ├── training/           # Training modules
│   └── analysis/           # Analysis tools
├── examples/               # Example scripts and notebooks
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── tests/                  # Test files
```

## Common Use Cases

### 1. Research Experiments

For systematic research:

1. **Generate configurations**:
   ```bash
   python scripts/generate_attitudes.py --angles 0,45,90,135,180,225,270,315
   ```

2. **Run batch experiments**:
   ```bash
   python scripts/batch_training.py --configs configs/attitudes/ --dilemma 0 --lr 0.001
   ```

3. **Analyze results**:
   ```bash
   jupyter notebook examples/training_analysis.ipynb
   ```

### 2. Development and Testing

For development work:

1. **Use quick configuration**:
   ```bash
   python examples/basic_training.py
   ```

2. **Test different attitudes**:
   ```bash
   python coin_game/training.py configs/attitudes/cooperative.txt 0 0.001 3
   ```

3. **Visualize behavior**:
   ```bash
   python examples/example_visualization.py /path/to/checkpoint
   ```

### 3. Production Training

For large-scale training:

1. **Use RLlib integration**:
   ```python
   from src.environments.coin_game_rllib import CoinGameRLLibEnv
   ```

2. **Configure distributed training**:
   ```yaml
   training:
     num_envs: 16
     distributed: true
   ```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when importing CoopCoins modules
**Solution**: Ensure the `src/` directory is in your Python path

#### 2. Training Issues
**Problem**: Training is slow or unstable
**Solution**: 
- Reduce `num_inner_steps`
- Adjust learning rate

#### 3. Memory Issues
**Problem**: Out of memory errors
**Solution**:
- Reduce `num_envs`
- Use smaller grid size
- Enable gradient checkpointing

#### 4. Configuration Issues
**Problem**: Configuration files not found
**Solution**: Check file paths and ensure configs are in the correct format

### Getting Help

1. **Check the documentation**: Start with the relevant guide above
2. **Look at examples**: Examine the example scripts for usage patterns
3. **Check issues**: Look for similar problems in the GitHub issues
4. **Create an issue**: If you can't find a solution, create a new issue

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions
- Write tests for new features

### Documentation

- Update relevant documentation when adding features
- Add examples for new functionality
- Keep the API documentation current

## Version History

### v1.0.0 (Current)
- Initial release with organized structure
- Comprehensive documentation
- Example scripts and utilities
- Multiple environment variants

### Planned Features
- Additional environment variants
- More analysis tools
- Web-based visualization interface
- Integration with more MARL frameworks

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- Based on the original Coin Game from Lerer & Peysakhovich (2017)
- Built on the JaxMARL framework
- Inspired by research on opponent shaping and social dilemmas

---

**Need help?** Check the [troubleshooting section](#troubleshooting) or create an issue on GitHub. 