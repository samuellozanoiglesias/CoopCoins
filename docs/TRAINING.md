# Training Guide

This guide explains how to train agents in the CoopCoins environment, including configuration, running experiments, and analyzing results.

## Quick Start

### 1. Basic Training

The simplest way to start training:

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training example
python examples/basic_training.py
```

### 2. Using Pre-configured Attitudes

Train with different cooperative attitudes:

```bash
# Selfish agents
python coin_game/training.py configs/attitudes/selfish.txt 0 0.001 3

# Cooperative agents
python coin_game/training.py configs/attitudes/cooperative.txt 0 0.001 3

# Altruistic agents
python coin_game/training.py configs/attitudes/altruistic.txt 0 0.001 3
```

## Configuration

### Training Parameters

The main training parameters are:

- **NUM_ENVS**: Number of parallel environments (default: 4)
- **NUM_INNER_STEPS**: Steps per episode (default: 300)
- **NUM_EPOCHS**: Total training epochs (default: 100000)
- **LR**: Learning rate (default: 0.001)
- **GRID_SIZE**: Size of the grid (default: 3)

### Cooperative Attitudes

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

### Environment Variants

#### Standard Coin Game
- Full social dilemma implementation
- Complex reward structure
- Suitable for detailed research

#### RLlib Integration
- Compatible with Ray RLlib
- Supports advanced MARL algorithms
- Good for production training

## Training Scripts

### 1. Basic Training Script

```python
from coin_game.training import make_train

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

### 2. Command Line Training

```bash
python coin_game/training.py <attitude_file> <dilemma> <lr> <grid_size>
```

Parameters:
- `attitude_file`: Path to attitude configuration file
- `dilemma`: 0 for no dilemma, 1 for prisoner's dilemma
- `lr`: Learning rate
- `grid_size`: Grid size

### 3. Batch Training

Run multiple experiments automatically:

```bash
# Generate attitude configurations
python scripts/generate_attitudes.py --predefined

# Run batch training
python scripts/batch_training.py --configs configs/attitudes/ --dilemma 0 --lr 0.001
```

## Hyperparameter Tuning

### Learning Rate

Start with 0.001 and adjust based on:
- **Too slow convergence**: Increase to 0.01
- **Unstable training**: Decrease to 0.0001

### Environment Parameters

#### Grid Size
- **3x3**: Standard, good for most experiments
- **5x5**: More complex, longer training required
- **7x7**: Very complex, requires careful tuning

#### Episode Length
- **Short episodes (10-50 steps)**: Fast training, good for development
- **Medium episodes (100-300 steps)**: Balanced, good for most research
- **Long episodes (500+ steps)**: Detailed behavior analysis

### Network Architecture

#### CNN vs Flattened
- **CNN**: Better spatial understanding, slower training
- **Flattened**: Faster training, simpler implementation

#### Egocentric vs Absolute
- **Egocentric**: Each agent sees itself as "red", better for generalization
- **Absolute**: Fixed perspective, simpler but less generalizable

## Monitoring Training

### Progress Tracking

Training progress is automatically logged with metrics:
- Episode rewards
- Cooperation/defection rates
- Action statistics
- Loss values

### Checkpointing

Models are saved every `SAVE_EVERY_N_EPOCHS` epochs:
- Model parameters
- Training statistics
- Configuration files

### Visualization

Real-time training progress can be monitored:
```python
# Show progress every N epochs
config["SHOW_EVERY_N_EPOCHS"] = 1000
```

## Common Issues and Solutions

### 1. Slow Training

**Symptoms**: Training takes too long
**Solutions**:
- Reduce `NUM_INNER_STEPS`
- Increase `NUM_ENVS`
- Reduce grid size

### 2. Unstable Training

**Symptoms**: Rewards oscillate wildly
**Solutions**:
- Decrease learning rate
- Increase `MAX_GRAD_NORM`
- Adjust `ENT_COEF`
- Use gradient clipping

### 3. Poor Convergence

**Symptoms**: Agents don't learn cooperative behavior
**Solutions**:
- Check reward coefficients
- Increase training epochs
- Adjust entropy coefficient
- Try different network architectures

### 4. Memory Issues

**Symptoms**: Out of memory errors
**Solutions**:
- Reduce `NUM_ENVS`
- Reduce `MINIBATCH_SIZE`
- Use smaller grid size
- Enable gradient checkpointing

## Advanced Training

### Multi-GPU Training

```python
config["DEVICE"] = jax.devices()  # Use all available devices
```

### Custom Reward Functions

Modify the reward structure in the environment:
```python
# Custom payoff matrix
PAYOFF_MATRIX = [[1, 0.5, -1], [1, 0.5, -1]]
```

### Curriculum Learning

Start with simple scenarios and gradually increase difficulty:
1. Train on 3x3 grid
2. Transfer to 5x5 grid
3. Add complexity gradually

## Results Analysis

### Training Statistics

After training, analyze results:
```python
# Load training results
df = pd.read_csv("results/training_stats.csv")

# Plot training progress
import matplotlib.pyplot as plt
plt.plot(df['episode'], df['pure_reward_total'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
```

### Behavioral Analysis

Analyze agent behavior patterns:
- Cooperation rates
- Action distributions
- Reward sharing patterns
- Emergent strategies

### Model Comparison

Compare different configurations:
```python
# Compare selfish vs cooperative agents
selfish_results = load_results("results/selfish/")
cooperative_results = load_results("results/cooperative/")

compare_performance(selfish_results, cooperative_results)
```

## Best Practices

### 1. Experiment Design
- Start with simple configurations
- Use systematic parameter sweeps
- Document all experiments
- Use version control for code and configs

### 2. Training Process
- Monitor training progress regularly
- Save checkpoints frequently
- Use multiple random seeds
- Validate results with different metrics

### 3. Analysis
- Use multiple evaluation metrics
- Compare against baselines
- Consider statistical significance
- Document findings thoroughly

### 4. Reproducibility
- Set random seeds
- Save exact configurations
- Use deterministic algorithms where possible
- Document hardware and software versions

## Example Workflows

### Development Workflow
1. Use quick configuration for fast iteration
2. Test ideas on small grid sizes
3. Monitor basic metrics

### Research Workflow
1. Start with systematic parameter sweeps
2. Use standard configuration for main experiments
3. Run multiple seeds for statistical significance
4. Perform detailed behavioral analysis

### Production Workflow
1. Use RLlib integration for scalability
2. Implement proper logging and monitoring
3. Use distributed training for large-scale experiments
4. Implement automated evaluation pipelines 