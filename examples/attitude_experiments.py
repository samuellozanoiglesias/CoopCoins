#!/usr/bin/env python3
"""
Cooperative attitude experiments for CoopCoins.

This script demonstrates how to:
1. Run experiments with different cooperative attitudes
2. Compare selfish, cooperative, and altruistic agents
3. Generate systematic parameter sweeps
4. Analyze results across different configurations

Usage:
    python examples/attitude_experiments.py
"""

import sys
import os
import yaml
import numpy as np
import subprocess
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def generate_attitude_configs():
    """Generate different cooperative attitude configurations."""
    attitudes = {
        'selfish': [[1.0, 0.0], [1.0, 0.0]],
        'cooperative': [[0.7, 0.3], [0.7, 0.3]],
        'altruistic': [[0.5, 0.5], [0.5, 0.5]],
        'mixed_cooperative': [[0.8, 0.2], [0.6, 0.4]],
        'mixed_altruistic': [[0.9, 0.1], [0.3, 0.7]]
    }
    return attitudes

def create_experiment_config(attitude_name, reward_coef, base_config):
    """Create a configuration for a specific attitude experiment."""
    config = base_config.copy()
    config['environment']['reward_coef'] = reward_coef
    config['paths']['save_dir'] = f"./results/attitude_experiments/{attitude_name}"
    config['paths']['log_dir'] = f"./logs/attitude_experiments/{attitude_name}"
    return config

def save_config(config, filepath):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_experiment(config_path, attitude_name):
    """Run a single experiment."""
    print(f"Running experiment: {attitude_name}")
    
    # This would call the training script with the config
    # For now, we'll just print what would happen
    print(f"  Config: {config_path}")
    print(f"  Save dir: {config['paths']['save_dir']}")
    print(f"  Reward coefficients: {config['environment']['reward_coef']}")
    print()

def main():
    print("=== CoopCoins Attitude Experiments ===")
    
    # Load base configuration
    base_config_path = "../configs/training_configs/quick.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print(f"Base configuration: {base_config_path}")
    print()
    
    # Generate attitude configurations
    attitudes = generate_attitude_configs()
    
    # Create experiment configurations
    experiments_dir = Path("../configs/experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    print("Generated attitude configurations:")
    for attitude_name, reward_coef in attitudes.items():
        print(f"  {attitude_name}: {reward_coef}")
    print()
    
    # Create and save experiment configs
    for attitude_name, reward_coef in attitudes.items():
        config = create_experiment_config(attitude_name, reward_coef, base_config)
        config_path = experiments_dir / f"{attitude_name}.yaml"
        save_config(config, config_path)
        
        print(f"Created config: {config_path}")
    
    print()
    print("To run experiments:")
    print("1. Use the generated configs in configs/experiments/")
    print("2. Run training with each config")
    print("3. Compare results using analysis tools")
    print()
    
    # Example of how to run experiments (commented out for safety)
    """
    for attitude_name in attitudes.keys():
        config_path = experiments_dir / f"{attitude_name}.yaml"
        run_experiment(config_path, attitude_name)
    """
    
    print("Experiment configurations created successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 