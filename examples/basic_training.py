#!/usr/bin/env python3
"""
Basic training example for CoopCoins environment.

This script demonstrates how to:
1. Set up the Coin Game environment
2. Configure training parameters
3. Run a simple training session
4. Save the trained model

Usage:
    python examples/basic_training.py
"""

import sys
import os
import yaml
import pickle
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=== CoopCoins Basic Training Example ===")
    
    # Load configuration
    config_path = "../configs/training_configs/quick.yaml"
    config = load_config(config_path)
    
    print(f"Loaded configuration from: {config_path}")
    print(f"Training for {config['training']['num_epochs']} epochs")
    print(f"Grid size: {config['environment']['grid_size']}")
    print(f"Reward coefficients: {config['environment']['reward_coef']}")
    print()
    
    # Import training function (this would be from the reorganized src)
    try:
        from training.trainer import make_train
        print("Starting training...")
        
        # Convert config to the format expected by make_train
        training_config = {
            "NUM_ENVS": config['training']['num_envs'],
            "NUM_INNER_STEPS": config['training']['num_inner_steps'],
            "NUM_EPOCHS": config['training']['num_epochs'],
            "NUM_AGENTS": 2,
            "SHOW_EVERY_N_EPOCHS": config['training']['show_every_n_epochs'],
            "SAVE_EVERY_N_EPOCHS": config['training']['save_every_n_epochs'],
            "LR": config['hyperparameters']['learning_rate'],
            "PAYOFF_MATRIX": config['environment']['payoff_matrix'],
            "GRID_SIZE": config['environment']['grid_size'],
            "REWARD_COEF": config['environment']['reward_coef'],
            "SAVE_DIR": config['paths']['save_dir'],
            "GAMMA": config['hyperparameters']['gamma'],
            "GAE_LAMBDA": config['hyperparameters']['gae_lambda'],
            "ENT_COEF": config['hyperparameters']['entropy_coef'],
            "CLIP_EPS": config['hyperparameters']['clip_eps'],
            "VF_COEF": config['hyperparameters']['vf_coef'],
            "MAX_GRAD_NORM": config['hyperparameters']['max_grad_norm'],
            "MINIBATCH_SIZE": config['hyperparameters']['minibatch_size'],
            "NUM_UPDATES_PER_MINIBATCH": config['hyperparameters']['num_updates_per_minibatch']
        }
        
        # Run training
        params, current_date = make_train(training_config)
        
        # Save final model
        os.makedirs(config['paths']['save_dir'], exist_ok=True)
        model_path = os.path.join(
            config['paths']['save_dir'], 
            f"model_{current_date}.pkl"
        )
        
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        
        print(f"Training completed!")
        print(f"Model saved to: {model_path}")
        
    except ImportError as e:
        print(f"Error: Could not import training module: {e}")
        print("Please ensure the src directory is properly set up.")
        return 1
    
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 