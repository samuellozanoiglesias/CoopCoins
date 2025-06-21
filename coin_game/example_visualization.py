#!/usr/bin/env python3
"""
Example script showing how to visualize trained RLlib Coin Game models.

This script demonstrates how to:
1. Load a trained RLlib checkpoint
2. Visualize episodes using the trained model
3. Generate GIF animations of the gameplay using the original render method

Usage:
    python example_visualization.py /path/to/checkpoint [--episodes 3] [--output-dir visualizations]
"""

import os
import sys
import argparse

# Add the JaxMARL path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'JaxMARL'))

def main():
    parser = argparse.ArgumentParser(description="Example Coin Game RLlib visualization")
    parser.add_argument("checkpoint_path", help="Path to the RLlib checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Import the visualization functions
    from visualize_rllib_models import (
        load_rllib_checkpoint, 
        load_config_from_file, 
        visualize_episode
    )
    
    print("=== Coin Game RLlib Model Visualization Example ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Find config file
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.txt")
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = load_config_from_file(config_path)
    else:
        print("No config file found, using defaults")
        config = {
            "NUM_INNER_STEPS": 10,
            "NUM_EPOCHS": 10,
            "PAYOFF_MATRIX": [[1, 1, -2], [1, 1, -2]],
            "GRID_SIZE": 3,
            "REWARD_COEF": [[1, 0], [1, 0]]
        }
    
    # Load the trained model
    print("Loading trained model...")
    trainer = load_rllib_checkpoint(args.checkpoint_path)
    print("Model loaded successfully!")
    
    # Visualize episodes
    print(f"\nVisualizing {args.episodes} episodes...")
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1} ---")
        visualize_episode(
            trainer, 
            config, 
            num_episodes=1,
            save_gif=True,
            output_dir=args.output_dir
        )
    
    print(f"\nVisualization complete! Check the '{args.output_dir}' directory for GIF files.")

if __name__ == "__main__":
    main() 