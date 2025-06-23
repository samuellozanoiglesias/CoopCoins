#!/usr/bin/env python3
"""
Generate attitude configuration files for systematic parameter sweeps.

This script creates input files for different cooperative attitudes
by varying the reward coefficients in a systematic way.

Usage:
    python scripts/generate_attitudes.py [--angles 0,45,90,135,180,225,270,315] [--output-dir configs/attitudes]
"""

import argparse
import numpy as np
import os
from pathlib import Path

def generate_attitude_from_angles(angle1, angle2):
    """Generate reward coefficients from angles in degrees."""
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)
    alpha1, beta1 = np.cos(rad1), np.sin(rad1)
    alpha2, beta2 = np.cos(rad2), np.sin(rad2)
    return [[alpha1, beta1], [alpha2, beta2]]

def generate_systematic_attitudes(angles):
    """Generate all combinations of attitudes from angle lists."""
    attitudes = {}
    
    for angle1 in angles:
        for angle2 in angles:
            if angle2 <= angle1:  # Only generate unique combinations
                name = f"angle_{int(angle1)}_{int(angle2)}"
                reward_coef = generate_attitude_from_angles(angle1, angle2)
                attitudes[name] = reward_coef
    
    return attitudes

def save_attitude_file(attitude_name, reward_coef, output_dir):
    """Save attitude configuration to file."""
    output_path = Path(output_dir) / f"{attitude_name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"{reward_coef[0][0]:.6f} {reward_coef[0][1]:.6f}\n")
        f.write(f"{reward_coef[1][0]:.6f} {reward_coef[1][1]:.6f}\n")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate attitude configuration files")
    parser.add_argument(
        "--angles", 
        type=str, 
        default="0,45,90,135,180,225,270,315",
        help="Comma-separated list of angles in degrees"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="configs/attitudes",
        help="Output directory for attitude files"
    )
    parser.add_argument(
        "--predefined", 
        action="store_true",
        help="Also generate predefined attitude configurations"
    )
    
    args = parser.parse_args()
    
    # Parse angles
    angles = [int(x.strip()) for x in args.angles.split(',')]
    
    print("=== Generating Attitude Configurations ===")
    print(f"Angles: {angles}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Generate systematic attitudes
    attitudes = generate_systematic_attitudes(angles)
    
    # Add predefined attitudes if requested
    if args.predefined:
        predefined = {
            'selfish': [[1.0, 0.0], [1.0, 0.0]],
            'cooperative': [[0.707107, 0.707107], [0.707107, 0.707107]],
            'altruistic': [[0.5, 0.5], [0.5, 0.5]],
            'mixed_selfish_cooperative': [[1.0, 0.0], [0.707107, 0.707107]],
            'mixed_cooperative_altruistic': [[0.707107, 0.707107], [0.5, 0.5]]
        }
        attitudes.update(predefined)
    
    # Save attitude files
    saved_files = []
    for attitude_name, reward_coef in attitudes.items():
        filepath = save_attitude_file(attitude_name, reward_coef, args.output_dir)
        saved_files.append(filepath)
        print(f"Created: {filepath}")
        print(f"  Reward coefficients: {reward_coef}")
    
    print()
    print(f"Generated {len(saved_files)} attitude configuration files")
    print(f"Files saved to: {args.output_dir}")
    
    # Print usage example
    print()
    print("Usage example:")
    print("  python coin_game/training.py configs/attitudes/selfish.txt 0 0.001 3")
    print("  python coin_game/training.py configs/attitudes/cooperative.txt 0 0.001 3")
    
    return 0

if __name__ == "__main__":
    exit(main()) 