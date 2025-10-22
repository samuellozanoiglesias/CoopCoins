#!/usr/bin/env python3
"""
Generate attitude configuration files for systematic parameter sweeps.

This script creates input files for different cooperative attitudes
by varying the reward coefficients in a systematic way.

Usage:
    nohup python generate_attitudes.py [--angles 0,45,90,135,180,225,270,315] [--output-dir configs/attitudes] > out_generate_attitudes.log 2>&1 &

Example:
    nohup python generate_attitudes.py --angles 0,45,315 > out_generate_attitudes.log 2>&1 &
    nohup python generate_attitudes.py --predefined individualistic,cooperative,competitive > out_generate_attitudes.log 2>&1 &
"""

import argparse
import numpy as np
import os
from pathlib import Path

def generate_attitude_from_angles(angle1):
    """Generate single-agent reward coefficients from an angle in degrees.

    The function accepts an optional second argument for backwards compatibility
    but only uses the first angle to produce a single-agent [alpha, beta].
    """
    rad1 = np.radians(angle1)
    alpha1, beta1 = np.cos(rad1), np.sin(rad1)
    return [alpha1, beta1]

def generate_systematic_attitudes(angles):
    """Generate all combinations of attitudes from angle lists."""
    attitudes = {}
    # First generate single-agent attitudes for each angle
    single_agent = {}
    for angle in angles:
        name = f"angle_{int(angle)}"
        single_agent[name] = generate_attitude_from_angles(angle)

    # Now combine all pairs (including same-with-same) but keep unique ordering
    for name1, coef1 in single_agent.items():
        for name2, coef2 in single_agent.items():
            # Create a deterministic ordering for file names to avoid duplicates
            combo_name = f"{name1}_{name2}"
            attitudes[combo_name] = [coef1, coef2]
    
    return attitudes

def save_attitude_file(attitude_name, reward_coef, output_dir):
    """Save attitude configuration to file."""
    output_path = Path(output_dir) / f"{attitude_name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # reward_coef is [[a1, b1], [a2, b2]]
        f.write(f"{reward_coef[0][0]:.4f} {reward_coef[0][1]:.4f}\n")
        f.write(f"{reward_coef[1][0]:.4f} {reward_coef[1][1]:.4f}\n")
    
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
        default="../configs/attitudes",
        help="Output directory for attitude files"
    )
    parser.add_argument(
        "--predefined",
        type=str,
        default="",
        help="Comma-separated list of predefined attitudes to include (e.g. individualistic,cooperative). If omitted or empty, no predefined attitudes are added."
    )
    
    args = parser.parse_args()
    
    # Parse angles
    angles = [int(x.strip()) for x in args.angles.split(',')]
    
    print("=== Generating Attitude Configurations ===")
    print(f"Angles: {angles}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # If the user requested predefined attitudes, only generate combinations of those names
    attitudes = {}
    if args.predefined:
        # Predefined single-agent attitudes (master list)
        predefined_single = {
            'individualistic': [1.0, 0.0],
            'cooperative': [0.7071, 0.7071],
            'altruistic': [0.0, 1.0],
            'sacrificial': [-0.7071, 0.7071],
            'martyrial': [-1.0, 0.0],
            'destructive': [-0.7071, -0.7071],
            'spiteful': [0.0, -1.0],
            'competitive': [0.7071, -0.7071],
        }

        # Parse requested names (allow user to pass comma-separated list)
        requested = [s.strip() for s in args.predefined.split(',') if s.strip()]

        # If the user wants all predefined, they can use the keyword 'all'
        if 'all' in [r.lower() for r in requested]:
            selected = dict(predefined_single)
        else:
            # Select only known predefined attitudes and warn about unknowns
            selected = {}
            unknown = []
            for name in requested:
                if name in predefined_single:
                    selected[name] = predefined_single[name]
                else:
                    unknown.append(name)
            if unknown:
                print(f"Warning: unknown predefined attitudes ignored: {unknown}")

        # Combine all selected predefined single-agent attitudes pairwise (with themselves)
        for name1, coef1 in selected.items():
            for name2, coef2 in selected.items():
                combo_name = f"{name1}_{name2}"
                attitudes[combo_name] = [coef1, coef2]
    else:
        # No predefined requested: generate all angle-based combinations
        attitudes = generate_systematic_attitudes(angles)
    
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
    
    return 0

if __name__ == "__main__":
    exit(main()) 