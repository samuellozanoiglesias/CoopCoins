#!/usr/bin/env python3
"""
Batch training launcher for CoopCoins experiments.

This script runs multiple training experiments with different configurations
and manages the experiment process.

Usage:
    python scripts/batch_training.py --configs configs/attitudes/ --dilemma 0 --lr 0.001 --grid-size 3
"""

import argparse
import subprocess
import os
import glob
from pathlib import Path
import time

def find_attitude_files(configs_dir):
    """Find all attitude configuration files in the directory."""
    pattern = os.path.join(configs_dir, "*.txt")
    return glob.glob(pattern)

def run_single_experiment(attitude_file, dilemma, lr, grid_size, log_dir):
    """Run a single training experiment."""
    attitude_name = Path(attitude_file).stem
    
    # Create log file
    log_file = os.path.join(log_dir, f"{attitude_name}_d{dilemma}_lr{lr}_gs{grid_size}.log")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running experiment: {attitude_name}")
    print(f"  Config: {attitude_file}")
    print(f"  Dilemma: {dilemma}")
    print(f"  Learning rate: {lr}")
    print(f"  Grid size: {grid_size}")
    print(f"  Log: {log_file}")
    
    # Run training command
    cmd = [
        "python", "coin_game/training.py",
        attitude_file,
        str(dilemma),
        str(lr),
        str(grid_size)
    ]
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )
        
        if result.returncode == 0:
            print(f"  ✓ Completed successfully")
        else:
            print(f"  ✗ Failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 1 hour")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="Batch training launcher")
    parser.add_argument(
        "--configs", 
        type=str, 
        default="configs/attitudes",
        help="Directory containing attitude configuration files"
    )
    parser.add_argument(
        "--dilemma", 
        type=int, 
        default=0,
        choices=[0, 1],
        help="Dilemma mode (0: no dilemma, 1: prisoner's dilemma)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=3,
        help="Grid size"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs/batch_training",
        help="Directory for log files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    args = parser.parse_args()
    
    print("=== CoopCoins Batch Training ===")
    print(f"Configs directory: {args.configs}")
    print(f"Dilemma mode: {args.dilemma}")
    print(f"Learning rate: {args.lr}")
    print(f"Grid size: {args.grid_size}")
    print(f"Log directory: {args.log_dir}")
    print()
    
    # Find attitude files
    attitude_files = find_attitude_files(args.configs)
    
    if not attitude_files:
        print(f"No attitude files found in {args.configs}")
        return 1
    
    print(f"Found {len(attitude_files)} attitude configurations:")
    for f in attitude_files:
        print(f"  {Path(f).name}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Would run the following experiments:")
        for attitude_file in attitude_files:
            attitude_name = Path(attitude_file).stem
            print(f"  {attitude_name}: {attitude_file}")
        return 0
    
    # Run experiments
    start_time = time.time()
    
    for i, attitude_file in enumerate(attitude_files, 1):
        print(f"Experiment {i}/{len(attitude_files)}")
        run_single_experiment(
            attitude_file, 
            args.dilemma, 
            args.lr, 
            args.grid_size, 
            args.log_dir
        )
    
    total_time = time.time() - start_time
    print(f"Batch training completed in {total_time:.1f} seconds")
    print(f"Results saved to: {args.log_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 