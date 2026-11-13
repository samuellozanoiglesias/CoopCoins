#!/usr/bin/env python3
"""
Batch training launcher for CoopCoins experiments.

This script runs multiple training experiments with different configurations
and manages the experiment process.

Usage:
    nohup python training.py [--configs configs/attitudes/ --dilemma <value> --lr <value> --grid-size <value>] > out_batch_training.log 2>&1 &

Example:
    nohup python training.py --rllib Yes --dilemma 0 --lr 0.001 --grid-size 3 --seed 42 > out_batch_training.log 2>&1 &
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

def run_single_experiment(attitude_file, rllib, dilemma, lr, grid_size, cluster, seed, log_dir, job_timeout=3600):
    """Run a single training experiment."""
    attitude_name = Path(attitude_file).stem
    
    # Create log file
    log_file = os.path.join(log_dir, f"{attitude_name}_r{rllib}_d{dilemma}_lr{lr}_gs{grid_size}_s{seed}.log")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running experiment: {attitude_name}")
    print(f"  Config: {attitude_file}")
    print(f"  Dilemma: {dilemma}")
    print(f"  Learning rate: {lr}")
    print(f"  Grid size: {grid_size}")
    print(f"  Log: {log_file}")
    
    # Run training command
    cmd = [
        "python", "./coin_game/trainer.py",
        attitude_file,
        str(rllib),
        str(dilemma),
        str(lr),
        str(grid_size),
        str(cluster),
        str(seed) if seed is not None else ""
    ]

    # Determine timeout to use for this job. If <= 0, disable timeouts.
    timeout = None if job_timeout is None or int(job_timeout) <= 0 else int(job_timeout)

    try:
        # By default keep original blocking behavior for compatibility.
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout
            )

        if result.returncode == 0:
            print(f"  ✓ Completed successfully")
        else:
            print(f"  ✗ Failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after {job_timeout} seconds")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print()

def main():
    parser = argparse.ArgumentParser(description="Batch training launcher")
    parser.add_argument(
        "--configs", 
        type=str, 
        default="./configs/attitudes",
        help="Directory containing attitude configuration files"
    )
    parser.add_argument(
        "--rllib", 
        type=str, 
        default='Yes',
        choices=['Yes', 'No'],
        help="Use RLLIB (Yes/No)"
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
        "--cluster",
        type=str,
        default=None,
        choices=["brigit", "cuenca", "local"],
        help="Cluster to use (don't include argument for None)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (optional)"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs/training",
        help="Directory for log files"
    )
    parser.add_argument(
        "--maximum-parallel",
        type=int,
        default=10,
        help="Maximum number of training processes to run in parallel (1 = sequential)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be run without actually running"
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=-1,
        help="Per-job timeout in seconds (default 3600). Set to 0 or a negative value to disable timeouts."
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
    # If maximum_parallel == 1, use the existing blocking behavior for simplicity.
    max_parallel = max(1, int(args.maximum_parallel))

    if max_parallel == 1:
        for i, attitude_file in enumerate(attitude_files, 1):
            print(f"Experiment {i}/{len(attitude_files)}")
            run_single_experiment(
                attitude_file,
                args.rllib,
                args.dilemma,
                args.lr,
                args.grid_size,
                args.cluster,
                args.seed,
                args.log_dir,
                args.job_timeout,
            )
    else:
        # Launch up to `max_parallel` processes concurrently using Popen.
        print(f"Launching up to {max_parallel} parallel experiments")
        active = []  # list of dicts: {proc, fh, attitude_name, log_file, start_time, cmd, idx}
        # Use user-provided job timeout (seconds). If <= 0, disable timeout.
        JOB_TIMEOUT = int(args.job_timeout)

        def start_bg(att_file, idx):
            attitude_name = Path(att_file).stem
            log_file = os.path.join(args.log_dir, f"{attitude_name}_r{args.rllib}_d{args.dilemma}_lr{args.lr}_gs{args.grid_size}_s{args.seed}.log")
            os.makedirs(args.log_dir, exist_ok=True)
            cmd = [
                "python", "./coin_game/trainer.py",
                att_file,
                str(args.rllib),
                str(args.dilemma),
                str(args.lr),
                str(args.grid_size),
                str(args.cluster),
                str(args.seed) if args.seed is not None else ""
            ]
            fh = open(log_file, 'w')
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
            return {"proc": proc, "fh": fh, "attitude_name": attitude_name, "log_file": log_file, "start_time": time.time(), "cmd": cmd, "idx": idx}

        total = len(attitude_files)
        launched = 0
        i = 0
        while i < total or active:
            # Launch new jobs while we have capacity
            while len(active) < max_parallel and i < total:
                attitude_file = attitude_files[i]
                launched += 1
                print(f"Starting experiment {launched}/{total}: {Path(attitude_file).stem}")
                job = start_bg(attitude_file, launched)
                active.append(job)
                i += 1
                # Wait a short time between starting jobs to avoid spikes
                time.sleep(5)

            # Poll active jobs
            still_active = []
            for job in active:
                proc = job["proc"]
                ret = proc.poll()
                elapsed = time.time() - job["start_time"]
                if ret is not None:
                    # Finished
                    job["fh"].close()
                    if ret == 0:
                        print(f"  [{job['attitude_name']}] ✓ Completed successfully (log: {job['log_file']})")
                    else:
                        print(f"  [{job['attitude_name']}] ✗ Failed with return code {ret} (log: {job['log_file']})")
                elif JOB_TIMEOUT > 0 and elapsed > JOB_TIMEOUT:
                    # Timeout
                    proc.kill()
                    job["fh"].close()
                    print(f"  [{job['attitude_name']}] ✗ Timed out after {JOB_TIMEOUT} seconds (killed). Log: {job['log_file']}")
                else:
                    still_active.append(job)

            active = still_active

            # Small sleep to avoid busy loop
            if active:
                time.sleep(1)
    
    total_time = time.time() - start_time
    print(f"Batch training completed in {total_time:.1f} seconds")
    print(f"Results saved to: {args.log_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 