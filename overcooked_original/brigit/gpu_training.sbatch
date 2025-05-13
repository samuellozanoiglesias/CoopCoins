#!/bin/bash
#SBATCH --job-name=custom_9
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1

# Extraer el valor de custom_N desde el nombre del job
layout=$(echo $SLURM_JOB_NAME | grep -o 'custom_[0-9]\+')
type_agents=${2:-"Asymmetric_Agents"}
cluster=${3:-"brigit"}

output_file="results-GPU-${cluster}-${type_agents}-${layout}.txt"
error_file="error-results-GPU-${cluster}-${type_agents}-${layout}.txt"

python3 /mnt/lustre/home/samuloza/hfsp_collective_learning/GPU-Training-${type_agents}.py < layout-${cluster}-${layout}.txt > "$output_file" 2> "$error_file"