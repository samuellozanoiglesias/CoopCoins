#!/bin/bash
#SBATCH --job-name=custom_9
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1

# Definir los valores de los hiperparámetros
VF_COEF_VALUES=(0.25 0.5 0.75)
CLIP_EPS_VALUES=(0.1 0.2 0.3)
ENT_COEF_VALUES=(0.05 0.1 0.25)

# Extraer el valor de custom_N desde el nombre del job
layout=$(echo $SLURM_JOB_NAME | grep -o 'custom_[0-9]\+')
type_agents=${2:-"Asymmetric_Agents"}
cluster=${3:-"brigit"}

for vf in "${VF_COEF_VALUES[@]}"; do
    for clip in "${CLIP_EPS_VALUES[@]}"; do
        for ent in "${ENT_COEF_VALUES[@]}"; do
            
            # Generar nombres únicos para los archivos de salida y error
            output_file="results-GPU-${cluster}-${type_agents}-${layout}-vf${vf}-clip${clip}-ent${ent}.txt"
            error_file="error-results-GPU-${cluster}-${type_agents}-${layout}-vf${vf}-clip${clip}-ent${ent}.txt"

            echo "Ejecutando experimento con VF_COEF=$vf, CLIP_EPS=$clip, ENT_COEF=$ent"

            # Ejecutar el script de entrenamiento con los hiperparámetros
            python3 /mnt/lustre/home/samuloza/hfsp_collective_learning/overcooked_original/GPU-Training-${type_agents}_hyperparameter_optimization.py \
                --vf_coef $vf --clip_eps $clip --ent_coef $ent < layout-${cluster}-${layout}.txt \
                > "$output_file" 2> "$error_file"

        done
    done
done