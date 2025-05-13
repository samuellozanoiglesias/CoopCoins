#!/bin/bash

# Verificar que se proporcionó un nombre de job
if [ -z "$1" ]; then
    echo "Uso: $0 <nombre_del_job> [type_agents] [cluster]"
    exit 1
fi

# Parámetros
layout="$1"
type_agents=${2:-"Asymmetric_Agents"}
cluster=${3:-"cuenca"}

# Archivos de salida
output_file="results-GPU-${cluster}-${type_agents}-${layout}.txt"
error_file="error-results-GPU-${cluster}-${type_agents}-${layout}.txt"

# Activar el entorno Conda
conda activate JaxMARL

# Ejecutar el script de Python
python3 /home/samuel_lozano/hfsp_collective_learning/overcooked_human/GPU-Training-${type_agents}.py < layout-${cluster}-${layout}.txt > "$output_file" 2> "$error_file"