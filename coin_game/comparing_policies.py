import os
import ast
import torch  # O cambia a ray si usas RLlib
import csv
import numpy as np

# Cambia a ray.rllib si usas RLlib
# from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.algorithm import Algorithm

# Ruta base donde buscar los entrenamientos
BASE_PATH = f"D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado/data/samuel_lozano/coin_game/RLLIB/No_dilemma"

# REWARD_COEFs a buscar
REWARD_COEFS = [
    [[0.707107, 0.707107], [1.0, 0.0]],
    [[0.707107, -0.707107], [1.0, 0.0]]
]

CHECKPOINT = 4500

# Función para leer el REWARD_COEF de un config.txt
def get_reward_coef(config_path):
    with open(config_path, 'r') as f:
        for line in f:
            if line.strip().startswith('REWARD_COEF'):
                # Extrae la parte después del igual
                coef_str = line.split(':', 1)[1].strip()
                try:
                    coef = ast.literal_eval(coef_str)
                    return coef
                except Exception as e:
                    print(f"Error parsing REWARD_COEF in {config_path}: {e}")
    return None

# Busca los directorios que contienen los REWARD_COEF deseados
def find_training_dirs():
    matches = {}
    for dir_name in os.listdir(BASE_PATH):
        dir_path = os.path.join(BASE_PATH, dir_name)
        if not os.path.isdir(dir_path):
            continue
        config_path = os.path.join(dir_path, 'config.txt')
        if not os.path.exists(config_path):
            continue
        coef = get_reward_coef(config_path)
        for target in REWARD_COEFS:
            if coef == target:
                matches[str(target)] = dir_path
    return matches

# Carga la política del segundo agente desde un checkpoint RLlib
def load_second_policy(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{CHECKPOINT}')
    # RLlib guarda un archivo extra con el nombre completo
    if not os.path.exists(checkpoint_path):
        # Busca el archivo real
        for f in os.listdir(checkpoint_dir):
            if f.startswith(f'checkpoint_{CHECKPOINT}'):
                checkpoint_path = os.path.join(checkpoint_dir, f)
                break
    # Carga el modelo con la API moderna
    algo = Algorithm.from_checkpoint(checkpoint_path)
    # Obtén la política del segundo agente ("agent_1")
    policy = algo.get_policy("agent_1")
    return policy

def generate_all_valid_observations(grid_size=3):
    """
    Genera todas las observaciones posibles (no-cnn) para CoinGame 3x3,
    filtrando combinaciones donde dos objetos ocupan la misma celda.
    Devuelve una lista de observaciones (vectores de 36 elementos).
    """
    positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    observations = []
    for red_pos in positions:
        for blue_pos in positions:
            if blue_pos == red_pos:
                continue
            for red_coin_pos in positions:
                if red_coin_pos == red_pos or red_coin_pos == blue_pos:
                    continue
                for blue_coin_pos in positions:
                    if blue_coin_pos in [red_pos, blue_pos, red_coin_pos]:
                        continue
                    # Construir la observación
                    obs = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
                    obs[red_pos[0], red_pos[1], 0] = 1
                    obs[blue_pos[0], blue_pos[1], 1] = 1
                    obs[red_coin_pos[0], red_coin_pos[1], 2] = 1
                    obs[blue_coin_pos[0], blue_coin_pos[1], 3] = 1
                    observations.append(obs.flatten())
    return observations

def extract_policy_to_csv(policy, output_csv):
    """
    Para cada observación válida, obtiene la acción de la política y la guarda en un CSV.
    """
    observations = generate_all_valid_observations()
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'obs_{i}' for i in range(36)] + ['action'])
        for obs in observations:
            # RLlib espera batch de observaciones
            action = policy.compute_single_action(obs)
            writer.writerow(list(obs) + [action])

if __name__ == "__main__":
    matches = find_training_dirs()
    for coef, dir_path in matches.items():
        checkpoint_dir = os.path.join(dir_path, f'checkpoint_{CHECKPOINT}')
        print(f"Cargando política del segundo agente para REWARD_COEF={coef} en {checkpoint_dir}")
        try:
            policy = load_second_policy(dir_path)
            output_csv = f"policy_obs_action_{coef.replace('[','').replace(']','').replace(',','_').replace(' ','')}.csv"
            extract_policy_to_csv(policy, output_csv)
            print(f"CSV guardado en {output_csv}")
        except Exception as e:
            print(f"Error cargando la política: {e}")
