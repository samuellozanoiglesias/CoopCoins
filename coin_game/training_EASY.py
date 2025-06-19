import jax
import os
import pickle
import sys
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.coin_game.make_train_EASY import make_train_EASY   

# Leer archivo de entrada
input_path = sys.argv[1]
DILEMMA = int(sys.argv[2])
LR = float(sys.argv[3])
GRID_SIZE = int(sys.argv[4])

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    alpha_2, beta_2 = map(float, lines[1].strip().split())

REWARD_COEF = [[alpha_1, beta_1], [alpha_2, beta_2]]

#local = '/mnt/lustre/home/samuloza'
local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

# Hiperparámetros
NUM_ENVS = 4
NUM_INNER_STEPS = 300
NUM_UPDATES_PER_EPOCH = 150
NUM_EPOCHS = 100000
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 10000
SAVE_EVERY_N_EPOCHS = 10000

if DILEMMA:
    PAYOFF_MATRIX = [[1, 0, 0], [1, 0, 0]]
    save_dir = f'{local}/data/samuel_lozano/coin_game/Prisioner_dilemma'
else:
    PAYOFF_MATRIX = [[1, 0, 0], [1, 0, 0]]
    save_dir = f'{local}/data/samuel_lozano/coin_game/No_dilemma'

os.makedirs(save_dir, exist_ok=True)

config = {
    "NUM_ENVS": NUM_ENVS,
    "NUM_INNER_STEPS": NUM_INNER_STEPS,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "GRID_SIZE": GRID_SIZE,
    "REWARD_COEF": REWARD_COEF,
    "SAVE_DIR": save_dir,
    "GAMMA": 0.9,  # Slightly reduced for more immediate rewards
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.15,  # Increased entropy coefficient for better exploration
    "CLIP_EPS": 0.1,  # PPO clip parameter
    "VF_COEF": 0.7,  # Value function coefficient
    "MAX_GRAD_NORM": 0.5,  # Gradient clipping
    "MINIBATCH_SIZE": NUM_INNER_STEPS // NUM_UPDATES_PER_EPOCH,
    "NUM_UPDATES_PER_MINIBATCH": 4,
    "DEVICE": jax.devices()
}

params, current_date = make_train(config)

path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
with open(os.path.join(path, f"params_epoch_{NUM_EPOCHS}.pkl"), "wb") as f:
    pickle.dump(params, f)