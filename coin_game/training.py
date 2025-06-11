import jax
import os
import pickle
import sys
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.coin_game.make_train import make_train

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

brigit = '/mnt/lustre/home/samuloza'

# Hiperparámetros
NUM_ENVS = 1
NUM_INNER_STEPS = 50
NUM_UPDATES_PER_EPOCH = 5
NUM_EPOCHS = 3000
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 500
#LR = 1e-4
#GRID_SIZE = 3

if DILEMMA:
    PAYOFF_MATRIX = [[1, 2, -3], [1, 2, -3]]
    save_dir = f'{brigit}/data/samuel_lozano/coin_game/Prisioner_dilemma'
else:
    PAYOFF_MATRIX = [[1, 1, -2], [1, 1, -2]]
    save_dir = f'{brigit}/data/samuel_lozano/coin_game/No_dilemma'

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
    "MINIBATCH_SIZE": NUM_INNER_STEPS // NUM_UPDATES_PER_EPOCH,
    "MINIBATCH_EPOCHS": 1,
    "GAMMA": 0.995,
    "DEVICE": jax.devices()
}


params, current_date = make_train(config)

path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
with open(os.path.join(path, f"params_epoch_{NUM_EPOCHS}.pkl"), "wb") as f:
    pickle.dump(params, f)