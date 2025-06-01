import jax
import sys
import jax.numpy as jnp
import optax
import numpy as np
from jaxmarl.environments.coin_game.make_train import make_train

# Leer archivo de entrada
input_path = sys.argv[1]
with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    alpha_2, beta_2 = map(float, lines[1].strip().split())

REWARD_COEF = [[alpha_1, beta_1], [alpha_2, beta_2]]

# Hiperparámetros
NUM_ENVS = 4
NUM_INNER_STEPS = 250
NUM_EPOCHS = 500
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 20
LR = 1e-3
PAYOFF_MATRIX = [[1, 1, -2], [1, 1, -2]]
GRID_SIZE = 3

save_dir = '/data/samuel_lozano/coin_game/No_dilemma'
training_type = 'AC Epoch'

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
    "TRAINING_TYPE": training_type
}

params = make_train(config)