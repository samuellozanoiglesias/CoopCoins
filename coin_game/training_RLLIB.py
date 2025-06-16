import jax
import os
import pickle
import sys
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.coin_game.make_train_RLLIB import make_train_RLLIB

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
NUM_EPOCHS = 3000
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 500

if DILEMMA:
    PAYOFF_MATRIX = [[1, 2, -3], [1, 2, -3]]
    save_dir = f'{brigit}/data/samuel_lozano/coin_game/Prisioner_dilemma'
else:
    PAYOFF_MATRIX = [[1, 1, -2], [1, 1, -2]]
    save_dir = f'{brigit}/data/samuel_lozano/coin_game/No_dilemma'

os.makedirs(save_dir, exist_ok=True)

# RLlib specific configuration
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
    # RLlib specific parameters
    "NUM_STEPS": NUM_INNER_STEPS,  # Steps per rollout
    "MINIBATCH_SIZE": 64,  # Size of minibatches for training
    "UPDATE_EPOCHS": 10,  # Number of epochs to update the policy
    "GAMMA": 0.995,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.01,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5,  # Value function coefficient
    "DEVICE": jax.devices()
}

# Run training
trainer, current_date = make_train_RLLIB(config)

# Save the final model
path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
os.makedirs(path, exist_ok=True)

# Save the final policy
final_checkpoint = trainer.save(os.path.join(path, "final_checkpoint"))
print(f"Final checkpoint saved at {final_checkpoint}")