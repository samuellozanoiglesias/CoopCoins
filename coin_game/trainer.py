import jax
import os
import pickle
import sys
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.coin_game.make_train import make_train
from jaxmarl.environments.coin_game.make_train_RLLIB import make_train_RLLIB

input_path = sys.argv[1]
RLLIB = True if str(sys.argv[2]).lower() == 'yes' else False
DILEMMA = True if str(sys.argv[3]).lower() == 'yes' else False
LR = float(sys.argv[4])
GRID_SIZE = int(sys.argv[5])
CLUSTER = str(sys.argv[6]).lower() if len(sys.argv) > 6 else None
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else None

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    alpha_2, beta_2 = map(float, lines[1].strip().split())

REWARD_COEF = [[alpha_1, beta_1], [alpha_2, beta_2]]

if CLUSTER == "brigit":
    cluster_path = '/mnt/lustre/home/samuloza'
elif CLUSTER == "cuenca":
    cluster_path = '.'
elif CLUSTER == "local":
    cluster_path = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    cluster_path = ''

# Hyperparameters
NUM_ENVS = 1
NUM_INNER_STEPS = 150
NUM_UPDATES_PER_EPOCH = 10
NUM_EPOCHS = 5000
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 50
SAVE_EVERY_N_EPOCHS = 500

if RLLIB:
    train_dir = f'{cluster_path}/data/samuel_lozano/CoopCoins/RLLIB'
    trainer = make_train_RLLIB
else:
    train_dir = f'{cluster_path}/data/samuel_lozano/CoopCoins/JAXMARL'
    trainer = make_train

if DILEMMA:
    PAYOFF_MATRIX = [[1, 2, -3], [1, 2, -3]]
    save_dir = f'{train_dir}/Prisioner_dilemma'
else:
    PAYOFF_MATRIX = [[1, 1, -2], [1, 1, -2]]
    save_dir = f'{train_dir}/No_dilemma'

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
    # Training specific parameters
    "NUM_UPDATES": 4,  # Number of updates of the policy
    "GAMMA": 0.9,  # Slightly reduced for more immediate rewards
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.01,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5,  # Value function coefficient
    # JaxMARL specific parameters
    "MAX_GRAD_NORM": 0.3,  # Gradient clipping
    "MINIBATCH_SIZE": NUM_INNER_STEPS // NUM_UPDATES_PER_EPOCH,
    # RLlib specific parameters
    "SEED": SEED,
}

params, current_date = trainer(config)

path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
os.makedirs(path, exist_ok=True)

if RLLIB:
    # Save the final model
    final_checkpoint = trainer.save(os.path.join(path, f"checkpoint_{NUM_EPOCHS}"))
    print(f"Final checkpoint (RLLIB) saved at {final_checkpoint}")

else:
    # Save parameters for JAXMARL
    final_checkpoint = os.path.join(path, f"checkpoint_{NUM_EPOCHS}.pkl")
    with open(final_checkpoint, "wb") as f:
        pickle.dump(params, f)
    print(f"Final checkpoint (JAXMARL) saved at {final_checkpoint}")