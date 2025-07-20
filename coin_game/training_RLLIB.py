import os
import sys
from jaxmarl.environments.coin_game.make_train_RLLIB import make_train_RLLIB

# Leer archivo de entrada
input_path = sys.argv[1]
DILEMMA = int(sys.argv[2])
LR = float(sys.argv[3])
GRID_SIZE = int(sys.argv[4])
SEED = int(sys.argv[5])

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = map(float, lines[0].strip().split())
    alpha_2, beta_2 = map(float, lines[1].strip().split())

REWARD_COEF = [[alpha_1, beta_1], [alpha_2, beta_2]]

#local = '/mnt/lustre/home/samuloza'
local = ''
#local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

# Hiperparámetros
NUM_ENVS = 1
NUM_INNER_STEPS = 150
NUM_EPOCHS = 5000
NUM_AGENTS = 2
SHOW_EVERY_N_EPOCHS = 1000
SAVE_EVERY_N_EPOCHS = 500

if DILEMMA:
    PAYOFF_MATRIX = [[1, 2, -3], [1, 2, -3]]
    save_dir = f'{local}/data/samuel_lozano/CoopCoins/RLLIB/Prisioner_dilemma'
else:
    PAYOFF_MATRIX = [[1, 1, -2], [1, 1, -2]]
    save_dir = f'{local}/data/samuel_lozano/CoopCoins/RLLIB/No_dilemma'

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
    "NUM_UPDATES": 4,  # Number of updates of the policy
    "GAMMA": 0.9,  # Discount factor
    "GAE_LAMBDA": 0.95,  # GAE-Lambda parameter
    "ENT_COEF": 0.05,  # Entropy coefficient
    "CLIP_EPS": 0.2,  # PPO clip parameter
    "VF_COEF": 0.5,  # Value function coefficient
    "SEED": SEED,
}


# Run training
trainer, current_date = make_train_RLLIB(config)

# Save the final model
path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
os.makedirs(path, exist_ok=True)

# Save the final policy
final_checkpoint = trainer.save(os.path.join(path, f"checkpoint_{NUM_EPOCHS}"))
print(f"Final checkpoint saved at {final_checkpoint}")