import os
import numpy as np
import subprocess

# Activar entorno de trabajo
os.chdir(r"/home/samuel_lozano/CoopCoins/coin_game")

# Parámetros fijos
dilemma = 0
grid_size = 3
lr = 3e-4
train_script = "./training_RLLIB.py"

seeds = range(1, 11)
configs = ["dest_2"]

os.makedirs("inputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for seed in seeds:
    for config in configs:
        if config == "ind_1":
            angle_1, angle_2 = 0, 0
        elif config == "coop_1":
            angle_1, angle_2 = 0, 45
        elif config == "coop_2":
            angle_1, angle_2 = 45, 0
        elif config == "alt_1":
            angle_1, angle_2 = 0, 90
        elif config == "alt_2":
            angle_1, angle_2 = 90, 0
        elif config == "sacr_1":
            angle_1, angle_2 = 0, 135
        elif config == "sacr_2":
            angle_1, angle_2 = 135, 0
        elif config == "mart_1":
            angle_1, angle_2 = 0, 180
        elif config == "mart_2":
            angle_1, angle_2 = 180, 0
        elif config == "dest_1":
            angle_1, angle_2 = 0, 225
        elif config == "dest_2":
            angle_1, angle_2 = 225, 0
        elif config == "comp_1":
            angle_1, angle_2 = 0, 315
        elif config == "comp_2":
            angle_1, angle_2 = 315, 0
        

        rad_1 = np.radians(angle_1)
        rad_2 = np.radians(angle_2)
        alpha_1, beta_1 = np.cos(rad_1), np.sin(rad_1)
        alpha_2, beta_2 = np.cos(rad_2), np.sin(rad_2)

        input_file = f"inputs/inputs_{config}_seed{seed}.txt"
        log_file = f"logs/out_{config}_seed{seed}.log"

        with open(input_file, "w") as f:
            f.write(f"{alpha_1:.6f} {beta_1:.6f}\n")
            f.write(f"{alpha_2:.6f} {beta_2:.6f}\n")

        print(f"Training {config} with seed {seed}")
        subprocess.run(
            ["python", train_script, input_file, str(dilemma), str(lr), str(grid_size), str(seed)],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
