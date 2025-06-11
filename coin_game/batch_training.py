import os
import numpy as np
import subprocess

angles_1 = np.arange(90, 360, 45)
dilemma = 0  # Set to 1 for dilemma, 0 for no dilemma
learning_rates = [1e-4, 1e-2]
grid_size = 3

os.makedirs("inputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for angle_1 in angles_1:
    angles_2 = np.arange(0, angle_1 + 1, 45)
    for angle_2 in angles_2:
        rad_1 = np.radians(angle_1)
        rad_2 = np.radians(angle_2)
        alpha_1, beta_1 = np.cos(rad_1), np.sin(rad_1)
        alpha_2, beta_2 = np.cos(rad_2), np.sin(rad_2)
        input_file = f"inputs/inputs_{int(angle_1)}_{int(angle_2)}.txt"
        with open(input_file, "w") as f:
            f.write(f"{alpha_1:.6f} {beta_1:.6f}\n")
            f.write(f"{alpha_2:.6f} {beta_2:.6f}\n")

        for lr in learning_rates:
            log_file = f"logs/out_{int(angle_1)}_{int(angle_2)}_lr{lr:.0e}.log"
            print(f"Running training for angles {angle_1}-{angle_2} with LR={lr}")
            subprocess.run(
                ["python", "training.py", input_file, str(dilemma), str(lr), str(grid_size)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT
            )
