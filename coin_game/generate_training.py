import os
import numpy as np
import time

angles_1 = np.arange(90, 360, 45)

os.makedirs("inputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for angle_1 in angles_1:
    angles_2 = np.arange(0, angle_1+1, 45)
    for angle_2 in angles_2:
        rad_1 = np.radians(angle_1)
        rad_2 = np.radians(angle_2)
        alpha_1, beta_1 = np.cos(rad_1), np.sin(rad_1)
        alpha_2, beta_2 = np.cos(rad_2), np.sin(rad_2)

        filename = f"inputs/inputs_{int(angle_1)}_{int(angle_2)}.txt"
        with open(filename, "w") as f:
            f.write(f"{alpha_1:.6f} {beta_1:.6f}\n")
            f.write(f"{alpha_2:.6f} {beta_2:.6f}\n")

        nohup_command = f"nohup python training.py {filename} > logs/out_{int(angle_1)}_{int(angle_2)}.log 2>&1 &"
        os.system(nohup_command)
        time.sleep(3)
