import os
import pickle
import csv
from datetime import datetime

def create_training_directory(save_dir):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(save_dir, f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    return path, current_date

def save_config(config, path):
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

def save_params(params, path, epoch):
    with open(os.path.join(path, f"params_epoch_{epoch}.pkl"), "wb") as f:
        pickle.dump(params, f)

def log_training_stats(csv_path, write_header, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)