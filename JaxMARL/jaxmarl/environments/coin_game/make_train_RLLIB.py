import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from datetime import datetime
from jaxmarl import make
import csv

def env_creator(config):
    env = make("coin_game_env_RLLIB", 
        num_inner_steps=config["NUM_INNER_STEPS"],
        num_outer_steps=config["NUM_EPOCHS"],
        cnn=False,
        egocentric=False,
        payoff_matrix=config["PAYOFF_MATRIX"],
        grid_size=config["GRID_SIZE"],
        reward_coef=config["REWARD_COEF"]
    )
    return PettingZooEnv(env)

def make_train_RLLIB(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # Initialize CSV logging
    csv_path = os.path.join(path, "training_stats.csv")
    write_header = True

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("coin_game_env_RLLIB", env_creator)

    # Configure PPO
    ppo_config = (
        PPOConfig()
        .environment("coin_game_env_RLLIB", env_config=config)
        .env_runners(num_env_runners=config["NUM_ENVS"])
        .training(
            train_batch_size=config["NUM_ENVS"] * config["NUM_INNER_STEPS"],
            lr=config["LR"],
            gamma=config["GAMMA"],
            lambda_=config["GAE_LAMBDA"],
            entropy_coeff=config["ENT_COEF"],
            clip_param=config["CLIP_EPS"],
            vf_clip_param=config["VF_COEF"],
            num_epochs=config["NUM_UPDATES"],
        )
        .framework("torch")
        .debugging(log_level="INFO")
        .resources(num_gpus=0)  # Set to number of GPUs available
        .evaluation(
            evaluation_interval=config["SHOW_EVERY_N_EPOCHS"],
            evaluation_duration=10,
        )
    )

    # Create the trainer
    trainer = ppo_config.build_algo()

    # Get the environment to access agent information
    env = env_creator(config)
    agents = env.env.agents

    # Training loop
    for epoch in range(config["NUM_EPOCHS"]):
        result = trainer.train()
        
        # Log metrics
        if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")
            print(f"Episode reward mean: {result['episode_reward_mean']}")
            print(f"Episode length mean: {result['episode_len_mean']}")

        # Extract and log detailed metrics
        row = {
            "epoch": epoch,
        }

        # Get the latest episode info
        episode_infos = result.get("hist_stats", {}).get("episode", {})
        if episode_infos:
            for agent in agents:
                # Get agent-specific metrics from the environment
                agent_info = episode_infos.get(f"{agent}_info", {})
                if agent_info:
                    row.update({
                        f"reward_{agent}": float(agent_info.get("cumulated_modified_reward", 0)),
                        f"pure_reward_{agent}": float(agent_info.get("cumulated_pure_reward", 0)),
                    })
                    
                    a_stats = agent_info.get("cumulated_action_stats", [0, 0, 0, 0, 0])
                    row.update({
                        f"own_coin_collected_{agent}": int(a_stats[0]),
                        f"other_coin_collected_{agent}": int(a_stats[1]),
                        f"reject_own_coin_{agent}": int(a_stats[2]),
                        f"reject_other_coin_{agent}": int(a_stats[3]),
                        f"no_coin_visible_{agent}": int(a_stats[4]),
                    })

            # Write to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

        # Save checkpoint
        if epoch % config["SAVE_EVERY_N_EPOCHS"] == 0:
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{epoch}"))
            print(f"Checkpoint saved at {checkpoint_path}")

    # Clean up
    trainer.stop()
    ray.shutdown()

    return trainer.get_policy(), current_date