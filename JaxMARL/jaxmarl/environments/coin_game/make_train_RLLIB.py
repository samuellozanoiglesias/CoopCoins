import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.registry import register_env
from datetime import datetime
import numpy as np
from gymnasium.spaces import Box, Discrete
from jaxmarl.environments.coin_game.coin_game_rllib_env import CoinGameRLLibEnv

def env_creator(config):
    env_idx = config.get("worker_index", 0)
    return CoinGameRLLibEnv(
        num_inner_steps=config["NUM_INNER_STEPS"],
        num_outer_steps=config["NUM_EPOCHS"],
        cnn=False,
        egocentric=False,
        payoff_matrix=config["PAYOFF_MATRIX"],
        grid_size=config["GRID_SIZE"],
        reward_coef=config["REWARD_COEF"],
        path=config["PATH"],
        env_idx=env_idx
    )

def make_train_RLLIB(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("coin_game_env_RLLIB", env_creator)

    # Create a temporary environment to get observation and action spaces
    temp_env = env_creator(config)

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
            vf_loss_coeff=config["VF_COEF"],
            num_epochs=config["NUM_UPDATES"],
            model={
                "fcnet_hiddens": [64, 64, 16],
                "fcnet_activation": "tanh",
                "use_lstm": False,
                "use_attention": False,
            }
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .debugging(log_level="INFO")
        .resources(num_gpus=1, num_gpus_per_worker=1)  # Allocate GPU for main trainer and workers
        .evaluation(
            evaluation_interval=config["SHOW_EVERY_N_EPOCHS"],
            evaluation_duration=10,
        )
        .multi_agent(
            policies={
                "agent_0": (None, temp_env.observation_space("agent_0"), temp_env.action_space("agent_0"), {}),
                "agent_1": (None, temp_env.observation_space("agent_1"), temp_env.action_space("agent_1"), {})
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            policies_to_train=["agent_0", "agent_1"]
        )
    )

    # Create the trainer
    trainer = ppo_config.build_algo()

    # Get the environment to access agent information
    env = env_creator(config)
    agents = env.agents

    # Training loop
    for epoch in range(config["NUM_EPOCHS"]):
        result = trainer.train()
        
        # Log metrics
        if epoch % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        # Save checkpoint
        if epoch % config["SAVE_EVERY_N_EPOCHS"] == 0:
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{epoch}"))
            print(f"Checkpoint saved at {checkpoint_path}")

    # Clean up
    trainer.stop()
    ray.shutdown()

    return trainer.get_policy(), current_date