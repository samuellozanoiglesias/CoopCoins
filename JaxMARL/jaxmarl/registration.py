from .environments import (
    CoinGame,
    CoinGameRLLibEnv,
)



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    
    # 1. Coin Game
    elif env_id == "coin_game":
        env = CoinGame(**env_kwargs)
    
    # 2. Coin Game RLLIB
    elif env_id == "coin_game_rllib_env":
        env = CoinGameRLLibEnv(**env_kwargs)

    return env

registered_envs = [
    "coin_game",
    "coin_game_rllib_env",
]
