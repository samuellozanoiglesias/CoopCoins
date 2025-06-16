import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any
from jaxmarl.environments.coin_game.jax_coin_game import CoinGame as JaxCoinGame

class CoinGameEnvRLLIB(gym.Env):
    """
    RLlib-compatible wrapper for the JAX Coin Game environment.
    """
    def __init__(
        self,
        num_inner_steps: int = 10,
        num_outer_steps: int = 10,
        cnn: bool = False,
        egocentric: bool = False,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        grid_size: int = 3,
        reward_coef=[[1,0],[1,0]]
    ):
        super().__init__()
        
        # Initialize the JAX environment
        self.jax_env = JaxCoinGame(
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            cnn=cnn,
            egocentric=egocentric,
            payoff_matrix=payoff_matrix,
            grid_size=grid_size,
            reward_coef=reward_coef
        )
        
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size * grid_size * 4,) if not cnn else (grid_size, grid_size, 4),
            dtype=np.uint8
        )
        
        # Initialize state
        self.state = None
        self.agents = self.jax_env.agents
        self.current_agent = 0  # Track which agent's turn it is
        
        # Initialize metrics
        self.episode_metrics = {
            agent: {
                "cumulated_pure_reward": 0.0,
                "cumulated_modified_reward": 0.0,
                "cumulated_action_stats": np.zeros(5, dtype=np.int32)
            }
            for agent in self.agents
        }
        
        # Initialize RNG key
        self.key = jax.random.PRNGKey(0)

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        
        self.key, subkey = jax.random.split(self.key)
        obs, self.state = self.jax_env.reset(subkey)
        
        # Reset episode metrics
        for agent in self.agents:
            self.episode_metrics[agent] = {
                "cumulated_pure_reward": 0.0,
                "cumulated_modified_reward": 0.0,
                "cumulated_action_stats": np.zeros(5, dtype=np.int32)
            }
        
        self.current_agent = 0
        return obs[self.agents[0]], {}  # Return first agent's observation

    def step(self, action):
        """Step the environment."""
        # Create action dictionary for both agents
        actions = {self.agents[0]: action, self.agents[1]: 0}  # Default action for second agent
        
        # Step the environment
        self.key, subkey = jax.random.split(self.key)
        obs, self.state, rewards, dones, infos = self.jax_env.step(subkey, self.state, actions)
        
        # Update episode metrics
        for agent in self.agents:
            self.episode_metrics[agent]["cumulated_pure_reward"] = float(infos[agent]["cumulated_pure_reward"])
            self.episode_metrics[agent]["cumulated_modified_reward"] = float(infos[agent]["cumulated_modified_reward"])
            self.episode_metrics[agent]["cumulated_action_stats"] = np.array(infos[agent]["cumulated_action_stats"])
        
        # Switch to next agent
        self.current_agent = (self.current_agent + 1) % len(self.agents)
        
        # Return observation for current agent
        current_agent = self.agents[self.current_agent]
        return obs[current_agent], rewards[current_agent], dones["__all__"], False, {
            "agent_id": current_agent,
            "cumulated_pure_reward": self.episode_metrics[current_agent]["cumulated_pure_reward"],
            "cumulated_modified_reward": self.episode_metrics[current_agent]["cumulated_modified_reward"],
            "cumulated_action_stats": self.episode_metrics[current_agent]["cumulated_action_stats"]
        }

    def get_episode_metrics(self):
        """Get the current episode metrics."""
        return {
            agent: {
                "cumulated_pure_reward": self.episode_metrics[agent]["cumulated_pure_reward"],
                "cumulated_modified_reward": self.episode_metrics[agent]["cumulated_modified_reward"],
                "cumulated_action_stats": self.episode_metrics[agent]["cumulated_action_stats"]
            }
            for agent in self.agents
        }