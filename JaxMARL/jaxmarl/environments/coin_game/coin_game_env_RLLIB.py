import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any
from jaxmarl.environments.coin_game.jax_coin_game import CoinGame as JaxCoinGame
from pettingzoo import ParallelEnv

class CoinGameEnvRLLIB(ParallelEnv):
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

        self.grid_size = grid_size
        self.agents = self.jax_env.agents
        self.possible_agents = self.agents.copy()
        
        # Set up action and observation spaces
        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.grid_size * self.grid_size * 4,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Initialize state
        self.state = None
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Initialize episode metrics
        self.episode_metrics = {}
        for agent in self.agents:
            self.episode_metrics[agent] = {
                "cumulated_pure_reward": 0.0,
                "cumulated_modified_reward": 0.0,
                "cumulated_action_stats": np.zeros(5, dtype=np.int32)
            }
        
        # Initialize RNG key
        self.key = jax.random.PRNGKey(0)

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.agents = self.possible_agents.copy()
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
        
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Convert observations to numpy arrays if they aren't already
        obs_dict = {}
        for agent in self.agents:
            if isinstance(obs[agent], jnp.ndarray):
                obs_dict[agent] = np.array(obs[agent])
            else:
                obs_dict[agent] = obs[agent]
        
        return obs_dict

    def step(self, actions):
        """Step the environment with simultaneous actions."""
        # Step the environment with all actions at once
        self.key, subkey = jax.random.split(self.key)
        obs, self.state, rewards, dones, infos = self.jax_env.step(subkey, self.state, actions)
        
        # Update episode metrics
        for agent in self.agents:
            self.episode_metrics[agent]["cumulated_pure_reward"] = float(infos[agent]["cumulated_pure_reward"])
            self.episode_metrics[agent]["cumulated_modified_reward"] = float(infos[agent]["cumulated_modified_reward"])
            self.episode_metrics[agent]["cumulated_action_stats"] = np.array(infos[agent]["cumulated_action_stats"])
            self.rewards[agent] = float(rewards[agent])  # Convert to Python float
            self.dones[agent] = bool(dones["__all__"])  # Convert to Python bool
            self.infos[agent] = {
                "agent_id": agent,
                "cumulated_pure_reward": float(self.episode_metrics[agent]["cumulated_pure_reward"]),
                "cumulated_modified_reward": float(self.episode_metrics[agent]["cumulated_modified_reward"]),
                "cumulated_action_stats": np.array(self.episode_metrics[agent]["cumulated_action_stats"])
            }
        
        # Convert observations to numpy arrays if they aren't already
        obs_dict = {}
        for agent in self.agents:
            if isinstance(obs[agent], jnp.ndarray):
                obs_dict[agent] = np.array(obs[agent])
            else:
                obs_dict[agent] = obs[agent]
        
        return obs_dict, self.rewards, self.dones, self.infos

    def observe(self, agent):
        """Return the observation for the specified agent."""
        if self.state is None:
            return None
        obs, _ = self.jax_env.reset(self.key)  # Get current observation
        if isinstance(obs[agent], jnp.ndarray):
            return np.array(obs[agent])
        return obs[agent]

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