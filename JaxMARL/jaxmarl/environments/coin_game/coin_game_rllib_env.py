import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple, Any
import chex
import copy
import os
import csv

@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    # stats
    red_coop: jnp.ndarray
    red_defect: jnp.ndarray
    blue_coop: jnp.ndarray
    blue_defect: jnp.ndarray
    counter: jnp.ndarray  # 9
    coop1: jnp.ndarray  # 9
    coop2: jnp.ndarray  # 9
    last_state: jnp.ndarray  # 2
    action_stats: jnp.ndarray

MOVES = jnp.array(
    [
        [0, 1],  # right
        [0, -1],  # left
        [1, 0],  # up
        [-1, 0],  # down
        [0, 0],  # stay
    ]
)

class CoinGameRLLibEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "CoinGameRLLibEnv"}

    def __init__(
        self,
        num_inner_steps: int = 10,
        num_outer_steps: int = 10,
        cnn: bool = False,
        egocentric: bool = False,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        grid_size: int = 3,
        reward_coef=[[1,0],[1,0]],
        path: str = "episode_log.csv",
        env_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        self.env_idx = env_idx
        self.episode_count = 0
        self.agents = [f"agent_{i}" for i in range(2)]
        #self.episode_infos_log = {agent: [] for agent in self.agents}
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.write_header = not os.path.exists(self.csv_path)

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.cnn = cnn
        self.egocentric = egocentric
        self.payoff_matrix = payoff_matrix
        self.grid_size = grid_size
        self.reward_coef = reward_coef

        self.possible_agents = self.agents.copy()

        _shape = (self.grid_size, self.grid_size, 4) if self.cnn else (self.grid_size * self.grid_size * 4,)
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=_shape, dtype=np.uint8)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.agents
        }

        self.key = jax.random.PRNGKey(0)
        self.state = None
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.episode_metrics = {agent: {
            "cumulated_pure_reward": 0.0,
            "cumulated_modified_reward": 0.0,
            "cumulated_action_stats": np.zeros(5, dtype=np.int32)
        } for agent in self.agents}
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.key, subkey = jax.random.split(self.key)
        obs, self.state = self._reset(subkey)
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            self.episode_metrics[agent] = {
                "cumulated_pure_reward": 0.0,
                "cumulated_modified_reward": 0.0,
                "cumulated_action_stats": np.zeros(5, dtype=np.int32)
            }
            self.cumulated_pure_rewards[agent] = 0.0
            self.cumulated_modified_rewards[agent] = 0.0
        obs_dict = {agent: np.array(obs[agent]) if isinstance(obs[agent], jnp.ndarray) else obs[agent] for agent in self.agents}
        info_dict = {agent: {} for agent in self.agents}
        return obs_dict, info_dict

    def step(self, actions):
        self.key, subkey = jax.random.split(self.key)
        obs, self.state, rewards, dones, infos = self._step(subkey, self.state, actions)
        for agent in self.agents:
            self.episode_metrics[agent]["cumulated_pure_reward"] = float(infos[agent]["cumulated_pure_reward"])
            self.episode_metrics[agent]["cumulated_modified_reward"] = float(infos[agent]["cumulated_modified_reward"])
            self.episode_metrics[agent]["cumulated_action_stats"] = np.array(infos[agent]["cumulated_action_stats"])
            self.rewards[agent] = float(rewards[agent])
            self.dones[agent] = bool(dones["__all__"])
            # Custom metrics for RLlib callback: these will be picked up and logged by the CustomMetricsCallback
            self.infos[agent] = {
                "agent_id": agent,
                "cumulated_pure_reward": float(self.episode_metrics[agent]["cumulated_pure_reward"]),
                "cumulated_modified_reward": float(self.episode_metrics[agent]["cumulated_modified_reward"]),
                "cumulated_action_stats": np.array(self.episode_metrics[agent]["cumulated_action_stats"])
            }
            #self.episode_infos_log[agent].append(copy.deepcopy(self.infos[agent]))

        obs_dict = {agent: np.array(obs[agent]) if isinstance(obs[agent], jnp.ndarray) else obs[agent] for agent in self.agents}
        # Split dones into terminateds and truncateds
        terminateds = {agent: self.dones[agent] for agent in self.agents}
        terminateds["__all__"] = self.dones["agent_0"] or self.dones["agent_1"]
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        # If episode is done, aggregate and log
        if self.dones["agent_0"] or self.dones["agent_1"]:
            row = {
                    "episode": self.episode_count,
                    "env": self.env_idx,
                }
            
            for agent in self.agents:
                #episode_data = self.episode_infos_log[agent]

                # Aggregate metrics
                total_pure_reward = float(self.episode_metrics[agent]["cumulated_pure_reward"])
                total_modified_reward = float(self.episode_metrics[agent]["cumulated_modified_reward"])
                action_stats_sum = np.array(self.episode_metrics[agent]["cumulated_action_stats"])

                row[f"pure_reward_{agent}"] = total_pure_reward
                row[f"modified_reward_{agent}"] = total_modified_reward
                row[f"own_coin_collected_{agent}"] = action_stats_sum[0]
                row[f"other_coin_collected_{agent}"] = action_stats_sum[1]
                row[f"reject_own_coin_{agent}"] = action_stats_sum[2]
                row[f"reject_other_coin_{agent}"] = action_stats_sum[3]
                row[f"no_coin_adjacent_{agent}"] = action_stats_sum[4]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if self.write_header:
                    writer.writeheader()
                    self.write_header = False
                writer.writerow(row)

            # Reset log for next episode
            self.episode_infos_log = {agent: [] for agent in self.agents}
            self.episode_count += 1

        return obs_dict, self.rewards, terminateds, truncateds, self.infos

    def _reset(self, key):
        key, subkey = jax.random.split(key)
        all_pos = jax.random.randint(subkey, shape=(4, 2), minval=0, maxval=self.grid_size)
        empty_stats = jnp.zeros((self.num_outer_steps), dtype=jnp.int8)
        state_stats = jnp.zeros(self.grid_size * self.grid_size)
        state = EnvState(
            red_pos=all_pos[0, :],
            blue_pos=all_pos[1, :],
            red_coin_pos=all_pos[2, :],
            blue_coin_pos=all_pos[3, :],
            inner_t=0,
            outer_t=0,
            red_coop=empty_stats,
            red_defect=empty_stats,
            blue_coop=empty_stats,
            blue_defect=empty_stats,
            counter=state_stats,
            coop1=state_stats,
            coop2=state_stats,
            last_state=jnp.zeros(2),
            action_stats=jnp.zeros((2, 5), dtype=jnp.int32)
        )
        obs = self._state_to_obs(state)
        return obs, state

    def _state_to_obs(self, state: EnvState):
        def _abs_position(state):
            obs1 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs2 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs1 = obs1.at[state.red_pos[0], state.red_pos[1], 0].set(1)
            obs1 = obs1.at[state.blue_pos[0], state.blue_pos[1], 1].set(1)
            obs1 = obs1.at[state.red_coin_pos[0], state.red_coin_pos[1], 2].set(1)
            obs1 = obs1.at[state.blue_coin_pos[0], state.blue_coin_pos[1], 3].set(1)
            obs2 = jnp.stack([obs1[:, :, 1], obs1[:, :, 0], obs1[:, :, 3], obs1[:, :, 2]], axis=-1)
            return {self.agents[0]: obs1, self.agents[1]: obs2}
        obs = _abs_position(state)
        return {agent: obs[agent].flatten() for agent in obs}

    def _step(self, key, state, actions):
        action_0, action_1 = actions["agent_0"], actions["agent_1"]
        new_red_pos = (state.red_pos + MOVES[action_0]) % self.grid_size
        new_blue_pos = (state.blue_pos + MOVES[action_1]) % self.grid_size
        red_reward, blue_reward = 0, 0
        red_red_matches = jnp.all(new_red_pos == state.red_coin_pos, axis=-1)
        red_blue_matches = jnp.all(new_red_pos == state.blue_coin_pos, axis=-1)
        blue_red_matches = jnp.all(new_blue_pos == state.red_coin_pos, axis=-1)
        blue_blue_matches = jnp.all(new_blue_pos == state.blue_coin_pos, axis=-1)
        _rr_reward = self.payoff_matrix[0][0]
        _rb_reward = self.payoff_matrix[0][1]
        _r_penalty = self.payoff_matrix[0][2]
        _br_reward = self.payoff_matrix[1][0]
        _bb_reward = self.payoff_matrix[1][1]
        _b_penalty = self.payoff_matrix[1][2]
        red_reward = jnp.where(red_red_matches, red_reward + _rr_reward, red_reward)
        red_reward = jnp.where(red_blue_matches, red_reward + _rb_reward, red_reward)
        red_reward = jnp.where(blue_red_matches, red_reward + _r_penalty, red_reward)
        blue_reward = jnp.where(blue_red_matches, blue_reward + _br_reward, blue_reward)
        blue_reward = jnp.where(blue_blue_matches, blue_reward + _bb_reward, blue_reward)
        blue_reward = jnp.where(red_blue_matches, blue_reward + _b_penalty, blue_reward)

        # --- Stats and done logic from original code ---
        def _classify_action(pos, coin_pos, got_coin):
            own_adjacent = abs(pos[0] - coin_pos[0][0]) + abs(pos[1] - coin_pos[0][1]) == 1
            other_adjacent = abs(pos[0] - coin_pos[1][0]) + abs(pos[1] - coin_pos[1][1]) == 1
            if got_coin == 0:
                return jnp.array([1, 0, 0, 0, 0])
            elif got_coin == 1:
                return jnp.array([0, 1, 0, 0, 0])
            elif own_adjacent:
                return jnp.array([0, 0, 1, 0, 0])
            elif other_adjacent:
                return jnp.array([0, 0, 0, 1, 0])
            else:
                return jnp.array([0, 0, 0, 0, 1])

        red_stats = _classify_action(
            new_red_pos,
            (state.red_coin_pos, state.blue_coin_pos),
            jnp.where(red_red_matches, 0, jnp.where(red_blue_matches, 1, -1))
        )
        blue_stats = _classify_action(
            new_blue_pos,
            (state.blue_coin_pos, state.red_coin_pos),
            jnp.where(blue_blue_matches, 0, jnp.where(blue_red_matches, 1, -1))
        )
        new_action_stats = state.action_stats + jnp.stack([red_stats, blue_stats])

        # Weight rewards
        rewards = {agent: reward for agent, reward in 
                           zip(self.agents, (self.reward_coef[0][0] * red_reward + self.reward_coef[0][1] * blue_reward, 
                                             self.reward_coef[1][0] * blue_reward + self.reward_coef[1][1] * red_reward))}
        
        # update cumulated rewards
        self.cumulated_pure_rewards["agent_0"] += float(red_reward)
        self.cumulated_pure_rewards["agent_1"] += float(blue_reward)        
        self.cumulated_modified_rewards["agent_0"] += float(rewards["agent_0"])
        self.cumulated_modified_rewards["agent_1"] += float(rewards["agent_1"])

        # Create dones dictionary
        inner_t = state.inner_t + 1
        outer_t = state.outer_t
        reset_inner = inner_t == self.num_inner_steps
        reset_outer = outer_t == self.num_outer_steps
        dones = {agent: reset_inner or reset_outer for agent in self.agents}
        dones['__all__'] = reset_inner or reset_outer

        # Create infos dictionary
        infos = {
            agent: {
                "cumulated_pure_reward": self.cumulated_pure_rewards[agent],
                "cumulated_modified_reward": self.cumulated_modified_rewards[agent],
                "cumulated_action_stats": np.array(new_action_stats[i])
            }
            for i, agent in enumerate(self.agents)
        }

        # Next state
        next_state = EnvState(
            red_pos=new_red_pos,
            blue_pos=new_blue_pos,
            red_coin_pos=state.red_coin_pos,
            blue_coin_pos=state.blue_coin_pos,
            inner_t=inner_t,
            outer_t=outer_t + int(reset_inner),
            red_coop=state.red_coop,
            red_defect=state.red_defect,
            blue_coop=state.blue_coop,
            blue_defect=state.blue_defect,
            counter=state.counter,
            coop1=state.coop1,
            coop2=state.coop2,
            last_state=state.last_state,
            action_stats=new_action_stats,
        )
        obs = self._state_to_obs(next_state)
        return obs, next_state, rewards, dones, infos

    # Optionally, add render, close, seed, etc. 