import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import chex
from jaxmarl.environments import spaces
from typing import Optional, Tuple

@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    last_state: jnp.ndarray  # 2
    action_stats: jnp.ndarray


# Interacciones entre agentes (S -> no coger moneda, C -> coger moneda propia, D -> coger moneda ajena)
STATES = jnp.array(
    [
        [0],  # SS
        [1],  # CC
        [2],  # CD
        [3],  # DC
        [4],  # DD
        [5],  # SC
        [6],  # SD
        [7],  # CS
        [8],  # DS
    ]
)

MOVES = jnp.array(
    [
        [0, 1],  # right
        [0, -1],  # left
        [1, 0],  # up
        [-1, 0],  # down
        [0, 0],  # stay
    ]
)


class CoinGame_EASY(MultiAgentEnv):
    """
    JAX Compatible version of coin game environment.
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

        super().__init__(num_agents=2)
        self.agents = [str(f'agent_{i}') for i in list(range(2))]
        self.payoff_matrix = payoff_matrix
        self.grid_size = grid_size
        self.reward_coef = reward_coef
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.cnn = cnn
        self.egocentric = egocentric
        
        # Initialize all accumulated statistics as instance variables
        self._cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self._cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}
        self._cumulated_action_stats = {agent: jnp.zeros(5, dtype=jnp.int32) for agent in self.agents}

        def _abs_position(state: EnvState) -> jnp.ndarray:
            obs1 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs2 = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)

            # obs channels are [red_player, blue_player, red_coin, blue_coin]
            obs1 = obs1.at[state.red_pos[0], state.red_pos[1], 0].set(1)
            obs1 = obs1.at[state.blue_pos[0], state.blue_pos[1], 1].set(1)
            obs1 = obs1.at[
                state.red_coin_pos[0], state.red_coin_pos[1], 2
            ].set(1)
            obs1 = obs1.at[
                state.blue_coin_pos[0], state.blue_coin_pos[1], 3
            ].set(1)

            # each agent has egotistic color (so thinks they are red)
            obs2 = jnp.stack(
                [obs1[:, :, 1], obs1[:, :, 0], obs1[:, :, 3], obs1[:, :, 2]],
                axis=-1,
            )
            obs = {self.agents[0]: obs1, self.agents[1]: obs2}
            return obs

        def _relative_position(state: EnvState) -> jnp.ndarray:
            """Assume canonical agent is red player"""
            # (x) redplayer at (2, 2)
            # (y) redcoin at   (0 ,0)
            #
            #  o o x        o o y
            #  o o o   ->   o x o
            #  y o o        o o o
            #
            # redplayer goes to (1, 1)
            # redcoing goes to  (2, 2)
            # offset = (-1, -1)
            # new_redcoin = (0, 0) + (-1, -1) = (-1, -1) mod3
            # new_redcoin = (2, 2)

            agent_loc = jnp.array([state.red_pos[0], state.red_pos[1]])
            ego_offset = jnp.ones(2, dtype=jnp.int8) - agent_loc

            rel_other_player = (state.blue_pos + ego_offset) % self.grid_size
            rel_red_coin = (state.red_coin_pos + ego_offset) % self.grid_size
            rel_blue_coin = (state.blue_coin_pos + ego_offset) % self.grid_size

            # create observation
            obs = jnp.zeros((self.grid_size, self.grid_size, 4), dtype=jnp.int8)
            obs = obs.at[1, 1, 0].set(1)
            obs = obs.at[rel_other_player[0], rel_other_player[1], 1].set(1)
            obs = obs.at[rel_red_coin[0], rel_red_coin[1], 2].set(1)
            obs = obs.at[rel_blue_coin[0], rel_blue_coin[1], 3].set(1)
            return obs

        def _state_to_obs(state: EnvState) -> jnp.ndarray:
            if egocentric:
                obs1 = _relative_position(state)

                # flip red and blue coins for second agent
                obs2 = _relative_position(
                    EnvState(
                        red_pos=state.blue_pos,
                        blue_pos=state.red_pos,
                        red_coin_pos=state.blue_coin_pos,
                        blue_coin_pos=state.red_coin_pos,
                        inner_t=0,
                        outer_t=0,
                    )
                )
                obs = (obs1, obs2)
                obs = {agent: obs for agent, obs in zip(self.agents, obs)}
            else:
                obs = _abs_position(state)

            if not cnn:
                return {agent: obs[agent].flatten() for agent in obs}
            return obs
        
        def _is_adjacent(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

        def _classify_action(pos, coin_pos, got_coin):
            """Devuelve un vector one-hot de 5 elementos para la acción realizada"""
            own_adjacent = _is_adjacent(pos, coin_pos[0])
            other_adjacent = _is_adjacent(pos, coin_pos[1])

            if got_coin == 0:  # own
                return jnp.array([1, 0, 0, 0, 0])
            elif got_coin == 1:  # other
                return jnp.array([0, 1, 0, 0, 0])
            elif own_adjacent:
                return jnp.array([0, 0, 1, 0, 0])
            elif other_adjacent:
                return jnp.array([0, 0, 0, 1, 0])
            else:
                return jnp.array([0, 0, 0, 0, 1])
        
        def sample_two_valid_positions(key, red_coin_pos, blue_coin_pos, grid_size):
            all_positions = jnp.stack(jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size)), axis=-1).reshape(-1, 2)
            
            def is_invalid(pos):
                return jnp.any(jnp.all(pos == red_coin_pos, axis=-1)) | jnp.any(jnp.all(pos == blue_coin_pos, axis=-1))
    
            valid_mask = ~jax.vmap(is_invalid)(all_positions)
            valid_positions = all_positions[valid_mask]

            key, subkey = jax.random.split(key)
            permuted = jax.random.permutation(subkey, valid_positions.shape[0])
            chosen = valid_positions[permuted[:2]]
    
            return key, chosen
            
        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
        ):
            action_0, action_1 = list(actions.values())
            new_red_pos = (state.red_pos + MOVES[action_0]) % self.grid_size
            new_blue_pos = (state.blue_pos + MOVES[action_1]) % self.grid_size

            # Reward only for 'stay' action (index 4)
            red_reward = jnp.where(action_0 == 4, 1.0, 0.0)
            blue_reward = jnp.where(action_1 == 4, 1.0, 0.0)

            # The rest of the step logic remains unchanged, but all other reward logic is removed
            key, subkey = jax.random.split(key)
            key, new_random_coin_poses = sample_two_valid_positions(
                key, state.red_coin_pos, state.blue_coin_pos, self.grid_size
            )

            new_red_coin_pos = state.red_coin_pos
            new_blue_coin_pos = state.blue_coin_pos
        
            red_stats = _classify_action(
                new_red_pos,
                (state.red_coin_pos, state.blue_coin_pos),
                -1
            )

            blue_stats = _classify_action(
                new_blue_pos,
                (state.blue_coin_pos, state.red_coin_pos),
                -1
            )

            # Just store the current step's stats, don't accumulate here
            new_action_stats = jnp.stack([red_stats, blue_stats])

            last_state = state.last_state

            next_state = EnvState(
                red_pos=new_red_pos,
                blue_pos=new_blue_pos,
                red_coin_pos=new_red_coin_pos,
                blue_coin_pos=new_blue_coin_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                last_state=last_state,
                action_stats=new_action_stats
            )

            obs = _state_to_obs(next_state)

            # now calculate if done for inner or outer episode
            inner_t = next_state.inner_t
            outer_t = next_state.outer_t
            reset_inner = inner_t == num_inner_steps

            # Get current accumulated values before any reset
            cumulated_pure_rewards = {k: v for k, v in self._cumulated_pure_rewards.items()}
            cumulated_modified_rewards = {k: v for k, v in self._cumulated_modified_rewards.items()}
            cumulated_action_stats = {k: v for k, v in self._cumulated_action_stats.items()}

            # if inner episode is done, return start state for next game
            reset_obs, reset_state = _reset(key)
            next_state = EnvState(
                red_pos=jnp.where(
                    reset_inner, reset_state.red_pos, next_state.red_pos
                ),
                blue_pos=jnp.where(
                    reset_inner, reset_state.blue_pos, next_state.blue_pos
                ),
                red_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.red_coin_pos,
                    next_state.red_coin_pos,
                ),
                blue_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.blue_coin_pos,
                    next_state.blue_coin_pos,
                ),
                inner_t=jnp.where(
                    reset_inner, jnp.zeros_like(inner_t), next_state.inner_t
                ),
                outer_t=jnp.where(reset_inner, outer_t + 1, outer_t),
                last_state=jnp.where(reset_inner, jnp.zeros(2), last_state),
                action_stats=jnp.where(reset_inner, jnp.zeros((2, 5), dtype=jnp.int32), new_action_stats)
            )

            obs = {agent: obs for agent, obs in zip(self.agents, [jnp.where(reset_inner, reset_obs[i], obs[i]) for i in obs])}

            # Only reward for 'up' action
            rewards = {agent: reward for agent, reward in zip(self.agents, (red_reward, blue_reward))}
            pure_rewards = {"agent_0": red_reward, "agent_1": blue_reward}

            # Update cumulated rewards and stats
            for agent in self.agents:
                # Only update if not resetting
                self._cumulated_pure_rewards[agent] = jnp.where(
                    reset_inner,
                    0.0,
                    self._cumulated_pure_rewards[agent] + pure_rewards[agent]
                )
                self._cumulated_modified_rewards[agent] = jnp.where(
                    reset_inner,
                    0.0,  
                    self._cumulated_modified_rewards[agent] + rewards[agent]
                )
                # For action stats, we need to handle each agent's stats separately
                agent_idx = self.agents.index(agent)
                self._cumulated_action_stats[agent] = jnp.where(
                    reset_inner,
                    jnp.zeros(5, dtype=jnp.int32),
                    self._cumulated_action_stats[agent] + new_action_stats[agent_idx]
                )

            dones = {agent: reset_inner for agent in self.agents}
            dones['__all__'] = reset_inner

            infos = {
                agent: {
                    "cumulated_pure_reward": cumulated_pure_rewards[agent],
                    "cumulated_modified_reward": cumulated_modified_rewards[agent],
                    "cumulated_action_stats": cumulated_action_stats[agent]
                }
                for agent in self.agents
            }

            return (
                obs,
                next_state,
                rewards,
                dones,
                infos,
            )

        def _reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, EnvState]:
            key, subkey = jax.random.split(key)
            # First get random positions for agents
            agent_pos = jax.random.randint(
                subkey, shape=(2, 2), minval=0, maxval=self.grid_size
            )
            
            # Then get valid positions for coins using sample_two_valid_positions
            key, coin_pos = sample_two_valid_positions(
                key, agent_pos[0], agent_pos[1], self.grid_size
            )

            state = EnvState(
                red_pos=agent_pos[0, :],
                blue_pos=agent_pos[1, :],
                red_coin_pos=coin_pos[0],
                blue_coin_pos=coin_pos[1],
                inner_t=0,
                outer_t=0,
                last_state=jnp.zeros(2),
                action_stats = jnp.zeros((2, 5), dtype=jnp.int32)
            )
            obs = _state_to_obs(state)
            return obs, state

        def _update_stats(
            state: EnvState,
            rr: jnp.ndarray,
            rb: jnp.ndarray,
            br: jnp.ndarray,
            bb: jnp.ndarray,
        ):
            # actions are S, C, D
            a1 = 0
            a1 = jnp.where(rr, 1, a1)
            a1 = jnp.where(rb, 2, a1)

            a2 = 0
            a2 = jnp.where(bb, 1, a2)
            a2 = jnp.where(br, 2, a2)

            # if we didn't get a coin this turn, use the last convention
            convention_1 = jnp.where(a1 > 0, a1, state.last_state[0])
            convention_2 = jnp.where(a2 > 0, a2, state.last_state[1])

            convention = jnp.stack([convention_1, convention_2]).reshape(2)
            return convention

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)
        self.cnn = cnn

        self.step = _step
        self.reset = _reset

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoinGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(self, agent_id=None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        _shape = (self.grid_size, self.grid_size, 4) if self.cnn else (self.grid_size * self.grid_size * 4,)
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (self.grid_size, self.grid_size, 4) if self.cnn else (self.grid_size * self.grid_size * 4,)
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def render(self, state: EnvState):
        import numpy as np
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        from PIL import Image

        """Small utility for plotting the agent's state."""
        fig = Figure((5, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(121)
        ax.imshow(
            np.zeros((self.grid_size, self.grid_size)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=[0, 3, 0, 3],
        )
        ax.set_aspect("equal")

        # ax.margins(0)
        ax.set_xticks(jnp.arange(1, self.grid_size + 1))
        ax.set_yticks(jnp.arange(1, self.grid_size + 1))
        ax.grid()
        red_pos = jnp.squeeze(state.red_pos)
        blue_pos = jnp.squeeze(state.blue_pos)
        red_coin_pos = jnp.squeeze(state.red_coin_pos)
        blue_coin_pos = jnp.squeeze(state.blue_coin_pos)
        ax.annotate(
            "R",
            fontsize=20,
            color="red",
            xy=(red_pos[0], red_pos[1]),
            xycoords="data",
            xytext=(red_pos[0] + 0.5, red_pos[1] + 0.5),
        )
        ax.annotate(
            "B",
            fontsize=20,
            color="blue",
            xy=(blue_pos[0], blue_pos[1]),
            xycoords="data",
            xytext=(blue_pos[0] + 0.5, blue_pos[1] + 0.5),
        )
        ax.annotate(
            "Rc",
            fontsize=20,
            color="red",
            xy=(red_coin_pos[0], red_coin_pos[1]),
            xycoords="data",
            xytext=(red_coin_pos[0] + 0.3, red_coin_pos[1] + 0.3),
        )
        ax.annotate(
            "Bc",
            color="blue",
            fontsize=20,
            xy=(blue_coin_pos[0], blue_coin_pos[1]),
            xycoords="data",
            xytext=(
                blue_coin_pos[0] + 0.3,
                blue_coin_pos[1] + 0.3,
            ),
        )

        ax2 = fig.add_subplot(122)
        ax2.text(0.0, 0.95, "Timestep: %s" % (state.inner_t))
        ax2.text(0.0, 0.75, "Episode: %s" % (state.outer_t))
        ax2.text(
            0.0, 0.45, "Red Coop: %s" % (state.red_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.45,
            "Red Defects : %s" % (state.red_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0, 0.25, "Blue Coop: %s" % (state.blue_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.25,
            "Blue Defects : %s" % (state.blue_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0,
            0.05,
            "Red Total: %s"
            % (
                state.red_defect[state.outer_t].sum()
                + state.red_coop[state.outer_t].sum()
            ),
        )
        ax2.text(
            0.6,
            0.05,
            "Blue Total: %s"
            % (
                state.blue_defect[state.outer_t].sum()
                + state.blue_coop[state.outer_t].sum()
            ),
        )
        ax2.axis("off")
        canvas.draw()
        image = Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        return image


if __name__ == "__main__":
    action = 1
    rng = jax.random.PRNGKey(0)
    env = CoinGame(8, 16, True, False)

    # params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    # obs, state = env.reset(rng, params)
    # pics = []

    # for _ in range(16):
    #     rng, rng1, rng2 = jax.random.split(rng, 3)
    #     a1 = jax.random.randint(rng1, (), minval=0, maxval=4)
    #     a2 = jax.random.randint(rng2, (), minval=0, maxval=4)
    #     obs, state, reward, done, info = env.step(
    #         rng, state, (a1 * action, a2 * action), params
    #     )
    #     img = env.render(state)
    #     pics.append(img)

    # pics[0].save(
    #     "test1.gif",
    #     format="gif",
    #     save_all=True,
    #     append_images=pics[1:],
    #     duration=300,
    #     loop=0,
    # )