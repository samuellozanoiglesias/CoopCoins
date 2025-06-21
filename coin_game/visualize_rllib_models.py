import os
import sys
import ray
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import argparse
from datetime import datetime

# Add the JaxMARL path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'JaxMARL'))

from jaxmarl.environments.coin_game.coin_game_rllib_env import CoinGameRLLibEnv
from jaxmarl.environments.coin_game.coin_game import CoinGame

def load_rllib_checkpoint(checkpoint_path):
    """Load a trained RLlib checkpoint."""
    from ray.rllib.algorithms.ppo import PPO
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    # Load the checkpoint
    trainer = PPO.from_checkpoint(checkpoint_path)
    return trainer

def create_visualization_env(config):
    """Create environment with same config as training."""
    return CoinGameRLLibEnv(
        num_inner_steps=config["NUM_INNER_STEPS"],
        num_outer_steps=config["NUM_EPOCHS"],
        cnn=False,
        egocentric=False,
        payoff_matrix=config["PAYOFF_MATRIX"],
        grid_size=config["GRID_SIZE"],
        reward_coef=config["REWARD_COEF"],
        path="temp_vis",
        env_idx=0
    )

def render_state_using_original_method(state, grid_size, step_info=None):
    """Render the current state using the original coin_game.py render method."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PIL import Image

    """Small utility for plotting the agent's state."""
    fig = Figure((8, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(121)
    ax.imshow(
        np.zeros((grid_size, grid_size)),
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="none",
        origin="lower",
        extent=[0, grid_size, 0, grid_size],
    )
    ax.set_aspect("equal")

    # ax.margins(0)
    ax.set_xticks(jnp.arange(1, grid_size + 1))
    ax.set_yticks(jnp.arange(1, grid_size + 1))
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
    
    # Use the original statistics from the state
    if hasattr(state, 'red_coop') and hasattr(state, 'red_defect'):
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
    else:
        # Fallback to action stats if coop/defect stats not available
        ax2.text(0.0, 0.45, "Agent 0 (Red) Action Stats:")
        ax2.text(0.0, 0.35, "  Own coins: %s" % (state.action_stats[0][0]))
        ax2.text(0.0, 0.25, "  Other coins: %s" % (state.action_stats[0][1]))
        ax2.text(0.0, 0.15, "  Reject own: %s" % (state.action_stats[0][2]))
        ax2.text(0.0, 0.05, "  Reject other: %s" % (state.action_stats[0][3]))
        
        ax2.text(0.6, 0.45, "Agent 1 (Blue) Action Stats:")
        ax2.text(0.6, 0.35, "  Own coins: %s" % (state.action_stats[1][0]))
        ax2.text(0.6, 0.25, "  Other coins: %s" % (state.action_stats[1][1]))
        ax2.text(0.6, 0.15, "  Reject own: %s" % (state.action_stats[1][2]))
        ax2.text(0.6, 0.05, "  Reject other: %s" % (state.action_stats[1][3]))
    
    # Add step info if provided
    if step_info:
        ax2.text(0.0, -0.1, f"Red reward: {step_info.get('red_reward', 0):.2f}", 
                transform=ax2.transAxes, fontsize=10)
        ax2.text(0.0, -0.2, f"Blue reward: {step_info.get('blue_reward', 0):.2f}", 
                transform=ax2.transAxes, fontsize=10)
        ax2.text(0.0, -0.3, f"Red action: {step_info.get('red_action', 'N/A')}", 
                transform=ax2.transAxes, fontsize=10)
        ax2.text(0.0, -0.4, f"Blue action: {step_info.get('blue_action', 'N/A')}", 
                transform=ax2.transAxes, fontsize=10)
    
    ax2.axis("off")
    canvas.draw()
    image = Image.frombytes(
        "RGB",
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb(),
    )
    return image

def get_action_name(action):
    """Convert action index to name."""
    actions = ["Right", "Left", "Up", "Down", "Stay"]
    return actions[action] if 0 <= action < len(actions) else f"Unknown({action})"

def visualize_episode(trainer, config, num_episodes=1, save_gif=True, output_dir="visualizations"):
    """Visualize one or more episodes using the trained model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = create_visualization_env(config)
    
    for episode in range(num_episodes):
        print(f"Visualizing episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        frames = []
        step_infos = []
        
        # Get policies
        policy_0 = trainer.get_policy("agent_0")
        policy_1 = trainer.get_policy("agent_1")
        
        episode_rewards = {"agent_0": 0, "agent_1": 0}
        
        # Run episode
        for step in range(config["NUM_INNER_STEPS"]):
            # Get actions from trained policies
            action_0 = policy_0.compute_single_action(obs["agent_0"])[0]
            action_1 = policy_1.compute_single_action(obs["agent_1"])[0]
            
            actions = {"agent_0": action_0, "agent_1": action_1}
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            episode_rewards["agent_0"] += rewards["agent_0"]
            episode_rewards["agent_1"] += rewards["agent_1"]
            
            # Create step info for visualization
            step_info = {
                'red_reward': rewards["agent_0"],
                'blue_reward': rewards["agent_1"],
                'red_action': get_action_name(action_0),
                'blue_action': get_action_name(action_1),
                'cumulative_red_reward': episode_rewards["agent_0"],
                'cumulative_blue_reward': episode_rewards["agent_1"]
            }
            step_infos.append(step_info)
            
            # Render frame using the original render method
            frame = render_state_using_original_method(env.state, config["GRID_SIZE"], step_info)
            frames.append(frame)
            
            # Check if episode is done
            if terminated["__all__"] or truncated["__all__"]:
                break
        
        # Save GIF
        if save_gif:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(output_dir, f"episode_{episode}_{timestamp}.gif")
            
            frames[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=500,  # 500ms per frame
                loop=0,
            )
            print(f"GIF saved: {gif_path}")
        
        # Print episode summary
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total steps: {len(frames)}")
        print(f"  Agent 0 (Red) total reward: {episode_rewards['agent_0']:.2f}")
        print(f"  Agent 1 (Blue) total reward: {episode_rewards['agent_1']:.2f}")
        print(f"  Agent 0 action stats: {env.state.action_stats[0]}")
        print(f"  Agent 1 action stats: {env.state.action_stats[1]}")
        print()

def load_config_from_file(config_path):
    """Load configuration from a saved config file."""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    if value.startswith('[') and value.endswith(']'):
                        # Handle lists
                        value = eval(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass  # Keep as string if conversion fails
                
                config[key] = value
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Visualize trained RLlib Coin Game models")
    parser.add_argument("checkpoint_path", help="Path to the RLlib checkpoint")
    parser.add_argument("--config", help="Path to config file (optional, will try to find automatically)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--no-gif", action="store_true", help="Don't save GIF files")
    
    args = parser.parse_args()
    
    # Find config file if not provided
    config_path = args.config
    if not config_path:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        potential_config = os.path.join(checkpoint_dir, "config.txt")
        if os.path.exists(potential_config):
            config_path = potential_config
            print(f"Found config file: {config_path}")
        else:
            print("Warning: No config file found. Using default values.")
            config_path = None
    
    # Load config
    if config_path and os.path.exists(config_path):
        config = load_config_from_file(config_path)
        print("Loaded configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        # Default config
        config = {
            "NUM_INNER_STEPS": 10,
            "NUM_EPOCHS": 10,
            "PAYOFF_MATRIX": [[1, 1, -2], [1, 1, -2]],
            "GRID_SIZE": 3,
            "REWARD_COEF": [[1, 0], [1, 0]]
        }
        print("Using default configuration")
    
    # Load trained model
    print(f"Loading checkpoint: {args.checkpoint_path}")
    trainer = load_rllib_checkpoint(args.checkpoint_path)
    
    # Visualize episodes
    visualize_episode(
        trainer, 
        config, 
        num_episodes=args.episodes,
        save_gif=not args.no_gif,
        output_dir=args.output_dir
    )
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main() 