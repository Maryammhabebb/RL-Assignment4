# training/train_td3.py

import argparse
import numpy as np
import torch
import wandb
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.td3.agent import TD3Agent
from envs.make_env import make_env
from common.replay_buffer import ReplayBuffer


def train(env_name, episodes=500):

    print(f"\n🚀 Training TD3 on {env_name}…")

    # Create environment
    env = make_env(env_name)

    # Extract dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"💻 Using device: {device}")

    # Create agent + replay buffer
    agent = TD3Agent(state_dim, action_dim, max_action, device)

    replay_buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # Init Weights & Biases
    wandb.init(
        project="RL-Assignment4",
        name=f"TD3-{env_name}",
        config={
            "algorithm": "TD3",
            "environment": env_name,
            "episodes": episodes,
            "policy_noise": agent.policy_noise,
            "noise_clip": agent.noise_clip,
            "gamma": agent.gamma,
            "tau": agent.tau,
            "policy_freq": agent.policy_freq,
            "batch_size": agent.batch_size
        }
    )

    # ------------------------
    # Training Loop
    # ------------------------
    total_steps = 0
    warmup_steps = 10_000
    episode_rewards = []
    
    for ep in range(episodes):

        state, _ = env.reset()
        ep_reward = 0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):

            # Select action (random during warmup)
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            # Apply action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Learn from replay buffer
            metrics = agent.train(replay_buffer)

            # Log critic + actor + Q-value losses
            if metrics is not None:
                wandb.log({
                    "critic_loss": metrics["critic_loss"],
                    "Q_value_mean": metrics["Q_value_mean"],
                    "actor_loss": metrics["actor_loss"] if metrics["actor_loss"] is not None else 0,
                    "learning_step": agent.total_it,
                })

            # Update counters
            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

        # Track episode reward
        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards[-100:])

        # Log main episode stats
        wandb.log({
            "episode_reward": ep_reward,
            "avg_reward_100": avg_reward,
            "episode_steps": steps,
            "total_steps": total_steps,
        })

        # Print progress
        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Reward: {ep_reward:.2f} | "
                f"Avg(100): {avg_reward:.2f} | Steps: {steps}"
            )
        else:
            print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f}")

    # ------------------------
    # Save Trained Model
    # ------------------------
    save_dir = "saved_models/td3"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{env_name}.pth"
    torch.save(agent.actor.state_dict(), save_path)

    print(f"✅ Model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="lunarlander",
        help="Environment name: lunarlander or carracing"
    )
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train(args.env.lower(), episodes=args.episodes)
