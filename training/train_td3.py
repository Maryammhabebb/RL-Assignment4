# training/train_td3.py

import argparse
import numpy as np
import torch
import wandb
import os

from algorithms.td3.agent import TD3Agent
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create agent + replay buffer
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=state_dim,
        action_dim=action_dim
    )

    # Init Weights & Biases
    wandb.init(
        project="RL-Assignment4",
        name=f"TD3-{env_name}",
        config={"algorithm": "TD3", "episodes": episodes}
    )

    # ------------------------
    # Training Loop
    # ------------------------
    for ep in range(episodes):

        state, _ = env.reset()
        ep_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Select action from agent
            action = agent.act(state)

            # Apply action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Train agent — returns training metrics
            metrics = agent.train(replay_buffer)

            # Log critic + actor losses
            if metrics is not None:
                wandb.log({
                    "critic_loss": metrics["critic_loss"],
                    "Q_value_mean": metrics["Q_value_mean"],
                })

                # Actor updates happen every policy_freq steps
                if metrics["actor_loss"] is not None:
                    wandb.log({"actor_loss": metrics["actor_loss"]})

            # Update counters
            state = next_state
            ep_reward += reward

        # Log episode reward
        wandb.log({"episode_reward": ep_reward})
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
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train(args.env.lower(), episodes=args.episodes)
