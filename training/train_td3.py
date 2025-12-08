# training/train_td3.py

import argparse
import numpy as np
import torch
import wandb

from algorithms.td3.agent import TD3Agent
from common.replay_buffer import ReplayBuffer
from envs.make_env import make_env


def train(env_name, episodes=500, max_steps=1000, seed=42):

    # -----------------------------
    # Setup
    # -----------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = TD3Agent(state_dim, action_dim, max_action, device=device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # -----------------------------
    # W&B Logging
    # -----------------------------
    wandb.init(
        project="RL-Assignment4",
        name=f"TD3-{env_name}",
        config={"env": env_name, "episodes": episodes}
    )

    # -----------------------------
    # Training Loop
    # -----------------------------
    total_steps = 0

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):

            # Select action + exploration noise
            action = agent.act(state, noise=0.1)

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Learning
            agent.train(replay_buffer)

            if done:
                break

        # -----------------------------
        # Logging
        # -----------------------------
        wandb.log({"episode_reward": episode_reward})

        print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward:.2f}")

    env.close()
    wandb.finish()


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        choices=["lunarlander", "carracing"])
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train(args.env, episodes=args.episodes)
