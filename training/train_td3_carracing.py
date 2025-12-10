# training/train_td3_carracing.py

import argparse
import numpy as np
import torch
import wandb
import os
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from models.td3.carracing_agent import TD3CarRacingAgent
from envs.make_env import make_env
from common.replay_buffer import ReplayBuffer


# ---------------------------------------------------------
# Device selector
# ---------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_carracing(episodes=500):

    print("\n🚗 Training TD3 on CarRacing-v3…")

    device = torch.device(get_device())
    print(f"💻 Device: {device}")

    # Environment already handles:
    # - grayscale
    # - resize
    # - frame stacking
    # - normalization
    # - CHW shape
    env = make_env("carracing")

    # Get observation shape
    sample_obs, _ = env.reset()
    state_shape = sample_obs.shape    # e.g. (4, 84, 84)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3CarRacingAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=state_shape,
        action_dim=action_dim,
        device=device,
    )

    wandb.init(
        project="RL-Assignment4",
        name="TD3-CarRacing",
        config={"episodes": episodes}
    )

    warmup_steps = 20_000
    total_steps = 0
    episode_rewards = []

    for ep in range(episodes):

        state, _ = env.reset()
        ep_reward = 0
        terminated = truncated = False

        while not (terminated or truncated):

            # Action selection
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Train agent
            metrics = agent.train(replay_buffer)

            if metrics:
                wandb.log({
                    "critic_loss": metrics["critic_loss"],
                    "actor_loss": metrics["actor_loss"] or 0,
                    "Q_value_mean": metrics["Q_value_mean"],
                })

            state = next_state
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)
        avg100 = np.mean(episode_rewards[-100:])

        wandb.log({
            "episode_reward": ep_reward,
            "avg_reward_100": avg100,
        })

        print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f} | Avg100: {avg100:.1f}")

    # Save model
    os.makedirs("saved_models/td3", exist_ok=True)
    torch.save(agent.actor.state_dict(), "saved_models/td3/carracing.pth")

    print("\n✅ Saved model to saved_models/td3/carracing.pth\n")
    env.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train_carracing(episodes=args.episodes)
