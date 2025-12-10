"""
Evaluation script for SAC
Runs 100 episodes, plots rewards, and records 3 videos.
"""

import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.SAC.SAC import SAC
from models.SAC.config import SAC_CONFIGS, GENERAL_CONFIG


def evaluate_episode(env, agent):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward


def load_sac_model(env_name, model_path, device="cpu"):
    """Load SAC agent and attach weights."""
    env = gym.make(
        "LunarLander-v3", continuous=True, render_mode=None
        if env_name == "LunarLanderContinuous-v3"
        else "CarRacing-v3"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = SAC_CONFIGS[env_name]

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        automatic_entropy_tuning=config["automatic_entropy_tuning"],
        device=device
    )

    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    return agent


def record_video(env_name, agent, out_dir, video_id):
    """Record a single evaluation video."""
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        render_mode="rgb_array"
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(out_dir),
        name_prefix=f"video_{video_id}"
    )

    evaluate_episode(env, agent)
    env.close()
    print(f"Saved video {video_id}")


def main():
    env_name = "LunarLanderContinuous-v3"
    model_path = "trained_models/sac/LunarLanderContinuous-v3/20251210_122349/best_model.pt"  # Change if needed
    device = GENERAL_CONFIG["device"]

    # Output directory
    out_dir = Path("test_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load agent
    agent = load_sac_model(env_name, model_path, device)

    # Evaluate 100 episodes
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    rewards = []

    print("Running 100 evaluation episodes...")

    for ep in range(100):
        ep_reward = evaluate_episode(env, agent)
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")

    env.close()

    rewards = np.array(rewards)

    # Plot graph
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("SAC Evaluation — Reward per Episode (100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(out_dir / "rewards.png")
    plt.close()

    print(f"Saved reward plot → {out_dir/'rewards.png'}")

    # Record 3 videos
    for vid in range(1, 4):
        record_video(env_name, agent, out_dir, vid)

    print("\nAll done!")
    print(f"Results stored in: {out_dir}")


if __name__ == "__main__":
    main()
