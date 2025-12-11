import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from models.SAC.SAC import SAC


def load_agent(model_path, state_dim, action_dim, device="cpu"):
    """Load SAC agent from checkpoint"""
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        device=device
    )
    agent.load(model_path)
    return agent


def evaluate_episode(env, agent):
    """Run one episode and return its reward"""
    state, _ = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward


def main():
    MODEL_PATH = "C:/Users/ziada/Downloads/final_model.pt"   # << CHANGE IF NEEDED
    ENV_NAME = "LunarLander-v3"

    # Create environment
    env = gym.make(ENV_NAME, continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load agent
    print("Loading model...")
    agent = load_agent(MODEL_PATH, state_dim, action_dim)

    rewards = []

    print("Evaluating 100 episodes...")
    for ep in range(100):
        r = evaluate_episode(env, agent)
        rewards.append(r)
        print(f"Episode {ep+1}: reward = {r:.2f}")

    env.close()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC Agent Evaluation (100 Episodes)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
