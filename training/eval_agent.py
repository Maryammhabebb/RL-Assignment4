# training/eval_agent.py

import argparse
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from algorithms.td3.agent import TD3Agent
from envs.make_env import make_env


def evaluate(env_name, model_path, episodes=5, record=False):

    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))

    if record:
        env = RecordVideo(env, f"videos/TD3-{env_name}")

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0

        done = False
        while not done:
            action = agent.act(state, noise=0)   # deterministic
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"Episode {ep+1} Reward: {ep_reward}")

    env.close()
    print("Average Reward:", np.mean(rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    evaluate(args.env, args.model, record=args.record)
