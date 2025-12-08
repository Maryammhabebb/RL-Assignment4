# training/eval_agent.py

import argparse, os, torch, numpy as np
from gymnasium.wrappers import RecordVideo
from algorithms.td3.agent import TD3Agent
from envs.make_env import make_env


def evaluate(env_name, model_path, episodes=100, record=False):

    # 🎥 enable rendering ONLY for evaluation
    env = make_env(env_name, render_mode="rgb_array" if record else None)

    # 🎥 wrap video recorder BEFORE reset()
    if record:
        os.makedirs("videos", exist_ok=True)
        env = RecordVideo(env, video_folder="videos", name_prefix=f"TD3-{env_name}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.act(state, noise=0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        print(f"Episode {ep+1}: {ep_reward}")
        rewards.append(ep_reward)

    print("Average Reward:", np.mean(rewards))
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    evaluate(args.env, args.model, args.episodes, args.record)
