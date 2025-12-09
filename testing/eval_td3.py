# testing/eval_td3.py

import argparse, os, torch, numpy as np
import sys
from pathlib import Path
from gymnasium.wrappers import RecordVideo

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.td3.agent import TD3Agent
from envs.make_env import make_env


def evaluate_td3(env_name, model_path, episodes=100, record=False):

    # 🎥 enable rendering ONLY for evaluation
    env = make_env(env_name, render_mode="rgb_array" if record else None)

    # 🎥 wrap video recorder BEFORE reset()
    if record:
        video_folder = os.path.join("videos", "TD3", env_name)
        os.makedirs(video_folder, exist_ok=True)
        # episode_trigger=lambda x: True records ALL episodes
        env = RecordVideo(
            env, 
            video_folder=video_folder, 
            name_prefix=f"TD3-{env_name}",
            episode_trigger=lambda episode_id: True  # Record every episode
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load TD3 agent
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    try:
        # PyTorch 2.6+ requires weights_only=False for numpy objects
        agent.actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"✅ Loaded TD3 model from {model_path}")
    except RuntimeError as e:
        print(f"❌ ERROR: Failed to load model!")
        print(f"   Make sure this is a TD3 model trained with the current architecture.")
        print(f"   Retrain with: python training/train_td3.py --env {env_name}")
        raise e

    rewards = []

    print(f"\n🎮 Evaluating TD3 on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Use deterministic action (no noise)
            action = agent.act(state, noise=0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        print(f"Episode {ep+1}/{episodes}: {ep_reward:.2f}")
        rewards.append(ep_reward)

    print(f"\n📊 Results:")
    print(f"   Average Reward: {np.mean(rewards):.2f}")
    print(f"   Std Deviation:  {np.std(rewards):.2f}")
    print(f"   Min Reward:     {np.min(rewards):.2f}")
    print(f"   Max Reward:     {np.max(rewards):.2f}")
    
    if record:
        print(f"\n🎥 Videos saved to {video_folder}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TD3 Agent")
    parser.add_argument("--env", type=str, default="lunarlander", help="Environment: lunarlander or carracing")
    parser.add_argument("--model", type=str, default="saved_models/td3/lunarlander.pth", help="Path to TD3 model (.pth file)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true", help="Record video of episodes")
    args = parser.parse_args()

    evaluate_td3(args.env, args.model, args.episodes, args.record)
