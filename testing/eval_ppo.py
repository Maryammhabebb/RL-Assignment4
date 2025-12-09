# testing/eval_ppo.py

import argparse, os, torch, numpy as np
import sys
from pathlib import Path
from gymnasium.wrappers import RecordVideo

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ppo.agent import PPOAgent
from envs.make_env import make_env


def evaluate_ppo(env_name, model_path, episodes=100, record=False):

    # 🎥 enable rendering ONLY for evaluation
    env = make_env(env_name, render_mode="rgb_array" if record else None)

    # 🎥 wrap video recorder BEFORE reset()
    if record:
        video_folder = os.path.join("videos", "PPO", env_name)
        os.makedirs(video_folder, exist_ok=True)
        # episode_trigger=lambda x: True records ALL episodes
        env = RecordVideo(
            env, 
            video_folder=video_folder, 
            name_prefix=f"PPO-{env_name}",
            episode_trigger=lambda episode_id: True  # Record every episode
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load PPO agent
    agent = PPOAgent(state_dim, action_dim, max_action, device)
    try:
        # PyTorch 2.6+ requires weights_only=False for numpy objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle both old (actor only) and new (full checkpoint) formats
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            if 'avg_reward' in checkpoint:
                print(f"✅ Loaded PPO model from {model_path}")
                print(f"   Training stats: {checkpoint.get('episodes', 'N/A')} episodes, "
                      f"Avg reward: {checkpoint['avg_reward']:.2f}")
        else:
            # Old format - just actor weights
            agent.actor.load_state_dict(checkpoint)
            print(f"✅ Loaded PPO model from {model_path} (legacy format)")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model!")
        print(f"   Make sure this is a PPO model trained with the current architecture.")
        print(f"   Retrain with: python training/train_ppo.py --env {env_name}")
        raise e

    rewards = []

    print(f"\n🎮 Evaluating PPO on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Use deterministic action
            action = agent.act(state, deterministic=True)
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
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent")
    parser.add_argument("--env", type=str, default="lunarlander", help="Environment: lunarlander or carracing")
    parser.add_argument("--model", type=str, default="saved_models/ppo/lunarlander.pth", help="Path to PPO model (.pth file)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true", help="Record video of episodes")
    args = parser.parse_args()

    evaluate_ppo(args.env, args.model, args.episodes, args.record)
