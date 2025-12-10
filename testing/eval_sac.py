# testing/eval_sac.py

import argparse, os, torch, numpy as np
import sys
from pathlib import Path
from gymnasium.wrappers import RecordVideo

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Project root (RL-Assignment4) so outputs go inside the repo, not outside
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from models.SAC.SAC import SAC
from envs.make_env import make_env


def evaluate_sac(env_name, model_path, episodes=100, record=False):

    # 🎥 enable rendering ONLY for evaluation
    env = make_env(env_name, render_mode="rgb_array" if record else None)

    # 🎥 wrap video recorder BEFORE reset()
    if record:
        # Save videos into the repository's `videos/SAC/<env_name>` folder
        video_folder = PROJECT_ROOT / "videos" / "SAC" / env_name
        video_folder.mkdir(parents=True, exist_ok=True)
        video_folder_str = str(video_folder)
        env = RecordVideo(
            env,
            video_folder=video_folder_str,
            name_prefix=f"SAC-{env_name}",
            episode_trigger=lambda episode_id: True
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) if hasattr(env.action_space, 'high') else 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    if device == "cuda":
        try:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Load SAC agent
    agent = SAC(state_dim=state_dim, action_dim=action_dim, device=device)
    loaded = False
    try:
        # Try loading checkpoint as a dict (supports various formats)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            # New-style SAC save: keys like 'actor' and 'critic'
            if 'actor' in ckpt and 'critic' in ckpt:
                agent.actor.load_state_dict(ckpt['actor'])
                try:
                    agent.critic.load_state_dict(ckpt['critic'])
                except Exception:
                    pass
                print(f"✅ Loaded SAC full checkpoint from {model_path}")
                loaded = True

            # Common PPO-style checkpoint keys
            elif 'actor_state_dict' in ckpt:
                agent.actor.load_state_dict(ckpt['actor_state_dict'])
                if 'critic_state_dict' in ckpt:
                    try:
                        agent.critic.load_state_dict(ckpt['critic_state_dict'])
                    except Exception:
                        pass
                print(f"✅ Loaded SAC actor/critic checkpoint from {model_path}")
                loaded = True

            # If checkpoint contains metrics (e.g., avg_reward), print them
            if 'avg_reward' in ckpt:
                print(f"   Checkpoint stats - episodes: {ckpt.get('episodes', 'N/A')}, avg_reward: {ckpt['avg_reward']}")

            # Legacy: if dict doesn't match expected keys, try loading into actor
            if not loaded:
                try:
                    agent.actor.load_state_dict(ckpt)
                    print(f"✅ Loaded SAC actor weights (legacy dict) from {model_path}")
                    loaded = True
                except Exception:
                    loaded = False

        # If ckpt was not a dict, it might be a bare state_dict - try loading into actor
        elif not loaded:
            try:
                agent.actor.load_state_dict(ckpt)
                print(f"✅ Loaded SAC actor weights from {model_path}")
                loaded = True
            except Exception:
                loaded = False

    except Exception:
        loaded = False

    # Fallback: let agent's load() handle its native checkpoint format
    if not loaded:
        try:
            agent.load(model_path)
            print(f"✅ Loaded SAC model via agent.load() from {model_path}")
            loaded = True
        except Exception as e:
            print(f"❌ ERROR: Failed to load SAC model!")
            print(f"   Make sure this is a SAC model trained with the current architecture.")
            print(f"   Retrain with: python training/train_sac.py --env {env_name}")
            raise e

    rewards = []

    print(f"\n🎮 Evaluating SAC on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Use deterministic action (mean)
            action = agent.select_action(state, evaluate=True)
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
        # report the repo-relative video folder
        try:
            print(f"\n🎥 Videos saved to {video_folder_str}")
        except NameError:
            print("\n🎥 Videos were recorded (folder path not available)")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent")
    parser.add_argument("--env", type=str, default="lunarlander", help="Environment: lunarlander or carracing")
    parser.add_argument("--model", type=str, default="saved_models/sac/lunarlander.pth", help="Path to SAC model (.pth file)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true", help="Record video of episodes")
    args = parser.parse_args()

    evaluate_sac(args.env, args.model, args.episodes, args.record)
