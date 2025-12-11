import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.make_env import make_env
from models.ppo.carracing_actor import CarRacingActor
from gymnasium.wrappers import RecordVideo


def evaluate_carracing(model_path, episodes=5, render=True, record=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = make_env("carracing", render_mode="rgb_array" if record else ("human" if render else None))

    # Video wrapper
    if record:
        video_folder = os.path.join("videos", "PPO", "carracing")
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(env, video_folder=video_folder, name_prefix="PPO-carracing", episode_trigger=lambda ep: True)
        print(f"📹 Recording videos to {video_folder}\n")

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Build actor
    actor = CarRacingActor(action_dim, max_action).to(device)

    # Load checkpoint safely. Prefer weights-only safe load, but fall back if needed.
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    checkpoint = None
    # Try the safe weights-only load first (PyTorch 2.6+ default behavior may require this)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # older torch may not accept weights_only kwarg; try default load
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"Model failed to load with default torch.load: {e}")
            raise
    except Exception as e:
        # If safe load fails due to restricted globals (common when file contains numpy scalars),
        # retry with weights_only=False. This can execute arbitrary code — only do if you trust the file.
        print("Safe weights-only load failed, retrying with weights_only=False (unsafe).")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e2:
            print(f"Model failed to load with unsafe fallback: {e2}")
            raise e2

    # Support several checkpoint formats
    try:
        if isinstance(checkpoint, dict):
            if 'actor_state_dict' in checkpoint:
                actor.load_state_dict(checkpoint['actor_state_dict'])
            elif 'actor' in checkpoint:
                actor.load_state_dict(checkpoint['actor'])
            else:
                actor.load_state_dict(checkpoint)
        else:
            actor.load_state_dict(checkpoint)
    except Exception as e:
        print("Failed to load actor weights from checkpoint. Make sure checkpoint contains actor state dict.")
        raise e

    actor.eval()

    print(f"Loaded model from {model_path}")
    if isinstance(checkpoint, dict) and 'episode' in checkpoint:
        print(f"  Trained for {checkpoint['episode']} episodes")
    if isinstance(checkpoint, dict) and 'val_reward' in checkpoint:
        print(f"  Validation reward: {checkpoint['val_reward']:.2f}")

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 1000:
            # Ensure state shape is (3,32,32) for CNN actor
            s = state
            if isinstance(s, np.ndarray) and s.ndim == 1:
                try:
                    s = s.reshape(3, 32, 32)
                except Exception:
                    pass

            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = actor(state_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]

            action = np.clip(action, -max_action, max_action)
            state, r, term, trunc, info = env.step(action)
            ep_reward += float(r)
            done = term or trunc
            steps += 1

        rewards.append(ep_reward)
        print(f"Episode {ep}: Steps={steps}, Reward={ep_reward:.2f}")

    env.close()

    avg = np.mean(rewards)
    std = np.std(rewards)
    print(f"\n📊 Results:")
    print(f"   Average Reward: {avg:.2f} ± {std:.2f}")
    print(f"   Min Reward: {np.min(rewards):.2f}")
    print(f"   Max Reward: {np.max(rewards):.2f}")
    if record:
        print("\n🎥 Videos saved to videos/PPO/carracing")

    return avg, std, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CarRacing PPO Agent")
    parser.add_argument("--model", type=str, default="saved_models/ppo/carracing_best.pth", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--record", action="store_true", help="Record video of episodes")
    args = parser.parse_args()

    evaluate_carracing(args.model, episodes=args.episodes, render=not args.no_render, record=args.record)
