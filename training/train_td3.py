# training/train_td3.py

import argparse
import numpy as np
import torch
import wandb
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.td3.agent import TD3Agent
from envs.make_env import make_env
from common.replay_buffer import ReplayBuffer


def train(env_name, episodes=500):

    print(f"\n{'='*60}")
    print(f"🚀 TD3 Training: {env_name.upper()}")
    print(f"{'='*60}")
    print(f"📊 Target episodes: {episodes:,}")
    print()

    # Create environment
    env = make_env(env_name)

    # Extract dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"📊 Environment Info:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Action range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Create agent + replay buffer
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # Init Weights & Biases
    wandb.init(
        project="RL-Assignment4",
        name=f"TD3-{env_name}-{episodes}eps",
        config={
            "algorithm": "TD3",
            "environment": env_name,
            "episodes": episodes,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "replay_buffer_size": 1_000_000,
            "warmup_steps": 10000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "device": device
        }
    )
    print("📊 Weights & Biases initialized\n")

    # ------------------------
    # Training Loop
    # ------------------------
    total_steps = 0
    warmup_steps = 10000  # Random actions for exploration
    episode_rewards = []
    best_avg_reward = -float('inf')
    
    print("Starting training...\n")
    
    for ep in range(episodes):

        state, _ = env.reset()
        ep_reward = 0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):

            # Select action from agent (random during warmup)
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            # Apply action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Train agent — returns training metrics
            metrics = agent.train(replay_buffer)

            # Log critic + actor losses
            if metrics is not None:
                wandb.log({
                    "losses/critic_loss": metrics["critic_loss"],
                    "losses/Q_value_mean": metrics["Q_value_mean"],
                    "timestep": total_steps
                })

                # Actor updates happen every policy_freq steps
                if metrics["actor_loss"] is not None:
                    wandb.log({
                        "losses/actor_loss": metrics["actor_loss"],
                        "timestep": total_steps
                    })

            # Update counters
            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

        # Track episode rewards
        episode_rewards.append(ep_reward)
        avg_reward_10 = np.mean(episode_rewards[-10:])
        avg_reward_100 = np.mean(episode_rewards[-100:])  # Last 100 episodes
        
        # Track best performance
        if avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100
        
        # Log episode reward
        wandb.log({
            "episode/reward": ep_reward,
            "episode/length": steps,
            "episode/avg_reward_10": avg_reward_10,
            "episode/avg_reward_100": avg_reward_100,
            "episode/best_avg_reward": best_avg_reward,
            "episode/number": ep + 1,
            "replay_buffer/size": replay_buffer.size,
            "replay_buffer/utilization": replay_buffer.size / replay_buffer.max_size,
            "training/warmup_complete": 1.0 if total_steps >= warmup_steps else 0.0,
            "timestep": total_steps
        })
        
        # Print progress
        progress_pct = ((ep + 1) / episodes) * 100
        
        if (ep + 1) % 10 == 0:
            print(f"[{progress_pct:5.1f}%] "
                  f"Ep {ep+1:>5}/{episodes} | "
                  f"Steps {total_steps:>8,} | "
                  f"R={ep_reward:>7.1f} | "
                  f"Avg10={avg_reward_10:>7.1f} | "
                  f"Avg100={avg_reward_100:>7.1f} | "
                  f"Best={best_avg_reward:>7.1f}")
        
        # Debug info every 100 episodes
        if (ep + 1) % 100 == 0:
            print(f"    📊 Buffer: {replay_buffer.size}/{replay_buffer.max_size} | "
                  f"Warmup: {'Complete' if total_steps >= warmup_steps else f'{total_steps}/{warmup_steps}'}")
        
        # Save checkpoints at milestones
        if avg_reward_100 > 150 and (ep + 1) % 100 == 0:
            checkpoint_dir = "saved_models/td3/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/{env_name}_ep{ep+1}_r{avg_reward_100:.0f}.pth"
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'target_actor_state_dict': agent.target_actor.state_dict(),
                'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
                'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
                'timesteps': total_steps,
                'episodes': ep + 1,
                'avg_reward': avg_reward_100
            }, checkpoint_path)
            print(f"    💾 Checkpoint saved: {checkpoint_path}")

    # ------------------------
    # Save Final Model
    # ------------------------
    save_dir = "saved_models/td3"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{env_name}_final.pth"
    final_avg = np.mean(episode_rewards[-100:])
    
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_actor_state_dict': agent.target_actor.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'timesteps': total_steps,
        'episodes': episodes,
        'avg_reward': final_avg,
        'best_avg_reward': best_avg_reward
    }, save_path)

    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"📈 Statistics:")
    print(f"   Total episodes: {episodes:,}")
    print(f"   Total timesteps: {total_steps:,}")
    print(f"   Final avg reward (100 eps): {final_avg:.2f}")
    print(f"   Best avg reward: {best_avg_reward:.2f}")
    print(f"   Replay buffer final size: {replay_buffer.size:,}")
    print(f"\n💾 Model saved: {save_path}")
    print(f"{'='*60}\n")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="lunarlander", help="Environment name: lunarlander or carracing (default: lunarlander)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes (default: 1000, recommended: 1000-2000 for best results)")
    args = parser.parse_args()

    train(args.env.lower(), episodes=args.episodes)
