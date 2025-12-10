# training/train_ppo.py

import argparse
import numpy as np
import torch
import wandb
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.ppo.agent import PPOAgent
from envs.make_env import make_env


def train(env_name, episodes=10000):
    """
    Train PPO on continuous control environments.
    
    For LunarLander-v3 continuous:
    - Minimum: 7,000 episodes (~2M timesteps)
    - Recommended: 10,000-15,000 episodes (~3-5M timesteps)
    """

    print(f"\n{'='*60}")
    print(f"🚀 PPO Training: {env_name.upper()} (Continuous Control)")
    print(f"{'='*60}")
    print(f"📊 Target episodes: {episodes:,}")
    print(f"⚠️  Continuous control is challenging - be patient!\n")

    # Create environment
    env = make_env(env_name)

    # Get environment info
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
    print()

    # Create agent
    agent = PPOAgent(state_dim, action_dim, max_action, device)

    # Init Weights & Biases
    wandb.init(
        project="RL-Assignment4",
        name=f"PPO-{env_name}-continuous-{episodes}eps",
        config={
            "algorithm": "PPO",
            "environment": f"{env_name}-continuous",
            "episodes": episodes,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "buffer_size": agent.buffer_size,
            "batch_size": agent.batch_size,
            "epochs": agent.epochs,
            "actor_lr": agent.actor_lr,
            "critic_lr": agent.critic_lr,
            "gamma": agent.gamma,
            "gae_lambda": agent.gae_lambda,
            "epsilon": agent.epsilon,
            "entropy_coef": agent.entropy_coef,
            "device": device
        }
    )
    print("📊 Weights & Biases initialized\n")

    # ------------------------
    # Training Loop
    # ------------------------
    total_timesteps = 0
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -float('inf')
    training_count = 0
    
    print("Starting training...")
    print(f"⌛ Buffer fills every ~{agent.buffer_size} steps, then trains for {agent.epochs} epochs\n")
    
    for episode in range(episodes):
        
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            
            # ALWAYS use stochastic actions during training
            # (PPO learns a stochastic policy)
            action = agent.act(state, deterministic=False)
            
            # Clip actions to environment bounds (safety)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(reward, done)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # Train when buffer is full
            if len(agent.states) >= agent.buffer_size:
                training_count += 1
                actor_loss, critic_loss, entropy = agent.train(next_state)
                
                # Only log if training actually happened (not dummy values)
                if actor_loss != 0.0 or critic_loss != 0.0:
                    wandb.log({
                        "losses/actor_loss": actor_loss,
                        "losses/critic_loss": critic_loss,
                        "losses/entropy": entropy,
                        "losses/total_loss": actor_loss + critic_loss,
                        "training/update_count": training_count,
                        "timestep": total_timesteps
                    })
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Calculate statistics
        avg_reward_10 = np.mean(episode_rewards[-10:])
        avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
        
        # Track best performance
        if avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100
        
        # Log to wandb
        wandb.log({
            "episode/reward": episode_reward,
            "episode/length": episode_steps,
            "episode/avg_reward_10": avg_reward_10,
            "episode/avg_reward_100": avg_reward_100,
            "episode/best_avg_reward": best_avg_reward,
            "episode/number": episode + 1,
            "episode/avg_length": avg_length,
            "buffer/size": len(agent.states),
            "buffer/utilization": len(agent.states) / agent.buffer_size,
            "training/updates": training_count,
            "timestep": total_timesteps
        })
        
        # Print progress
        progress_pct = ((episode + 1) / episodes) * 100
        
        if (episode + 1) % 10 == 0:
            print(f"[{progress_pct:5.1f}%] "
                  f"Ep {episode+1:>5}/{episodes} | "
                  f"Steps {total_timesteps:>8,} | "
                  f"Trains {training_count:>4} | "
                  f"R={episode_reward:>7.1f} | "
                  f"Avg10={avg_reward_10:>7.1f} | "
                  f"Avg100={avg_reward_100:>7.1f} | "
                  f"Best={best_avg_reward:>7.1f}")
        
        # Debug info every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"    📊 Buffer: {len(agent.states)} states | "
                  f"Avg ep length: {avg_length:.0f} | "
                  f"Training updates: {training_count}")
        
        # Save checkpoints at milestones
        if avg_reward_100 > 150 and (episode + 1) % 500 == 0:
            checkpoint_dir = "saved_models/ppo/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/{env_name}_ep{episode+1}_r{avg_reward_100:.0f}.pth"
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'timesteps': total_timesteps,
                'episodes': episode + 1,
                'avg_reward': avg_reward_100
            }, checkpoint_path)
            print(f"    💾 Checkpoint saved: {checkpoint_path}")
        
        # Run deterministic evaluation every 100 episodes
        if (episode + 1) % 100 == 0:
            eval_rewards = []
            for _ in range(10):
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                while not eval_done:
                    eval_action = agent.act(eval_state, deterministic=True)
                    eval_action = np.clip(eval_action, env.action_space.low, env.action_space.high)
                    eval_state, r, terminated, truncated, _ = env.step(eval_action)
                    eval_reward += r
                    eval_done = terminated or truncated
                eval_rewards.append(eval_reward)
            
            eval_mean = np.mean(eval_rewards)
            wandb.log({
                "eval/mean_reward": eval_mean,
                "eval/std_reward": np.std(eval_rewards),
                "episode/number": episode + 1,
                "timestep": total_timesteps
            })
            print(f"    🎯 Eval (deterministic, 10 eps): {eval_mean:.1f} ± {np.std(eval_rewards):.1f}")
    
    # Final training of remaining samples
    if len(agent.states) > 0:
        print(f"\nFinal training with {len(agent.states)} remaining samples...")
        agent.train(state)

    # ------------------------
    # Save Final Model
    # ------------------------
    save_dir = "saved_models/ppo"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{save_dir}/{env_name}_final.pth"
    final_avg = np.mean(episode_rewards[-100:])
    
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'timesteps': total_timesteps,
        'episodes': episodes,
        'avg_reward': final_avg,
        'best_avg_reward': best_avg_reward
    }, save_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"📈 Statistics:")
    print(f"   Total episodes: {episodes:,}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Total training updates: {training_count}")
    print(f"   Final avg reward (100 eps): {final_avg:.2f}")
    print(f"   Best avg reward: {best_avg_reward:.2f}")
    print(f"   Avg episode length: {np.mean(episode_lengths[-100:]):.0f} steps")
    print(f"\n💾 Model saved: {save_path}")
    print(f"{'='*60}\n")
    
    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on continuous control environments"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="lunarlander",
        help="Environment: lunarlander or carracing"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500,
        help="Number of training episodes"
    )
    args = parser.parse_args()
    
    train(args.env.lower(), episodes=args.episodes)