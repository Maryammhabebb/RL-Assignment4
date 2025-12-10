"""
Training script for SAC on continuous action environments
Usage: python train_sac.py --env LunarLanderContinuous-v3
"""

import argparse
import os
import sys
import numpy as np
import torch
import gymnasium as gym
import wandb
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so `models` package can be imported
# (when running the script directly from `training/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SAC.SAC import SAC
from models.SAC.config import SAC_CONFIGS, WANDB_CONFIG, GENERAL_CONFIG


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_env(env_name, seed):
    """Create and configure environment"""
    if env_name == "LunarLanderContinuous-v3":
        env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    elif env_name == "CarRacing-v3":
        env = gym.make("CarRacing-v3", continuous=True, render_mode=None)  # Added render_mode=None
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    env.reset(seed=seed)
    return env


def evaluate(agent, env, num_episodes=5):
    """Evaluate agent performance"""
    eval_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards), np.std(eval_rewards)


def train_sac(env_name, config, use_wandb=True):
    """Main training loop for SAC"""
    
    # Setup
    set_seed(config.get('seed', GENERAL_CONFIG['seed']))
    device = config.get('device', GENERAL_CONFIG['device'])
    
    # Create directories
    save_dir = Path(GENERAL_CONFIG['save_dir']) / env_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = create_env(env_name, config.get('seed', GENERAL_CONFIG['seed']))
    eval_env = create_env(env_name, config.get('seed', GENERAL_CONFIG['seed']) + 100)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    # Initialize SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        automatic_entropy_tuning=config['automatic_entropy_tuning'],
        device=device
    )
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity'],
            config=config,
            name=f"SAC-{env_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=WANDB_CONFIG['mode']
        )
    
    # Training variables
    total_steps = 0
    best_eval_reward = -float('inf')
    solved_count = 0
    # Track episode rewards for final statistics (like PPO)
    episode_rewards = []

    def _build_checkpoint_dict(ep, avg_reward):
        """Build a checkpoint dict from the agent in a few compatible key styles.
        Tries to include actor and critic state dicts if available."""
        ckpt = {
            'timesteps': total_steps,
            'episodes': ep,
            'avg_reward': float(avg_reward)
        }

        # Actor
        if hasattr(agent, 'actor'):
            try:
                ckpt['actor_state_dict'] = agent.actor.state_dict()
            except Exception:
                pass
        elif hasattr(agent, 'policy'):
            try:
                ckpt['actor_state_dict'] = agent.policy.state_dict()
            except Exception:
                pass

        # Critics: support several naming conventions
        if hasattr(agent, 'critic'):
            try:
                ckpt['critic_state_dict'] = agent.critic.state_dict()
            except Exception:
                pass
        else:
            # try critic_1 / critic_2
            if hasattr(agent, 'critic_1'):
                try:
                    ckpt['critic_1_state_dict'] = agent.critic_1.state_dict()
                except Exception:
                    pass
            if hasattr(agent, 'critic_2'):
                try:
                    ckpt['critic_2_state_dict'] = agent.critic_2.state_dict()
                except Exception:
                    pass

        return ckpt
    
    print("\nStarting training...")
    print("=" * 50)
    
    for episode in range(config['max_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(config['max_steps']):
            # Select action
            if total_steps < config['start_steps']:
                action = env.action_space.sample()  # Random exploration
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            
            # Update agent
            # (Training moved to per-episode) updates are performed after episode ends
            
            if done or truncated:
                break
        
        # Logging
        if episode % config['log_frequency'] == 0:
            print(f"Episode {episode}/{config['max_episodes']} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Total Steps: {total_steps}")
        
        if use_wandb:
            wandb.log({
                'train/episode_reward': episode_reward,
                'train/episode_steps': episode_steps,
                'train/episode': episode
            })

        # Save episode reward for final summary
        episode_rewards.append(episode_reward)
        # Perform SAC updates once per episode (instead of during steps)
        if total_steps >= config.get('update_after', 0):
            for _ in range(config.get('num_updates', 0)):
                critic_loss, actor_loss, alpha_loss, alpha = agent.update(config['batch_size'])
                if use_wandb and critic_loss is not None:
                    wandb.log({
                        'train/critic_loss': critic_loss,
                        'train/actor_loss': actor_loss,
                        'train/alpha_loss': alpha_loss if alpha_loss else 0,
                        'train/alpha': alpha,
                        'train/total_steps': total_steps,
                        'train/episode': episode
                    })
        
        # Evaluation
        if episode % config['eval_frequency'] == 0 and episode > 0:
            eval_reward, eval_std = evaluate(agent, eval_env, config['num_eval_episodes'])
            print(f"Evaluation | Mean Reward: {eval_reward:.2f} ± {eval_std:.2f}")
            
            if use_wandb:
                wandb.log({
                    'eval/mean_reward': eval_reward,
                    'eval/std_reward': eval_std,
                    'eval/episode': episode
                })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                # keep previous timestamped save for traceability
                agent.save(save_dir / 'best_model.pt')

                # Also save a standardized checkpoint into saved_models like PPO
                saved_base = Path('saved_models') / 'sac'
                saved_base.mkdir(parents=True, exist_ok=True)
                ckpt_path = saved_base / f"{env_name}_best.pth"
                try:
                    torch.save(_build_checkpoint_dict(episode, eval_reward), ckpt_path)
                    print(f"New best model saved! Reward: {eval_reward:.2f} -> {ckpt_path}")
                except Exception as e:
                    print(f"Warning: failed to write standardized best checkpoint: {e}")
            
            # Check if solved
            if eval_reward >= config['target_reward']:
                solved_count += 1
                if solved_count >= config['solved_episodes']:
                    print(f"\nEnvironment solved! Target reward {config['target_reward']} "
                          f"reached for {solved_count} consecutive evaluations.")
                    agent.save(save_dir / 'final_model.pt')
                    break
            else:
                solved_count = 0
        
        # Save checkpoint
        if episode % config['save_frequency'] == 0 and episode > 0:
            # Keep the original checkpoint in the timestamped folder
            agent.save(save_dir / f'checkpoint_ep{episode}.pt')

            # Also write a PPO-style checkpoint into saved_models/sac/checkpoints
            ckpt_dir = Path('saved_models') / 'sac' / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_file = ckpt_dir / f"{env_name}_ep{episode}.pth"
            try:
                torch.save(_build_checkpoint_dict(episode, episode_reward), ckpt_file)
                print(f"Checkpoint saved: {ckpt_file}")
            except Exception as e:
                print(f"Warning: failed to write standardized checkpoint: {e}")
    
    # Final save
    agent.save(save_dir / 'final_model.pt')

    # Also save final standardized model in saved_models/sac
    saved_base = Path('saved_models') / 'sac'
    saved_base.mkdir(parents=True, exist_ok=True)
    final_avg = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) > 0 else 0.0
    final_path = saved_base / f"{env_name}_final.pth"
    try:
        torch.save(_build_checkpoint_dict(config.get('max_episodes', episode), final_avg), final_path)
        print(f"\nTraining completed! Models saved to {save_dir} and {final_path}")
    except Exception as e:
        print(f"\nTraining completed! Models saved to {save_dir}. Failed to write standardized final: {e}")

    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train SAC on continuous environments')
    parser.add_argument('--env', type=str, required=True,
                      choices=['LunarLanderContinuous-v3', 'CarRacing-v3'],
                      help='Environment to train on')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                      choices=['cuda', 'cpu'],
                      help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = SAC_CONFIGS[args.env].copy()
    
    # Override with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    
    # Train
    train_sac(args.env, config, use_wandb=True)


if __name__ == "__main__":
    main()