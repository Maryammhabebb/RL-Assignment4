"""
Training script for SAC on continuous action environments
Usage: python train_sac.py --env LunarLanderContinuous-v3
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import wandb
from datetime import datetime
from pathlib import Path

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
        env = gym.make("LunarLander-v3", continuous=True)
    elif env_name == "CarRacing-v3":
        env = gym.make("CarRacing-v3", continuous=True)
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
            if total_steps >= config['update_after'] and total_steps % config['update_every'] == 0:
                for _ in range(config['num_updates']):
                    critic_loss, actor_loss, alpha_loss, alpha = agent.update(config['batch_size'])
                    
                    if use_wandb and critic_loss is not None:
                        wandb.log({
                            'train/critic_loss': critic_loss,
                            'train/actor_loss': actor_loss,
                            'train/alpha_loss': alpha_loss if alpha_loss else 0,
                            'train/alpha': alpha,
                            'train/total_steps': total_steps
                        })
            
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
                agent.save(save_dir / 'best_model.pt')
                print(f"New best model saved! Reward: {eval_reward:.2f}")
            
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
            agent.save(save_dir / f'checkpoint_ep{episode}.pt')
    
    # Final save
    agent.save(save_dir / 'final_model.pt')
    print(f"\nTraining completed! Models saved to {save_dir}")
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
    train_sac(args.env, config, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()