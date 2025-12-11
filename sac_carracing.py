import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import argparse
import os
import gymnasium as gym
from datetime import datetime
from pathlib import Path

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class CNNEncoder(nn.Module):
    """CNN encoder for image observations"""
    def __init__(self, input_channels=3, output_dim=256):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        # Input: 96x96
        # After conv1: (96-8)/4 + 1 = 23
        # After conv2: (23-4)/2 + 1 = 10
        # After conv3: (10-3)/1 + 1 = 8
        self.conv_output_size = 64 * 8 * 8
        
        self.fc = nn.Linear(self.conv_output_size, output_dim)
        
    def forward(self, x):
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Ensure correct shape: (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 3:  # If channels are last
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class ActorCNN(nn.Module):
    """CNN-based policy network for CarRacing"""
    def __init__(self, action_dim, input_channels=3, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorCNN, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.encoder = CNNEncoder(input_channels, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        return action, log_prob, mean


class CriticCNN(nn.Module):
    """CNN-based Q-network for CarRacing"""
    def __init__(self, action_dim, input_channels=3, hidden_dim=256):
        super(CriticCNN, self).__init__()
        
        # Shared encoder
        self.encoder = CNNEncoder(input_channels, hidden_dim)
        
        # Q1 network
        self.fc1_q1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc1_q2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = self.encoder(state)
        xa = torch.cat([x, action], 1)
        
        # Q1 forward
        q1 = F.relu(self.fc1_q1(xa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2 forward
        q2 = F.relu(self.fc1_q2(xa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        x = self.encoder(state)
        xa = torch.cat([x, action], 1)
        q1 = F.relu(self.fc1_q1(xa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1


class SAC_CarRacing:
    """Soft Actor-Critic for CarRacing with CNN"""
    def __init__(
        self,
        action_dim,
        input_channels=3,
        hidden_dim=256,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        automatic_entropy_tuning=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.action_dim = action_dim
        
        # Actor network
        self.actor = ActorCNN(action_dim, input_channels, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks
        self.critic = CriticCNN(action_dim, input_channels, hidden_dim).to(device)
        self.critic_target = CriticCNN(action_dim, input_channels, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def select_action(self, state, evaluate=False):
        # Ensure state is in correct format
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None, None, None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature (alpha)
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss, self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
    
    def save(self, filepath):
        """Save model to filepath"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'config': {
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim if hasattr(self, 'hidden_dim') else 256,
                'automatic_entropy_tuning': self.automatic_entropy_tuning
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from filepath"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
        print(f"Model loaded from {filepath}")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_env(seed):
    """Create CarRacing environment"""
    env = gym.make("CarRacing-v2", continuous=True, render_mode=None)
    env.reset(seed=seed)
    return env



def evaluate(agent, env, num_episodes=3, render=False):
    """Evaluate agent performance"""
    eval_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 1000:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        
        eval_rewards.append(episode_reward)
        print(f"  Eval episode {ep+1}/{num_episodes}: {episode_reward:.2f}")
    
    return np.mean(eval_rewards), np.std(eval_rewards)


def train_sac_carracing(config, use_wandb=False):
    """Main training loop for SAC on CarRacing"""
    
    # Setup
    set_seed(config['seed'])
    device = config['device']
    
    # Create directories - save to Kaggle working directory
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # Running on Kaggle
        save_dir = Path("/kaggle/working/models")
    else:
        # Running locally
        save_dir = Path(config['save_dir']) / "CarRacing-v2" / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environments
    env = create_env(config['seed'])
    eval_env = create_env(config['seed'] + 100)
    
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: CarRacing-v2")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    print(f"Models will be saved to: {save_dir}")
    print(f"Total episodes: {config['max_episodes']}")
    print("-" * 70)
    
    # Initialize SAC agent
    agent = SAC_CarRacing(
        action_dim=action_dim,
        input_channels=3,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        automatic_entropy_tuning=config['automatic_entropy_tuning'],
        device=device
    )
    
    # Initialize wandb (optional)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'RL-Assignment4-SAC-CarRacing'),
                entity=config.get('wandb_entity', None),
                config=config,
                name=f"SAC-CarRacing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                mode=config.get('wandb_mode', 'online')
            )
            wandb_available = True
        except ImportError:
            print("WandB not available, continuing without logging")
            wandb_available = False
    else:
        wandb_available = False
    
    # Training variables
    total_steps = 0
    best_eval_reward = -float('inf')
    episode_rewards = []
    
    print("\nStarting training...")
    print("=" * 70)
    
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
                losses = {'critic': [], 'actor': [], 'alpha': []}
                
                for _ in range(config['num_updates']):
                    critic_loss, actor_loss, alpha_loss, alpha = agent.update(config['batch_size'])
                    
                    if critic_loss is not None:
                        losses['critic'].append(critic_loss)
                        losses['actor'].append(actor_loss)
                        if alpha_loss is not None:
                            losses['alpha'].append(alpha_loss)
                
                # Log average losses
                if wandb_available and losses['critic']:
                    wandb.log({
                        'train/critic_loss': np.mean(losses['critic']),
                        'train/actor_loss': np.mean(losses['actor']),
                        'train/alpha_loss': np.mean(losses['alpha']) if losses['alpha'] else 0,
                        'train/alpha': alpha,
                        'train/total_steps': total_steps,
                        'train/buffer_size': len(agent.replay_buffer)
                    })
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % config['log_frequency'] == 0:
            avg_reward = np.mean(episode_rewards[-config['log_frequency']:]) if len(episode_rewards) >= config['log_frequency'] else np.mean(episode_rewards)
            print(f"Episode {episode:4d}/{config['max_episodes']} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Total Steps: {total_steps:7d} | "
                  f"Buffer: {len(agent.replay_buffer)}")
        
        if wandb_available:
            wandb.log({
                'train/episode_reward': episode_reward,
                'train/episode_steps': episode_steps,
                'train/episode': episode,
                'train/avg_reward_last_10': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            })
        
        # Evaluation
        if episode > 0 and episode % config['eval_frequency'] == 0:
            print(f"\n{'='*70}")
            print(f"Evaluating at episode {episode}...")
            eval_reward, eval_std = evaluate(agent, eval_env, config['num_eval_episodes'])
            print(f"Evaluation Result: {eval_reward:.2f} ± {eval_std:.2f}")
            print(f"{'='*70}\n")
            
            if wandb_available:
                wandb.log({
                    'eval/mean_reward': eval_reward,
                    'eval/std_reward': eval_std,
                    'eval/episode': episode
                })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(save_dir / 'best_model.pt')
                print(f"✓ New best model saved! Reward: {eval_reward:.2f}\n")
            
            # Check if solved (CarRacing is considered solved at 900+)
            if eval_reward >= config['target_reward']:
                print(f"\n{'='*70}")
                print(f"🎉 Environment solved! Target reward {config['target_reward']} reached!")
                print(f"{'='*70}\n")
                agent.save(save_dir / 'solved_model.pt')
                if config.get('stop_on_solve', True):
                    break
        
        # Save checkpoint
        if episode > 0 and episode % config['save_frequency'] == 0:
            agent.save(save_dir / f'checkpoint_ep{episode}.pt')
            print(f"✓ Checkpoint saved at episode {episode}")
    
    # Final save
    agent.save(save_dir / 'final_model.pt')
    
    # Save training summary
    summary_file = save_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total episodes: {config['max_episodes']}\n")
        f.write(f"Total steps: {total_steps}\n")
        f.write(f"Best evaluation reward: {best_eval_reward:.2f}\n")
        f.write(f"Final buffer size: {len(agent.replay_buffer)}\n")
        f.write(f"Models saved to: {save_dir}\n")
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Total steps: {total_steps}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*70}")
    
    # List all saved models
    print("\nSaved models:")
    for model_file in save_dir.glob("*.pt"):
        file_size = model_file.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"  - {model_file.name} ({file_size:.2f} MB)")
    
    env.close()
    eval_env.close()
    
    if wandb_available:
        wandb.finish()


def run_training():
    """Run training with default parameters (for Kaggle/Colab)"""
    
    # Configuration for CarRacing
    config = {
        # Environment
        "max_episodes": 500,  # Reduced for faster training
        "max_steps": 1000,
        
        # SAC hyperparameters
        "hidden_dim": 256,
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.1,
        "automatic_entropy_tuning": True,
        
        # Training parameters
        "batch_size": 128,
        "start_steps": 5000,
        "update_after": 1000,
        "update_every": 50,
        "num_updates": 50,
        
        # Evaluation
        "eval_frequency": 20,
        "num_eval_episodes": 3,
        
        # Logging
        "log_frequency": 10,
        "save_frequency": 50,
        
        # Target
        "target_reward": 900,
        "stop_on_solve": False,  # Continue training even after solving
        
        # System
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "/kaggle/working/models" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else "models/sac",
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Train without wandb by default
    train_sac_carracing(config, use_wandb=False)


def test_model(model_path=None):
    """Test a trained model"""
    import matplotlib.pyplot as plt
    
    if model_path is None:
        # Look for the latest model
        model_dir = Path("/kaggle/working/models" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else "models/sac")
        model_files = list(model_dir.glob("*.pt"))
        if not model_files:
            print("No models found to test")
            return
        model_path = model_files[-1]  # Get the most recent model
    
    print(f"Testing model: {model_path}")
    
    # Create environment
    env = gym.make("CarRacing-v2", continuous=True, render_mode='human')
    
    # Initialize agent
    action_dim = env.action_space.shape[0]
    agent = SAC_CarRacing(
        action_dim=action_dim,
        input_channels=3,
        hidden_dim=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load model
    agent.load(model_path)
    
    # Test for a few episodes
    num_test_episodes = 3
    rewards = []
    
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 1000:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        print(f"Test episode {episode+1}/{num_test_episodes}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\nTest completed!")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    # Check if we're in a Jupyter environment
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except:
        in_notebook = False
    
    if in_notebook:
        # Running in notebook, use simple function call
        print("Running in Jupyter notebook mode...")
        run_training()
    else:
        # Running as script, use argparse
        parser = argparse.ArgumentParser(description='Train or test SAC on CarRacing-v2')
        parser.add_argument('--mode', type=str, default='train',
                          choices=['train', 'test'],
                          help='Mode: train or test')
        parser.add_argument('--model-path', type=str, default=None,
                          help='Path to model for testing')
        parser.add_argument('--no-wandb', action='store_true',
                          help='Disable Weights & Biases logging')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        parser.add_argument('--episodes', type=int, default=500,
                          help='Maximum number of episodes')
        parser.add_argument('--device', type=str, default=None,
                          choices=['cuda', 'cpu'],
                          help='Device to use')
        
        args = parser.parse_args()
        
        if args.mode == 'train':
            config = {
                "max_episodes": args.episodes,
                "max_steps": 1000,
                "hidden_dim": 256,
                "lr": 1e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.1,
                "automatic_entropy_tuning": True,
                "batch_size": 128,
                "start_steps": 5000,
                "update_after": 1000,
                "update_every": 50,
                "num_updates": 50,
                "eval_frequency": 20,
                "num_eval_episodes": 3,
                "log_frequency": 10,
                "save_frequency": 50,
                "target_reward": 900,
                "stop_on_solve": False,
                "seed": args.seed,
                "device": args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
                "save_dir": "/kaggle/working/models" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else "models/sac",
            }
            
            print(f"Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
            
            train_sac_carracing(config, use_wandb=not args.no_wandb)
        
        elif args.mode == 'test':
            test_model(args.model_path)