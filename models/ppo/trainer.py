# models/ppo/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

try:
    import wandb as _wandb
except Exception:
    _wandb = None

class PPOTrainer:
    """Trainer for PPO algorithm compatible with CNN-based CarRacing networks"""
    
    def __init__(self, actor, critic, config, device):
        self.actor = actor
        self.critic = critic
        self.device = device
        
        # Hyperparameters from config
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_epsilon = config["clip_epsilon"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]
        self.entropy_coef = config["entropy_coef"]
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        # Logging / wandb
        self.use_wandb = config.get("use_wandb", False)
        self.wandb = _wandb if (self.use_wandb and _wandb is not None) else None
        self.update_count = 0
        # Keep recent log history for aggregation
        self.last_log = None
        self.log_history = deque(maxlen=200)
        
        # Optimizers with numerical stability and weight decay
        weight_decay = config.get("weight_decay", 0.0)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"], eps=1e-5, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], eps=1e-5, weight_decay=weight_decay)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self):
        """Compute Generalized Advantage Estimation"""
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute next value for terminal states
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self):
        """Perform PPO update"""
        if len(self.states) < self.batch_size:
            return
        
        # Compute advantages
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for epoch in range(self.epochs):
            # Shuffle indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Actor loss
                new_log_probs, entropy = self.actor.get_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # Critic loss
                values_pred = self.critic(batch_states)
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                # Check for NaN before backward
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    print(f"Warning: NaN detected in losses. Skipping update.")
                    continue
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Diagnostics and logging
                try:
                    entropy_mean = float(entropy.mean().item()) if hasattr(entropy, 'mean') else float(entropy)
                except Exception:
                    entropy_mean = float(entropy) if isinstance(entropy, (float, int)) else 0.0

                # Clip fraction and approx KL
                try:
                    clip_frac = float(((ratio > 1 + self.clip_epsilon) | (ratio < 1 - self.clip_epsilon)).float().mean().item())
                except Exception:
                    clip_frac = 0.0

                try:
                    approx_kl = float((batch_old_log_probs - new_log_probs).mean().item())
                except Exception:
                    approx_kl = 0.0

                # Learning rates
                actor_lr = float(self.actor_optimizer.param_groups[0].get('lr', 0.0))
                critic_lr = float(self.critic_optimizer.param_groups[0].get('lr', 0.0))

                # Increment update counter and log
                self.update_count += 1

                log_data = {
                    'update': int(self.update_count),
                    'actor_loss': float(actor_loss.item()),
                    'critic_loss': float(critic_loss.item()),
                    'entropy': entropy_mean,
                    'actor_grad_norm': float(actor_grad_norm),
                    'critic_grad_norm': float(critic_grad_norm),
                    'clip_fraction': clip_frac,
                    'approx_kl': approx_kl,
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                }

                # store last log and history for external inspection
                try:
                    self.last_log = log_data
                    self.log_history.append(log_data)
                except Exception:
                    pass

                # Console summary every 50 updates
                if self.update_count % 50 == 0:
                    print(f"[PPOTrainer] Update {self.update_count}: actor_loss={log_data['actor_loss']:.4f}, critic_loss={log_data['critic_loss']:.4f}, "
                          f"entropy={log_data['entropy']:.4f}, a_grad={log_data['actor_grad_norm']:.3f}, c_grad={log_data['critic_grad_norm']:.3f}, "
                          f"clip_frac={log_data['clip_fraction']:.3f}, approx_kl={log_data['approx_kl']:.5f}")

                # Send to wandb if available
                if self.wandb is not None:
                    try:
                        self.wandb.log(log_data)
                    except Exception:
                        pass
        
    def get_recent_log_stats(self, n: int = 50):
        """Return averaged stats over the last `n` update logs (or fewer if not available)."""
        if len(self.log_history) == 0:
            return None
        # take up to n most recent entries
        items = list(self.log_history)[-n:]
        agg = {}
        keys = items[0].keys()
        for k in keys:
            vals = [it.get(k, 0.0) for it in items]
            try:
                agg[k] = float(np.mean(vals))
            except Exception:
                agg[k] = vals[-1]
        return agg
        
        # Clear buffer
        self.clear_buffer()
    
    def clear_buffer(self):
        """Clear rollout buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []