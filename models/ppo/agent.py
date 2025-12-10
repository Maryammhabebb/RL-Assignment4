# models/ppo/agent.py

import torch
import torch.nn as nn
import numpy as np

from models.ppo.actor import Actor
from models.ppo.critic import Critic
from models.ppo.config import PPO_CONFIG


class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, device="cpu"):
        self.device = device

        # Hyperparameters
        self.gamma = PPO_CONFIG["gamma"]
        self.gae_lambda = PPO_CONFIG["gae_lambda"]
        self.clip_epsilon = PPO_CONFIG["clip_epsilon"]
        self.epochs = PPO_CONFIG["epochs"]
        self.batch_size = PPO_CONFIG["batch_size"]
        self.buffer_size = PPO_CONFIG["buffer_size"]
        self.entropy_coef = PPO_CONFIG["entropy_coef"]
        self.value_coef = PPO_CONFIG["value_coef"]
        self.max_grad_norm = PPO_CONFIG["max_grad_norm"]
        self.actor_lr = PPO_CONFIG["actor_lr"]
        self.critic_lr = PPO_CONFIG["critic_lr"]

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def act(self, state, deterministic=False):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = mean
            else:
                action, log_prob, _ = self.actor.sample(state_tensor)
                value = self.critic(state_tensor)
                
                # Store tensors (not detached yet - keep on device)
                self.states.append(state_tensor)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.values.append(value)
            
            action_np = action.squeeze(0).cpu().numpy()

        return action_np

    def store_transition(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_state):
        """Compute GAE advantages"""
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).item()

        # Convert to numpy for GAE computation
        values = torch.cat(self.values).squeeze().cpu().numpy()
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute GAE
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def train(self, next_state):
        """Train using collected rollout"""
        
        # Must have enough samples AND buffer should be "full"
        if len(self.states) < self.batch_size:
            # Return dummy values to prevent crashes
            return 0.0, 0.0, 0.0

        # Compute advantages
        advantages, returns = self.compute_gae(next_state)

        # Convert lists to tensors
        states = torch.cat(self.states).to(self.device)  # Already tensors
        actions = torch.cat(self.actions).to(self.device)
        old_log_probs = torch.cat(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track losses for logging
        actor_losses = []
        critic_losses = []
        entropies = []

        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Shuffle indices for mini-batch updates
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # -----------------------------------------
                # Actor loss (PPO clipped objective)
                # -----------------------------------------
                new_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                entropy = (0.5 * (1 + torch.log(2 * np.pi * std.pow(2)))).mean()
                
                actor_loss_total = actor_loss - self.entropy_coef * entropy

                # -----------------------------------------
                # Critic loss (value function)
                # -----------------------------------------
                values_pred = self.critic(batch_states)
                critic_loss = nn.MSELoss()(values_pred, batch_returns)

                # -----------------------------------------
                # Update networks
                # -----------------------------------------
                self.actor_optimizer.zero_grad()
                actor_loss_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Track for logging
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())

        # Clear buffer after training
        self.clear_buffer()
        
        # Return average losses
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)

    def clear_buffer(self):
        """Clear rollout buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []