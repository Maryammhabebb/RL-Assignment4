# algorithms/td3/agent.py

import torch
import torch.nn as nn
import numpy as np

from algorithms.td3.actor import Actor
from algorithms.td3.critic import Critic
from algorithms.td3.config import TD3_CONFIG


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device="cpu"):
        self.device = device

        # Hyperparameters
        self.gamma = TD3_CONFIG["gamma"]
        self.tau = TD3_CONFIG["tau"]
        self.policy_noise = TD3_CONFIG["policy_noise"]
        self.noise_clip = TD3_CONFIG["noise_clip"]
        self.policy_freq = TD3_CONFIG["policy_freq"]
        self.batch_size = TD3_CONFIG["batch_size"]

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks (2 critics for TD3)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=TD3_CONFIG["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=TD3_CONFIG["critic_lr"])

        # Internal counter
        self.total_it = 0

    # -----------------------------------------------------
    # Select action (with or without noise)
    # -----------------------------------------------------
    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)

        action = self.actor(state).cpu().data.numpy()

        # Add exploration noise
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)

        # Clamp to valid range
        return np.clip(action, -1, 1)

    # -----------------------------------------------------
    # Train from Replay Buffer
    # -----------------------------------------------------
    def train(self, replay_buffer):
        if replay_buffer.size < self.batch_size:
            return  # Not enough samples yet

        self.total_it += 1

        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # -----------------------------------------
        # 1. Compute target action (policy smoothing)
        # -----------------------------------------
        noise = (
            torch.randn_like(action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        # -----------------------------------------
        # 2. Compute target Q using target networks
        # -----------------------------------------
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target = reward + (1 - done) * self.gamma * target_Q

        # -----------------------------------------
        # 3. Update Critics
        # -----------------------------------------
        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = (
            nn.MSELoss()(current_Q1, target.detach()) +
            nn.MSELoss()(current_Q2, target.detach())
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -----------------------------------------
        # 4. Delayed Actor Update
        # -----------------------------------------
        if self.total_it % self.policy_freq == 0:

            # Maximize Q → minimize negative Q
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # -------------------------------------
            # 5. Soft update target networks
            # -------------------------------------
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

    # -----------------------------------------------------
    # Soft-update helper
    # -----------------------------------------------------
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
