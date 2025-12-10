import numpy as np
import torch
import torch.nn as nn

from models.td3.cnn_actor import CNNActor
from models.td3.cnn_critic import CNNCritic
from models.td3.config import TD3_CONFIG


class TD3CarRacingAgent:
    """
    TD3 with CNN actor+critic for image-based control.
    """

    def __init__(self, state_shape, action_dim, max_action, device="cpu"):
        self.device = device
        self.max_action = max_action

        # Hyperparameters
        self.gamma = TD3_CONFIG["gamma"]
        self.tau = TD3_CONFIG["tau"]
        self.policy_noise = TD3_CONFIG["policy_noise"]
        self.noise_clip = TD3_CONFIG["noise_clip"]
        self.policy_freq = TD3_CONFIG["policy_freq"]
        self.batch_size = TD3_CONFIG["batch_size"]

        # Networks
        self.actor = CNNActor(state_shape, action_dim, max_action).to(self.device)
        self.actor_target = CNNActor(state_shape, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CNNCritic(state_shape, action_dim).to(self.device)
        self.critic_target = CNNCritic(state_shape, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=TD3_CONFIG["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=TD3_CONFIG["critic_lr"])

        self.total_it = 0

    # -------------------------- act() --------------------------
    def act(self, state, noise=0.1):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy()[0]

        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)

        return np.clip(action, -self.max_action, self.max_action)

    # -------------------------- train() ------------------------
    def train(self, replay_buffer):
        if replay_buffer.size < self.batch_size:
            return None

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

        # Add policy noise
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.gamma * target_Q

        # Compute critic loss
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q.detach()) + \
                      nn.MSELoss()(current_Q2, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss else None,
            "Q_value_mean": current_Q1.mean().item()
        }

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
