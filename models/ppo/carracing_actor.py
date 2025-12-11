# models/ppo/carracing_actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CarRacingActor(nn.Module):
    """CNN-based actor for CarRacing image observations"""
    
    def __init__(self, action_dim, max_action):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Regularization to prevent overfitting
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Regularization
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Regularization
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
        )
        
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize mean output layer with smaller weights
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        
        # Initialize log_std to -0.5 (std = 0.6) for balanced exploration
        nn.init.constant_(self.log_std.weight, 0)
        nn.init.constant_(self.log_std.bias, -0.5)
    
    def forward(self, state):
        """
        Forward pass to get mean and log_std for Gaussian policy.
        state: (batch_size, 3, 32, 32) for CarRacing
        """
        features = self.cnn(state)
        mean = self.mean(features)
        mean = torch.tanh(mean) * 0.9  # Slight squashing for stability
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -1, 0.5)  # Std between 0.37 and 1.65
        return mean, log_std

    def sample(self, state):
        """
        Sample action from Gaussian policy using reparameterization trick.
        Returns: action, log_prob, mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from N(mean, std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(z) * self.max_action
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) / (self.max_action ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean

    def get_log_prob(self, state, action):
        """
        Compute log probability of given action under current policy.
        Used during PPO updates.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Inverse tanh to get z
        z = torch.atanh(action / self.max_action)
        
        # Log probability from Gaussian
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) / (self.max_action ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob, log_std.exp().mean()
