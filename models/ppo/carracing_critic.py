# models/ppo/carracing_critic.py

import torch
import torch.nn as nn
import numpy as np


class CarRacingCritic(nn.Module):
    """CNN-based critic for CarRacing image observations"""
    
    def __init__(self):
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
            nn.Linear(256, 1)
        )
        
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

    def forward(self, state):
        """
        Compute state value V(s).
        state: (batch_size, 3, 32, 32) for CarRacing
        Returns: (batch_size, 1) value estimates
        """
        return self.cnn(state)
