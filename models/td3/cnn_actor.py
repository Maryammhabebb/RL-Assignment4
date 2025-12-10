

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNActor(nn.Module):
    """
    Light CNN encoder + MLP head for continuous actions.
    Input:  (B, C, H, W)
    Output: (B, action_dim) in [-max_action, max_action]
    """

    def __init__(self, state_shape, action_dim, max_action):
        super().__init__()

        c, h, w = state_shape
        self.max_action = max_action

        # Simple convolutional encoder (inspired by Atari-style networks)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        # Squash to [-max_action, max_action]
        x = torch.tanh(x) * self.max_action
        return x
