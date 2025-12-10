

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCritic(nn.Module):
    """
    Two Q-networks (Q1, Q2), both using a shared CNN encoder for the state,
    but with separate MLP heads for Q1 and Q2.
    """

    def __init__(self, state_shape, action_dim):
        super().__init__()

        c, h, w = state_shape

        # Shared CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.view(1, -1).shape[1]

        # Q1 head
        self.q1_fc = nn.Sequential(
            nn.Linear(conv_flat + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Q2 head
        self.q2_fc = nn.Sequential(
            nn.Linear(conv_flat + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """
        state:  (B, C, H, W)
        action: (B, A)
        """
        feat = self.conv(state)
        feat = feat.view(feat.size(0), -1)

        sa = torch.cat([feat, action], dim=-1)

        q1 = self.q1_fc(sa)
        q2 = self.q2_fc(sa)
        return q1, q2

    def q1(self, state, action):
        feat = self.conv(state)
        feat = feat.view(feat.size(0), -1)
        sa = torch.cat([feat, action], dim=-1)
        q1 = self.q1_fc(sa)
        return q1
