import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        q = nn.ReLU()(self.l1(sa))
        q = nn.ReLU()(self.l2(q))
        q = self.l3(q)
        
        return q