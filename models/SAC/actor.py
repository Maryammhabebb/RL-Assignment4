import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = nn.ReLU()(self.l1(state))
        a = nn.ReLU()(self.l2(a))
        
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, -10, 1)
        
        return mean, log_std