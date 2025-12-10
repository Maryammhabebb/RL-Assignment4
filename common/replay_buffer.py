import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size: int, state_dim, action_dim: int, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # Handle both int and tuple for state dimension
        if isinstance(state_dim, int):
            state_shape = (state_dim,)
        else:
            state_shape = tuple(state_dim)

        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # Convert direct to DEVICE (cpu/mps/cuda)
        state = torch.as_tensor(self.state[idxs], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(self.action[idxs], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(self.reward[idxs], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(self.next_state[idxs], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(self.done[idxs], dtype=torch.float32, device=self.device)

        return state, action, reward, next_state, done
