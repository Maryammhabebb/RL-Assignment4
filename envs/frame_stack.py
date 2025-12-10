import numpy as np
import gymnasium as gym

class FrameStack(gym.Wrapper):
    """
    Minimal FrameStack compatible with Gymnasium 1.2.x.
    Supports:
        - env.reset(seed=..., options=...)
        - stacking frames along channel dimension
        - image observations (H, W, C)
    """

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack

        H, W, C = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(H, W, C * num_stack),
            dtype=np.uint8
        )

        # Buffer for stacked frames
        self.frames = np.zeros((H, W, C * num_stack), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Stack the first obs num_stack times
        stacked = [obs] * self.num_stack
        self.frames = np.concatenate(stacked, axis=2)

        return self.frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Drop oldest frame and append new one
        C = obs.shape[2]
        self.frames = np.concatenate(
            (self.frames[:, :, C:], obs),
            axis=2
        )

        return self.frames, reward, terminated, truncated, info
