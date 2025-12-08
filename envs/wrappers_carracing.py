# envs/wrappers_carracing.py

import gymnasium as gym
import numpy as np

class CarRacingWrapper:
    """
    Simple wrapper for CarRacing-v3 to expose a clean continuous environment
    for TD3 (steering, gas, brake).
    """

    def __init__(self):
        self.env = gym.make("CarRacing-v3", continuous=True)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        action = np.clip(action, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
