import numpy as np
import gymnasium as gym
import cv2
from envs.frame_stack import FrameStack   # custom stacker


def make_carracing_env(render_mode=None, num_stack=4, resize_shape=84):

    # Base environment
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
    )

    # ---------------------------------------
    # 1) Grayscale wrapper
    # ---------------------------------------
    class Gray(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            h, w, _ = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(h, w, 1), dtype=np.uint8
            )

        def observation(self, obs):
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            return np.expand_dims(gray, axis=2)  # (H, W, 1)

    # ---------------------------------------
    # 2) Resize wrapper
    # ---------------------------------------
    class Resize(gym.ObservationWrapper):
        def __init__(self, env, size):
            super().__init__(env)
            self.size = size
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(size, size, 1), dtype=np.uint8
            )

        def observation(self, obs):
            resized = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized, axis=2)

    # ---------------------------------------
    # 3) Normalize + CHW wrapper
    # ---------------------------------------
    class NormalizeCHW(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            h, w, c = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, shape=(c, h, w), dtype=np.float32
            )

        def observation(self, obs):
            obs = obs.astype(np.float32) / 255.0
            return np.transpose(obs, (2, 0, 1))  # (C, H, W)

    # ---------------------------------------
    # Apply wrappers in correct order
    # ---------------------------------------
    env = Gray(env)
    env = Resize(env, resize_shape)
    env = FrameStack(env, num_stack)   # our custom fixed stacker
    env = NormalizeCHW(env)

    return env
