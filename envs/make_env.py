# envs/make_env.py

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation


class SimplifyCarRacing(gym.ObservationWrapper):
    def __init__(self, env, size=(32, 32)):
        super().__init__(env)
        self.size = size
        # New observation space: (3, 32, 32) normalized to [0,1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, size[0], size[1]), dtype=np.float32
        )

    def observation(self, obs):
        import cv2
        resized = cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        # Transpose to (3,32,32) and normalize
        return resized.transpose(2,0,1).astype(np.float32) / 255.0


def make_env(env_name, render_mode=None):
    """
    Creates the correct continuous-action environment for TD3/PPO/SAC.
    Supports: lunarlander, carracing
    """

    env_name = env_name.lower()

    # ----------------------------
    # LunarLander Continuous
    # ----------------------------
    if env_name == "lunarlander":
        env_id = "LunarLander-v3"
        kwargs = {"continuous": True}

    # ----------------------------
    # CarRacing Environment
    # ----------------------------
    elif env_name == "carracing":
        env_id = "CarRacing-v3"
        kwargs = {"continuous": True}
        
        # Add render mode only when needed (e.g., for eval)
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        
        # Create base environment
        env = gym.make(env_id, **kwargs)
        
        # Downsample RGB: 96x96x3 -> 32x32x3 -> flatten = 3072 dims
        env = SimplifyCarRacing(env, size=(32, 32))
        
        return env
    
    else:
        raise ValueError(f"❌ Unknown environment: {env_name}")

    # Add render mode only when needed (e.g., for eval)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    return gym.make(env_id, **kwargs)
