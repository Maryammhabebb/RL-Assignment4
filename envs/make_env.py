# envs/make_env.py

import gymnasium as gym
from envs.wrappers_carracing import CarRacingWrapper

def make_env(env_name):
    """
    Returns the environment, automatically applying
    wrappers when needed.
    """

    if env_name.lower() == "lunarlander":
        env = gym.make("LunarLander-v3", continuous=True)
        return env

    elif env_name.lower() == "carracing":
        env = CarRacingWrapper()
        return env

    else:
        raise ValueError(f"Unknown environment: {env_name}")
