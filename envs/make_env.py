# envs/make_env.py

import gymnasium as gym
from envs.wrappers_carracing import make_carracing_env


def make_env(env_name: str, render_mode=None):
    """
    Factory for environments used in the assignment.

    Supported values for env_name (case-insensitive):

      - "lunarlander"  -> LunarLanderContinuous-v3
      - "carracing"    -> CarRacing-v3 (continuous, with preprocessing)

    `render_mode`:
      - None          -> no rendering (training)
      - "rgb_array"   -> for RecordVideo, evaluation
    """
    name = env_name.lower()

    if name in ["lunarlander", "lunarlandercontinuous-v3"]:
        # LunarLander continuous
        return gym.make(
            "LunarLander-v3",
            continuous=True,
            render_mode=render_mode
        )

    if name in ["carracing", "carracing-v3"]:
        # Preprocessed CarRacing (grayscale, resized, stacked frames)
        return make_carracing_env(render_mode=render_mode)

    # Fallback: try to make it directly if a full Gymnasium ID is passed
    return gym.make(env_name, render_mode=render_mode)
