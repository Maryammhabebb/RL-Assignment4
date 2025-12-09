# envs/make_env.py

import gymnasium as gym


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
    
    else:
        raise ValueError(f"❌ Unknown environment: {env_name}")

    # Add render mode only when needed (e.g., for eval)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    return gym.make(env_id, **kwargs)
