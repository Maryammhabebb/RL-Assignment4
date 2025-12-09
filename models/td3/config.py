# models/td3/config.py

TD3_CONFIG = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,                  # soft update coefficient
    "policy_noise": 0.2,           # noise added to target policy
    "noise_clip": 0.5,             # clip range for noise
    "policy_freq": 2,              # delay actor updates
    "buffer_size": int(1e6),
    "batch_size": 256,
}
