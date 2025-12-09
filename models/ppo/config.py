# models/ppo/config.py

# Stable hyperparameters for CONTINUOUS LunarLander-v3
# Tuned for reliable convergence
PPO_CONFIG = {
    "actor_lr": 5e-4,              # Slightly higher for faster learning
    "critic_lr": 5e-4,             # Match actor for stability
    "gamma": 0.999,                # Long-term planning for landing
    "gae_lambda": 0.98,            # Better advantage estimation
    "clip_epsilon": 0.2,           # Standard PPO clipping
    "epochs": 4,                   # Fewer epochs to prevent overfitting
    "batch_size": 64,              
    "buffer_size": 2048,           # Good balance for continuous control
    "entropy_coef": 0.0,           # No entropy bonus - let tanh handle exploration
    "value_coef": 0.5,             
    "max_grad_norm": 0.5,          
}
