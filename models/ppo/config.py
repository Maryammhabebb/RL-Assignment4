# models/ppo/config.py

# Stable hyperparameters for CONTINUOUS LunarLander-v3
# Tuned for reliable convergence
PPO_CONFIG_LUNARLANDER = {
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
PPO_CONFIG_CARRACING = {
    "actor_lr": 5e-5,          # Lower LR to prevent rapid overfitting
    "critic_lr": 1e-4,         # Lower critic LR as well
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "epochs": 2,               # Fewer epochs to reduce overfitting
    "batch_size": 256,         # Larger batches for better generalization
    "buffer_size": 2048,
    "entropy_coef": 0.05,      # Higher entropy to maintain exploration
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "weight_decay": 1e-4,      # L2 Regularization to prevent overfitting
    # Default names and metadata for experiments (can be overridden)
    "env_name": "CarRacing-v3",
    "experiment_name": "ppo_carracing_v3",
    "use_wandb": True,
    "wandb_project": "ppo-carracing-v3",
    "wandb_run_name": "ppo_carracing_v3_default",
}