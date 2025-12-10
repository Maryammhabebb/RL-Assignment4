"""
Configuration file for SAC training on different environments
"""

SAC_CONFIGS = {
    "LunarLanderContinuous-v3": {
        # Environment settings
        "env_name": "LunarLanderContinuous-v3",
        "max_episodes": 600,
        "max_steps": 600,
        
        # SAC hyperparameters
        "hidden_dim": 128,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "automatic_entropy_tuning": True,
        
        # Training parameters
        "batch_size": 128,
        "buffer_size": 10000,
        "start_steps": 10000,  # Random exploration steps
        "update_after": 1000,  # Start training after this many steps
        "update_every": 50,    # Update frequency
        "num_updates": 10,     # Number of updates per update step
        
        # Evaluation
        "eval_frequency": 50,  # Evaluate every N episodes
        "num_eval_episodes": 3,
        
        # Logging
        "log_frequency": 10,   # Log every N episodes
        "save_frequency": 100,  # Save model every N episodes
        
        # Early stopping
        "target_reward": 200,  # Stop if this reward is reached
        "solved_episodes": 5, # Number of consecutive episodes above target
    },
    
    "CarRacing-v3": {
        # Environment settings
        "env_name": "CarRacing-v3",
        "max_episodes": 2000,
        "max_steps": 1000,
        
        # SAC hyperparameters
        "hidden_dim": 512,
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.1,
        "automatic_entropy_tuning": True,
        
        # Training parameters
        "batch_size": 128,
        "buffer_size": 500000,
        "start_steps": 5000,
        "update_after": 1000,
        "update_every": 50,
        "num_updates": 50,
        
        # Evaluation
        "eval_frequency": 20,
        "num_eval_episodes": 3,
        
        # Logging
        "log_frequency": 10,
        "save_frequency": 100,
        
        # Early stopping
        "target_reward": 900,
        "solved_episodes": 5,
        
        # Image preprocessing (if using raw pixels)
        "use_grayscale": True,
        "frame_stack": 4,
        "image_size": (84, 84),
    },
}

# WandB configuration
WANDB_CONFIG = {
    "project": "RL-Assignment4-SAC",
    "entity": None,  # Set to your wandb username
    "mode": "online",  # "online", "offline", or "disabled"
}

# General settings
GENERAL_CONFIG = {
    "seed": 42,
    "device": "cpu",  # "cuda" or "cpu"
    "save_dir": "trained_models/sac",
    "log_dir": "logs/sac",
}