# train_carracing_ppo.py

import torch
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.make_env import make_env
from models.ppo.carracing_actor import CarRacingActor
from models.ppo.carracing_critic import CarRacingCritic
from models.ppo.trainer import PPOTrainer
from models.ppo.config import PPO_CONFIG_CARRACING as CAR_RACING_CONFIG
import warnings
warnings.filterwarnings('ignore')
import argparse

# Add max_episodes and save_interval to config
CAR_RACING_CONFIG["max_episodes"] = 10000
CAR_RACING_CONFIG["save_interval"] = 50


def _save_state_dicts_with_meta(prefix_path, actor, critic, meta: dict):
        """Save actor/critic state_dicts and metadata JSON alongside a checkpoint prefix.

        Example: prefix_path='saved_models/ppo/carracing_best' will create
            saved_models/ppo/carracing_best.actor_state.pth
            saved_models/ppo/carracing_best.critic_state.pth
            saved_models/ppo/carracing_best.meta.json
        """
        import json
        actor_path = f"{prefix_path}.actor_state.pth"
        critic_path = f"{prefix_path}.critic_state.pth"
        meta_path = f"{prefix_path}.meta.json"
        try:
                torch.save(actor.state_dict(), actor_path)
                torch.save(critic.state_dict(), critic_path)
                with open(meta_path, "w") as f:
                        json.dump(meta, f)
        except Exception as e:
                print(f"Warning: failed to save state dicts/meta: {e}")


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment
    env = make_env("carracing")

    # Optional Weights & Biases logging
    import os
    use_wandb = CAR_RACING_CONFIG.get("use_wandb", False)
    # Auto-enable if user provided WANDB_API_KEY environment variable
    if not use_wandb and os.environ.get("WANDB_API_KEY"):
        use_wandb = True
        CAR_RACING_CONFIG["use_wandb"] = True

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.init(project=CAR_RACING_CONFIG.get("wandb_project", "ppo-carracing"),
                        config=CAR_RACING_CONFIG,
                        name=CAR_RACING_CONFIG.get("wandb_run_name", None))
            wandb = _wandb
            print("WandB logging enabled.")
        except Exception as e:
            print(f"WandB init failed (continuing without wandb): {e}")
            wandb = None
            use_wandb = False
            CAR_RACING_CONFIG["use_wandb"] = False
    
    # Get dimensions
    state_shape = env.observation_space.shape  # Should be (3, 32, 32)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"State shape: {state_shape}, Action dim: {action_dim}, Max action: {max_action}")
    
    # Initialize networks
    actor = CarRacingActor(action_dim, max_action).to(device)
    critic = CarRacingCritic().to(device)
    
    # Initialize trainer
    trainer = PPOTrainer(actor, critic, CAR_RACING_CONFIG, device)
    # Ensure trainer uses same wandb run (if enabled)
    try:
        trainer.wandb = wandb
    except Exception:
        pass
    
    # Training loop
    total_steps = 0
    best_reward = -float('inf')
    episode_rewards = []
    running_reward = None
    
    # Validation tracking to prevent overfitting
    best_val_reward = -float('inf')
    best_val_episode = 0
    val_rewards_history = []
    
    # Early stopping to prevent overfitting
    best_avg_reward = -float('inf')
    patience = 50  # Stop if no improvement for 50 episodes
    episodes_without_improvement = 0
    val_patience = 3  # Stop if validation degrades 3 times in a row
    val_degradation_count = 0

    # Step-based logging/validation settings
    log_interval_steps = CAR_RACING_CONFIG.get("log_interval_steps", 25000)
    next_log_step = log_interval_steps
    val_episodes = CAR_RACING_CONFIG.get("val_episodes", 10)

    def do_validation(trigger_label):
        nonlocal best_val_reward, best_val_episode, val_degradation_count
        actor.eval()
        val_rewards = []
        print(f"\n🔍 Validation ({trigger_label}) on {val_episodes} fresh tracks...")
        for _ in range(val_episodes):
            val_env = make_env("carracing")
            val_state, _ = val_env.reset()
            val_reward = 0
            val_done = False
            val_steps = 0
            max_steps = 1000

            while not val_done and val_steps < max_steps:
                val_state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    val_action, _, _ = actor.sample(val_state_tensor)
                val_action_np = val_action.cpu().numpy()[0]
                val_action_np = np.clip(val_action_np, -1.0, 1.0)
                val_state, val_r, val_term, val_trunc, _ = val_env.step(val_action_np)
                val_reward += np.clip(val_r, -10, 10)
                val_done = val_term or val_trunc
                val_steps += 1

            val_rewards.append(val_reward)
            val_env.close()

        val_avg = np.mean(val_rewards)
        val_std = np.std(val_rewards)
        val_min = min(val_rewards)
        val_max = max(val_rewards)
        val_rewards_history.append(val_avg)

        print(f"✅ Validation: Avg={val_avg:.2f}±{val_std:.2f}, Min={val_min:.2f}, Max={val_max:.2f}")
        print(f"   Training running avg: {running_reward:.2f}, Train/Val Gap={running_reward - val_avg:.2f}\n")
        actor.train()

        # WandB logging
        if wandb:
            try:
                wandb.log({
                    "val_avg": float(val_avg),
                    "val_std": float(val_std),
                    "val_min": float(val_min),
                    "val_max": float(val_max),
                    "train_running_reward": float(running_reward),
                    "train_val_gap": float(running_reward - val_avg),
                    "total_steps": int(total_steps),
                    "trigger": trigger_label,
                })
            except Exception:
                pass

        # Save best validation model
        if val_avg > best_val_reward:
            best_val_reward = val_avg
            best_val_episode = episode
            val_degradation_count = 0
            os.makedirs("saved_models/ppo", exist_ok=True)
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'config': CAR_RACING_CONFIG,
                'episode': episode,
                'val_reward': val_avg,
                'train_reward': running_reward
            }, "saved_models/ppo/carracing_best_val.pth")
            print(f"🌟 New BEST VALIDATION model saved! Val={val_avg:.2f}\n")
            if wandb:
                try:
                    wandb.save("saved_models/ppo/carracing_best_val.pth")
                    wandb.log({"best_val_reward": float(best_val_reward), "best_val_episode": int(best_val_episode)})
                except Exception:
                    pass
        else:
            val_degradation_count += 1
            print(f"⚠️  Validation not improving ({val_degradation_count}/{val_patience})\n")

        # Stop if validation consistently degrading
        if val_degradation_count >= val_patience and episode > 100:
            print(f"\n🛑 Stopping: Validation degraded {val_patience} times in a row!")
            print(f"Best validation: {best_val_reward:.2f} at episode {best_val_episode}")
            print(f"Use saved_models/ppo/carracing_best_val.pth for evaluation!\n")
            return True
        return False
    
    for episode in range(CAR_RACING_CONFIG["max_episodes"]):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Convert state to tensor - already in (3, 32, 32) format from wrapper
            # Check for NaN in state
            if np.isnan(state).any():
                print(f"Warning: NaN detected in state at episode {episode}, step {episode_steps}")
                break
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Sample action
            with torch.no_grad():
                action_tensor, log_prob_tensor, _ = actor.sample(state_tensor)
                value_tensor = critic(state_tensor)
                
            action = action_tensor.cpu().numpy()[0]
            log_prob = log_prob_tensor.cpu().numpy()[0][0]  # Note: log_prob is scalar
            value = value_tensor.cpu().numpy()[0][0]
            
            # Clip actions to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Clip rewards only to prevent extreme values (numerical stability)
            reward = np.clip(reward, -10, 10)
            
            # Store transition
            trainer.store_transition(state, action, log_prob, value, reward, done)
            
            # Update when buffer is full
            if len(trainer.states) >= CAR_RACING_CONFIG["buffer_size"]:
                trainer.update()
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Render occasionally
            if episode % 20 == 0:
                env.render()
        
        # Update if there are remaining transitions
        if len(trainer.states) > 0:
            trainer.update()
        
        # Track running reward
        episode_rewards.append(episode_reward)
        if running_reward is None:
            running_reward = episode_reward
        else:
            running_reward = 0.05 * episode_reward + 0.95 * running_reward
        
        # Early stopping check (every 10 episodes)
        if episode > 0 and episode % 10 == 0:
            recent_rewards = episode_rewards[-30:] if len(episode_rewards) >= 30 else episode_rewards
            avg_reward = np.mean(recent_rewards)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 10
            
            # Stop if overfitting detected
            if episodes_without_improvement >= patience and episode > 100:
                print(f"\n⚠️  Early stopping at episode {episode}!")
                print(f"No improvement in last {patience} episodes.")
                print(f"Best 30-episode average: {best_avg_reward:.2f}")
                print(f"Preventing overfitting by stopping training.")
                break
        
        # Logging
        if episode % 10 == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Avg(10)={avg_reward:.2f}, Running={running_reward:.2f}, "
                  f"Steps={episode_steps}, Total={total_steps}")
            # also include recent trainer stats if available
            trainer_stats = trainer.get_recent_log_stats(n=20)
            log_payload = {
                "episode_reward": float(episode_reward),
                "avg_10_reward": float(avg_reward),
                "running_reward": float(running_reward),
                "episode": int(episode),
                "total_steps": int(total_steps)
            }
            if trainer_stats is not None:
                # prefix trainer keys to avoid collisions
                for k, v in trainer_stats.items():
                    log_payload[f"trainer/{k}"] = float(v)

            if wandb:
                try:
                    wandb.log(log_payload)
                except Exception:
                    pass
        else:
            print(f"Episode {episode}: Reward={episode_reward:.2f}")
            if wandb:
                try:
                    trainer_stats = trainer.get_recent_log_stats(n=20)
                    log_payload = {"episode_reward": float(episode_reward), "episode": int(episode), "total_steps": int(total_steps)}
                    if trainer_stats is not None:
                        for k, v in trainer_stats.items():
                            log_payload[f"trainer/{k}"] = float(v)
                    wandb.log(log_payload)
                except Exception:
                    pass
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            import os
            os.makedirs("saved_models/ppo", exist_ok=True)
            ckpt_path = "saved_models/ppo/carracing_best.pth"
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'config': CAR_RACING_CONFIG,
                'episode': episode,
                'reward': episode_reward
            }, ckpt_path)
            # also save separate state_dicts + metadata for safe loading
            _save_state_dicts_with_meta(ckpt_path[:-4], actor, critic, {"episode": episode, "reward": float(episode_reward)})
            print(f"New best model saved with reward: {best_reward:.2f}")
            if wandb:
                try:
                    wandb.save("saved_models/ppo/carracing_best.pth")
                    wandb.log({"best_reward": float(best_reward), "best_episode": int(episode)})
                except Exception:
                    pass
        
        # Validate on fresh tracks periodically to detect overfitting
        if episode % 25 == 0 and episode > 0:
            actor.eval()  # Disable dropout for validation
            val_rewards = []
            print(f"\n🔍 Validation on 10 fresh tracks...")
            for _ in range(10):
                val_env = make_env("carracing")
                val_state, _ = val_env.reset()
                val_reward = 0
                val_done = False
                val_steps = 0
                max_steps = 1000
                
                while not val_done and val_steps < max_steps:
                    val_state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        val_action, _, _ = actor.sample(val_state_tensor)
                    val_action_np = val_action.cpu().numpy()[0]
                    val_action_np = np.clip(val_action_np, -1.0, 1.0)
                    val_state, val_r, val_term, val_trunc, _ = val_env.step(val_action_np)
                    val_reward += np.clip(val_r, -10, 10)
                    val_done = val_term or val_trunc
                    val_steps += 1
                
                val_rewards.append(val_reward)
                val_env.close()
            
            val_avg = np.mean(val_rewards)
            val_std = np.std(val_rewards)
            val_min = min(val_rewards)
            val_max = max(val_rewards)
            val_rewards_history.append(val_avg)
            
            print(f"✅ Validation: Avg={val_avg:.2f}±{val_std:.2f}, Min={val_min:.2f}, Max={val_max:.2f}")
            print(f"   Training: Avg={running_reward:.2f}, Train/Val Gap={running_reward - val_avg:.2f}\n")
            actor.train()  # Re-enable dropout
            if wandb:
                wandb.log({
                    "val_avg": float(val_avg),
                    "val_std": float(val_std),
                    "val_min": float(val_min),
                    "val_max": float(val_max),
                    "train_running_reward": float(running_reward),
                    "train_val_gap": float(running_reward - val_avg),
                    "episode": int(episode)
                })
            
            # Save best validation model (most important!)
            if val_avg > best_val_reward:
                best_val_reward = val_avg
                best_val_episode = episode
                val_degradation_count = 0
                import os
                os.makedirs("saved_models/ppo", exist_ok=True)
                ckpt_path = "saved_models/ppo/carracing_best_val.pth"
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'config': CAR_RACING_CONFIG,
                    'episode': episode,
                    'val_reward': val_avg,
                    'train_reward': running_reward
                }, ckpt_path)
                _save_state_dicts_with_meta(ckpt_path[:-4], actor, critic, {"episode": episode, "val_reward": float(val_avg), "train_reward": float(running_reward)})
                print(f"🌟 New BEST VALIDATION model saved! Val={val_avg:.2f}\n")
                if wandb:
                    try:
                        wandb.save("saved_models/ppo/carracing_best_val.pth")
                        wandb.log({"best_val_reward": float(best_val_reward), "best_val_episode": int(best_val_episode)})
                    except Exception:
                        pass
            else:
                val_degradation_count += 1
                print(f"⚠️  Validation not improving ({val_degradation_count}/{val_patience})\n")
            
            # Stop if validation consistently degrading
            if val_degradation_count >= val_patience and episode > 100:
                print(f"\n🛑 Stopping: Validation degraded {val_patience} times in a row!")
                print(f"Best validation: {best_val_reward:.2f} at episode {best_val_episode}")
                print(f"Use saved_models/ppo/carracing_best_val.pth for evaluation!\n")
                break
            
            # Warning if large train/val gap
            if running_reward - val_avg > 150:
                print(f"⚠️  WARNING: Large train/val gap ({running_reward - val_avg:.2f})")
                print(f"   Model is overfitting! Consider stopping soon.\n")
        
        # Save checkpoints periodically
        if episode % CAR_RACING_CONFIG["save_interval"] == 0 and episode > 0:
            import os
            checkpoint_dir = "saved_models/ppo/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'config': CAR_RACING_CONFIG,
                'episode': episode,
                'reward': episode_reward
            }, f"{checkpoint_dir}/carracing_ep{episode}.pth")
            print(f"Checkpoint saved: carracing_ep{episode}.pth")
            if wandb:
                try:
                    wandb.save(f"{checkpoint_dir}/carracing_ep{episode}.pth")
                    wandb.log({"checkpoint_episode": int(episode)})
                except Exception:
                    pass
            # save separate state dicts + meta for this checkpoint
            try:
                ckpt_prefix = f"{checkpoint_dir}/carracing_ep{episode}"
                _save_state_dicts_with_meta(ckpt_prefix, actor, critic, {"episode": episode, "reward": float(episode_reward)})
            except Exception:
                pass

        # Step-based logging and validation every `log_interval_steps` steps
        if total_steps >= next_log_step:
            recent_rewards = episode_rewards[-30:] if len(episode_rewards) >= 30 else episode_rewards
            avg_recent = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else float(episode_reward)
            print(f"\n🕒 Step-based log: total_steps={total_steps}, recent_avg={avg_recent:.2f}, running={running_reward:.2f}, best_reward={best_reward:.2f}")
            if wandb:
                try:
                    wandb.log({
                        "step_log/total_steps": int(total_steps),
                        "step_log/recent_avg": float(avg_recent),
                        "step_log/running_reward": float(running_reward),
                        "step_log/best_reward": float(best_reward)
                    })
                except Exception:
                    pass

            # Run validation and optionally stop training if validation degrades
            stopped = do_validation("steps")
            next_log_step += log_interval_steps
            if stopped:
                break
    
    # Save final model
    import os
    save_dir = "saved_models/ppo"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'config': CAR_RACING_CONFIG,
        'episodes': CAR_RACING_CONFIG["max_episodes"],
        'best_reward': best_reward
    }, f"{save_dir}/carracing_final.pth")
    
    print(f"\nTraining completed!")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final model saved: saved_models/ppo/carracing_final.pth")
    if wandb:
        try:
            wandb.save(f"{save_dir}/carracing_final.pth")
            wandb.log({"final_best_reward": float(best_reward)})
        except Exception:
            pass
    # save actor/critic state_dicts + meta for final model
    try:
        _save_state_dicts_with_meta(f"{save_dir}/carracing_final", actor, critic, {"best_reward": float(best_reward)})
    except Exception:
        pass
        try:
            wandb.finish()
        except Exception:
            pass
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no-auto-wandb", action="store_true", help="Disable auto-enabling wandb from WANDB_API_KEY env var")
    args = parser.parse_args()

    # CLI args override config/env auto-detection
    if args.use_wandb:
        CAR_RACING_CONFIG["use_wandb"] = True
    if args.wandb_project:
        CAR_RACING_CONFIG["wandb_project"] = args.wandb_project
    if args.wandb_run_name:
        CAR_RACING_CONFIG["wandb_run_name"] = args.wandb_run_name
    if args.no_auto_wandb:
        # If user explicitly disables auto detection, clear env-driven enable
        os.environ.pop("WANDB_API_KEY", None)

    train()