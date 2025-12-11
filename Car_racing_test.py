"""
Comprehensive testing script for trained SAC agent on CarRacing-v2
- Tests for 100 episodes
- Generates episode vs reward plot
- Records 3 video demonstrations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import imageio

# Import the SAC agent
from sac_carracing import SAC_CarRacing


def test_agent_100_episodes(agent, env, save_dir):
    """Test agent for 100 episodes and collect statistics"""
    print("=" * 70)
    print("Testing agent for 100 episodes...")
    print("=" * 70)
    
    all_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(100), desc="Testing"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 1000:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        
        all_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Print every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episodes {episode-8}-{episode+1}: Avg Reward = {avg_reward:.2f}")
    
    # Calculate statistics
    stats = {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'min_reward': float(np.min(all_rewards)),
        'max_reward': float(np.max(all_rewards)),
        'median_reward': float(np.median(all_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'episodes_tested': 100,
        'all_rewards': [float(r) for r in all_rewards],
        'episode_lengths': [int(l) for l in episode_lengths]
    }
    
    # Save statistics
    stats_file = save_dir / 'test_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Min Reward: {stats['min_reward']:.2f}")
    print(f"Max Reward: {stats['max_reward']:.2f}")
    print(f"Median Reward: {stats['median_reward']:.2f}")
    print(f"Mean Episode Length: {stats['mean_episode_length']:.0f} steps")
    print("=" * 70)
    
    return all_rewards, episode_lengths, stats


def plot_rewards(rewards, save_dir):
    """Create and save reward plots"""
    print("\nGenerating plots...")
    
    episodes = range(1, len(rewards) + 1)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SAC Agent Performance on CarRacing-v2 (100 Episodes)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Episode vs Reward (Line plot)
    axes[0, 0].plot(episodes, rewards, linewidth=1.5, color='#2E86AB', alpha=0.7)
    axes[0, 0].axhline(y=np.mean(rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 0].axhline(y=900, color='green', linestyle='--', 
                       linewidth=2, label='Target: 900')
    axes[0, 0].set_xlabel('Episode', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Episode vs Reward', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # 2. Moving Average (window=10)
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(range(window, len(rewards) + 1), moving_avg, 
                    linewidth=2.5, color='#A23B72', label='10-Episode Moving Avg')
    axes[0, 1].fill_between(range(window, len(rewards) + 1), 
                            moving_avg - np.std(rewards), 
                            moving_avg + np.std(rewards), 
                            alpha=0.2, color='#A23B72')
    axes[0, 1].axhline(y=np.mean(rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Overall Mean: {np.mean(rewards):.2f}')
    axes[0, 1].set_xlabel('Episode', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Reward', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('10-Episode Moving Average', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # 3. Reward Distribution (Histogram)
    axes[1, 0].hist(rewards, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    axes[1, 0].axvline(x=np.median(rewards), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    axes[1, 0].set_xlabel('Reward', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Box Plot
    box = axes[1, 1].boxplot([rewards], vert=True, patch_artist=True, 
                             labels=['100 Episodes'])
    box['boxes'][0].set_facecolor('#6A994E')
    box['boxes'][0].set_alpha(0.7)
    axes[1, 1].axhline(y=900, color='green', linestyle='--', 
                       linewidth=2, label='Target: 900')
    axes[1, 1].set_ylabel('Reward', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Reward Distribution (Box Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend(fontsize=10)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    Mean: {np.mean(rewards):.2f}
    Std: {np.std(rewards):.2f}
    Min: {np.min(rewards):.2f}
    Max: {np.max(rewards):.2f}
    Median: {np.median(rewards):.2f}
    """
    axes[1, 1].text(0.5, 0.02, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = save_dir / 'performance_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {plot_file}")
    
    # Also save individual plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, rewards, linewidth=2, color='#2E86AB', marker='o', 
            markersize=3, alpha=0.7, label='Episode Reward')
    ax.axhline(y=np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax.axhline(y=900, color='green', linestyle='--', 
               linewidth=2, label='Target: 900')
    ax.fill_between(episodes, rewards, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('SAC Performance: Episode vs Reward', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    simple_plot_file = save_dir / 'episode_vs_reward.png'
    plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Simple plot saved to: {simple_plot_file}")
    
    plt.close('all')


def record_video(agent, env, video_path, max_steps=1000):
    """Record a single episode as video"""
    frames = []
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated) and steps < max_steps:
        # Get frame (render the environment)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Take action
        action = agent.select_action(state, evaluate=True)
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        steps += 1
    
    # Save video
    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        return episode_reward, steps
    return episode_reward, steps


def record_3_videos(agent, save_dir):
    """Record 3 demonstration videos"""
    print("\n" + "=" * 70)
    print("Recording 3 demonstration videos...")
    print("=" * 70)
    
    videos_dir = save_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)
    
    video_stats = []
    
    for i in range(3):
        print(f"\nRecording video {i+1}/3...")
        
        # Create environment with rgb_array render mode for recording
        env = gym.make("CarRacing-v2", continuous=True, render_mode='rgb_array')
        
        video_path = videos_dir / f'demo_episode_{i+1}.mp4'
        reward, steps = record_video(agent, env, video_path)
        
        video_stats.append({
            'video': f'demo_episode_{i+1}.mp4',
            'reward': float(reward),
            'steps': int(steps)
        })
        
        print(f"✓ Video {i+1} saved: {video_path}")
        print(f"  Reward: {reward:.2f}, Steps: {steps}")
        
        env.close()
    
    # Save video statistics
    video_stats_file = videos_dir / 'video_statistics.json'
    with open(video_stats_file, 'w') as f:
        json.dump(video_stats, f, indent=4)
    
    print("\n" + "=" * 70)
    print("Video Recording Complete!")
    print(f"Videos saved to: {videos_dir}")
    print("=" * 70)
    
    return video_stats


def main(model_path=None):
    """Main testing function"""
    print("\n" + "=" * 70)
    print("SAC CarRacing-v2 Comprehensive Testing")
    print("=" * 70)
    
    # Find model if not specified
    if model_path is None:
        print("\nSearching for trained model...")
        model_dir = Path("models/sac")
        if not model_dir.exists():
            model_dir = Path("/kaggle/working/models")
        
        model_files = list(model_dir.glob("**/*.pt"))
        if not model_files:
            print("ERROR: No model files found!")
            print(f"Searched in: {model_dir}")
            return
        
        # Look for best model first, otherwise use the most recent
        best_models = [f for f in model_files if 'best' in f.name.lower()]
        if best_models:
            model_path = best_models[0]
        else:
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    model_path = Path(model_path)
    print(f"✓ Using model: {model_path}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = model_path.parent / f'test_results_{timestamp}'
    results_dir.mkdir(exist_ok=True)
    print(f"✓ Results will be saved to: {results_dir}")
    
    # Initialize agent
    print("\nInitializing SAC agent...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    agent = SAC_CarRacing(
        action_dim=3,
        input_channels=3,
        hidden_dim=256,
        device=device
    )
    
    # Load trained model
    print("Loading trained model...")
    agent.load(str(model_path))
    print("✓ Model loaded successfully!")
    
    # Create testing environment (no render for 100 episodes)
    print("\nCreating test environment...")
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    
    # Test for 100 episodes
    all_rewards, episode_lengths, stats = test_agent_100_episodes(
        agent, env, results_dir
    )
    
    env.close()
    
    # Generate plots
    plot_rewards(all_rewards, results_dir)
    
    # Record 3 videos
    video_stats = record_3_videos(agent, results_dir)
    
    # Create final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Model tested: {model_path.name}")
    print(f"\n100-Episode Statistics:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min/Max: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(f"  Median: {stats['median_reward']:.2f}")
    print(f"  Avg Episode Length: {stats['mean_episode_length']:.0f} steps")
    
    print(f"\nVideo Demonstrations:")
    for i, vid_stat in enumerate(video_stats, 1):
        print(f"  Video {i}: Reward={vid_stat['reward']:.2f}, Steps={vid_stat['steps']}")
    
    print(f"\nAll results saved to: {results_dir}")
    print("  - test_statistics.json")
    print("  - performance_plots.png")
    print("  - episode_vs_reward.png")
    print("  - videos/demo_episode_1.mp4")
    print("  - videos/demo_episode_2.mp4")
    print("  - videos/demo_episode_3.mp4")
    print("  - videos/video_statistics.json")
    print("=" * 70)


if __name__ == "__main__":
   
    
    main("C:/Users/ziada/Downloads/best_model_car.pt")