import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from models.SAC.SAC import SAC


def load_agent(model_path, state_dim, action_dim, device="cpu"):
    """Load SAC agent from checkpoint"""
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        device=device
    )
    agent.load(model_path)
    return agent


def evaluate_episode(env, agent):
    """Runs a single episode for recording"""
    state, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.select_action(state, evaluate=True)
        next_state, _, done, truncated, _ = env.step(action)
        state = next_state


def main():
    MODEL_PATH = "C:/Users/ziada/Downloads/final_model.pt"
    ENV_NAME = "LunarLander-v3"

    # Create environment that supports video frames
    base_env = gym.make(ENV_NAME, continuous=True, render_mode="rgb_array")

    # Wrap with video recorder — only trigger episodes 0, 1, 2
    env = RecordVideo(
        base_env,
        video_folder="videos_sac_test",
        episode_trigger=lambda ep: ep < 3   # Only record first 3 episodes
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Loading SAC model...")
    agent = load_agent(MODEL_PATH, state_dim, action_dim)

    NUM_EPISODES = 3
    print(f"Recording {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        print(f"Recording episode {ep + 1}...")
        evaluate_episode(env, agent)

    env.close()
    print("\n🎥 Videos saved in: videos_sac_test/")
    print("Done.")


if __name__ == "__main__":
    main()
