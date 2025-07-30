import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env
import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback

# --- 2. Training Config ---
TOTAL_TIMESTEPS = 10_000  # DQN usually needs more steps than PPO
MODEL_PATH = "dqn_multimerge.zip"
VIDEO_DIR = "./videos"

# --- 3. Create the environment ---
def make_env():
    env = gym.make(
        "multimerge-v0",
        render_mode="rgb_array", 
        config={
            "controlled_vehicles": 3,
            "vehicles_count": 12,
        },
    )
    return env

# Vectorized for SB3 (DQN also works with this)
env = DummyVecEnv([make_env])

# --- 4. Define the DQN agent ---
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,        # Replay buffer size
    learning_starts=10_000,     # Start learning after this many steps
    batch_size=64,
    gamma=0.99,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.3,   # Fraction of training for exploration
    exploration_final_eps=0.05, # Final epsilon for epsilon-greedy
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log="./dqn_multimerge_tb/",
)

# --- 5. Optional: Checkpoint every 50k steps ---
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="dqn_multimerge"
)

# --- 6. Train the agent ---
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- 7. Record a demo video after training ---
def record_video(model, video_length=300, filename="dqn_multimerge_demo.mp4"):
    def make_render_env():
        return gym.make(
            "multimerge-v0",
            render_mode="rgb_array",  # Required for VecVideoRecorder
            config={
                "controlled_vehicles": 3,
                "vehicles_count": 10,
            },
        )

    video_env = DummyVecEnv([make_render_env])
    video_env = VecVideoRecorder(
        video_env, VIDEO_DIR, record_video_trigger=lambda x: x == 0,
        video_length=video_length, name_prefix="multimerge_eval"
    )

    obs = video_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = video_env.step(action)
        if done.any():
            break
    video_env.close()
    print(f"Video saved to {VIDEO_DIR}/{filename}")

record_video(model)
