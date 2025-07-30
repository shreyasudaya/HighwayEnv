import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback



# --- 2. Training Config ---
TOTAL_TIMESTEPS = 5_000
MODEL_PATH = "ppo_multimerge.zip"
VIDEO_DIR = "./videos"

# --- 3. Create the environment ---
def make_env():
    env = gym.make(
        "multimerge-v0",
        render_mode=None,
        config={
            "controlled_vehicles": 10,
            "vehicles_count": 40,
        },
    )
    return env

# Vectorized for SB3
env = DummyVecEnv([make_env])

# --- 4. Define the PPO agent ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log="./ppo_multimerge_tb/",
)

# --- 5. Optional: Checkpoint every 50k steps ---
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="ppo_multimerge"
)

# --- 6. Train the agent ---
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- 7. Record a demo video after training ---
def record_video(model, video_length=300, filename="ppo_multimerge_demo.mp4"):
    # IMPORTANT: use render_mode="rgb_array" for recording
    def make_render_env():
        return gym.make(
            "multimerge-v0",
            render_mode="rgb_array",  # Required for VecVideoRecorder
            config={
                "controlled_vehicles": 5,
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