import gymnasium as gym
import highway_env
import numpy as np
import cv2
from stable_baselines3 import PPO
from pathlib import Path

# --- Load the trained model ---
model_path = "ppo_lanedropmerge.zip"
model = PPO.load(model_path)

# --- Define environment ---
def make_env_for_render():
    return gym.make(
        "zippermerge-v0",
        render_mode="rgb_array",
        config={
            "controlled_vehicles": 2,
            "vehicles_count": 10,
        },
    )

# --- Run policy and collect frames ---
def record_agent_video(model, env, steps=300, episodes=5):
    all_frames = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, truncated = False, False
        print(f"Recording agent in episode {ep + 1}...")

        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            frame = env.render()
            all_frames.append(frame)

            if done or truncated:
                break

    return all_frames

# --- Save frames to video ---
def save_video(frames, filename="ppo_agent_simulation.mp4", fps=5):
    if not frames:
        print("No frames recorded.")
        return

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved: {filename}")

# --- Main execution ---
env = make_env_for_render()
frames = record_agent_video(model, env, steps=300, episodes=3)
save_video(frames, filename="ppo_agent_simulation.mp4", fps=5)
env.close()
