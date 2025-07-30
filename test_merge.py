import gymnasium as gym
import highway_env
from pathlib import Path
import cv2
import numpy as np

def collect_frames(env, steps=300, num_episodes=5):
    all_frames = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done, truncated = False, False
        print(f"Recording episode {ep + 1}...")

        for _ in range(steps):
            if hasattr(env, "controlled_vehicles") and len(env.controlled_vehicles) > 1:
                action = [env.action_space.sample() for _ in env.controlled_vehicles]
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            all_frames.append(frame)

            if done or truncated:
                break

    return all_frames

def save_video(frames, video_path="multimerge_combined.mp4", fps=5):
    if not frames:
        print("No frames to save.")
        return None

    height, width, _ = frames[0].shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if not out.isOpened():
        print("Failed to open VideoWriter.")
        return None

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Combined video saved to: {video_path}")
    return video_path

# --- Create and run environment ---
env = gym.make(
    "multimerge-v0",
    render_mode="rgb_array",
    config={
        "controlled_vehicles": 10,
        "vehicles_count": 40,
    },
)

# Collect frames across episodes
frames = collect_frames(env, steps=300, num_episodes=5)

# Save them as one video
save_video(frames, video_path="multimerge_combined.mp4", fps=5)

env.close()
