import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
from pathlib import Path
import os
import cv2 # Import OpenCV
import numpy as np # Add numpy as frames are typically numpy arrays

# --- Utility to record frames and save video ---
def record_episode(env, steps=200, video_path="multimerge_episode.mp4", fps=30):
    frames = []
    obs, info = env.reset(seed=0)
    done, truncated = False, False

    for step in range(steps):
        # Random actions for all controlled vehicles (for now)
        if hasattr(env, "controlled_vehicles") and len(env.controlled_vehicles) > 1:
            action = [env.action_space.sample() for _ in env.controlled_vehicles]
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        frame = env.render() # This returns a NumPy array (H, W, C) for rgb_array
        frames.append(frame)

        if done or truncated:
            break

    env.close()

    # --- THIS IS THE MISSING PART: Saving the frames to video using OpenCV ---
    if not frames:
        print("No frames were captured to save video.")
        return None

    # Get dimensions from the first frame
    # OpenCV expects (width, height)
    height, width, _ = frames[0].shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    # For MP4, 'mp4v' (MPEG-4) is a widely compatible codec.
    # Other options: 'XVID', 'MJPG' (for .avi), 'avc1' (H.264, more system-dependent)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Good starting point for MP4

    out = None # Initialize out to None
    try:
        print(f"Saving video with OpenCV to {video_path}...")
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        if not out.isOpened():
            print(f"Error: Could not open video writer for {video_path}.")
            print("Check if the video_path is valid, file permissions, and if the codec ('mp4v') is supported on your system.")
            # For Windows, sometimes you might need K-Lite Codec Pack or ensure FFmpeg is robustly integrated with OpenCV.
            return None

        for frame in frames:
            # HighwayEnv's render() typically outputs RGB. OpenCV expects BGR.
            # Convert if colors look wrong (e.g., blue-ish).
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        print(f"Video successfully saved to {video_path}")
    except Exception as e:
        print(f"An error occurred during video writing: {e}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure
    finally:
        if out is not None:
            out.release() # IMPORTANT: Release the VideoWriter object

    return video_path

# --- Display the video inline (for testing) ---
def show_video(video_path):
    # This function is typically used in Jupyter notebooks or similar environments.
    # If you're running this from a plain Python script, this won't "display" anything
    # in your console. You'd need a video player.
    if Path(video_path).exists():
        from IPython.display import Video, display
        print(f"Attempting to display video from: {video_path}")
        display(Video(video_path, embed=True))
    else:
        print(f"Video file not found for display: {video_path}")

# --- Run the environment ---
env = gym.make(
    "multimerge-v0",
    render_mode="rgb_array", # Ensure this is 'rgb_array' for frame capture
    config={
        "controlled_vehicles": 3,
        "vehicles_count": 10,
    },
)

# Attempt to record and save the video
video_file = record_episode(env, steps=300, video_path="multimerge_demo.mp4")

# Only try to display if the video was successfully saved
if video_file:
    show_video(video_file)
else:
    print("Video saving failed, skipping display.")
    
# --- Display the video inline (for testing) ---
def show_video(video_path):
    from IPython.display import Video, display
    display(Video(video_path, embed=True))

# --- Run the environment ---
env = gym.make(
    "multimerge-v0",
    render_mode="rgb_array",
    config={
        "controlled_vehicles": 3,
        "vehicles_count": 10,
    },
)

video_file = record_episode(env, steps=300, video_path="multimerge_demo.mp4")
show_video(video_file)
