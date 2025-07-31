import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt  
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# --- Config ---
TOTAL_TIMESTEPS = 10_000
MODEL_PATH = "dqn_multimerge.zip"
VIDEO_DIR = "./videos"

# --- Create environment ---
def make_env():
    env = gym.make(
        "zippermerge-v0",
        render_mode="rgb_array",
        config={
            "controlled_vehicles": 4,
            "vehicles_count": 20,
        },
    )
    return env

env = DummyVecEnv([make_env])

# --- Evaluation callback during training ---
class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_freq=5000, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.eval_timesteps = []
        self.rewards = []
        self.collision_rates = []
        self.avg_speeds = []
        self.speed_vars = []
        self.merge_success_rates = []
        self.mean_travel_times = []
        self.avg_queue_lengths = []
        self.avg_merge_times = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            stats = evaluate_policy(self.model, self.eval_env_fn, self.n_eval_episodes)

            self.eval_timesteps.append(self.num_timesteps)
            self.rewards.append(stats.get("avg_reward", 0.0))  # Optional, if added in future
            self.collision_rates.append(stats["collision_rate"])
            self.avg_speeds.append(stats["avg_speed"])
            self.speed_vars.append(stats["speed_variance"])
            self.merge_success_rates.append(stats["merge_success_rate"])
            self.mean_travel_times.append(stats["mean_travel_time"])
            self.avg_queue_lengths.append(stats["avg_queue_length"])
            self.avg_merge_times.append(stats["avg_time_to_merge"])

            print(f"\n[Eval after {self.n_calls} steps]")
            print(f"- Collision Rate:        {stats['collision_rate']:.2%}")
            print(f"- Merge Success Rate:    {stats['merge_success_rate']:.2%}")
            print(f"- Avg Speed:             {stats['avg_speed']:.2f} m/s")
            print(f"- Speed Variance:        {stats['speed_variance']:.2f} (m/s)^2")
            print(f"- Mean Travel Time:      {stats['mean_travel_time']:.2f} s")
            print(f"- Avg Queue Length:      {stats['avg_queue_length']:.2f} vehicles")
            print(f"- Avg Time to Merge:     {stats['avg_time_to_merge']:.2f} s")
            print(f"- Avg Reward:            {stats.get('avg_reward', 0.0):.2f}")
        return True

# --- Evaluation function ---
def evaluate_policy(model, env_fn, n_eval_episodes=5):
    eval_env = env_fn()

    # Lists to accumulate per-episode statistics
    episode_speeds = []
    episode_travel_times = []
    episode_merge_times = []
    episode_queue_lengths = []
    episode_entered = []
    episode_merged = []
    episode_collisions = []

    for episode in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False

        # Per-episode tracking
        speeds = []
        travel_times = []
        merge_times = []
        queue_lengths = []

        entered_vehicles = set()
        merge_candidates = set()
        merged_vehicles = set()
        crashed_vehicles = set()
        entry_time = {}
        merge_time = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            base_env = eval_env
            while hasattr(base_env, "env"):
                base_env = base_env.env

            for v in base_env.road.vehicles:
                speeds.append(v.speed)

                if getattr(v, "crashed", False) and v not in crashed_vehicles:
                    crashed_vehicles.add(v)

                if hasattr(v, "route") and v not in entered_vehicles:
                    entered_vehicles.add(v)
                    entry_time[v] = base_env.time

                if getattr(v, "lane_index", None) == ("a", "b", 1):
                    merge_candidates.add(v)

                if (
                    v in merge_candidates
                    and not v.crashed
                    and getattr(v, "lane_index", None) == ("b", "c", 0)
                    and v not in merged_vehicles
                    and v.lane.local_coordinates(v.position)[0] > 300
                ):
                    merged_vehicles.add(v)
                    merge_time[v] = base_env.time

            queue = sum(
                1
                for v in base_env.road.vehicles
                if getattr(v, "lane_index", None) == ("a", "b", 1)
                and v.lane.local_coordinates(v.position)[0] > 250
                and v.speed < 0.1
            )
            queue_lengths.append(queue)

        # Post-episode stats
        for v in entered_vehicles:
            t_entry = entry_time.get(v, 0)
            t_exit = base_env.time
            travel_times.append(t_exit - t_entry)

            if v in merge_time:
                merge_times.append(merge_time[v] - t_entry)

        # Accumulate per-episode stats
        episode_speeds.append(np.mean(speeds) if speeds else 0)
        episode_travel_times.append(np.mean(travel_times) if travel_times else 0)
        episode_merge_times.append(np.mean(merge_times) if merge_times else 0)
        episode_queue_lengths.append(np.mean(queue_lengths) if queue_lengths else 0)
        episode_entered.append(len(entered_vehicles))
        episode_merged.append(len(merged_vehicles))
        episode_collisions.append(len(crashed_vehicles))

    # Final metrics
    total_entered = sum(episode_entered)
    total_merged = sum(episode_merged)
    total_collisions = sum(episode_collisions)

    avg_speed = np.mean(episode_speeds)
    mean_travel_time = np.mean(episode_travel_times)
    avg_merge_time = np.mean(episode_merge_times)
    avg_queue_length = np.mean(episode_queue_lengths)
    merge_rate = total_merged / total_entered if total_entered > 0 else 0
    collision_rate = total_collisions / total_entered if total_entered > 0 else 0
    speed_variance = np.var(episode_speeds)

    # Display
    print("\n--- Evaluation Metrics (averaged over episodes) ---")
    print(f"Average speed: {avg_speed:.2f} m/s")
    print(f"Mean travel time: {mean_travel_time:.2f} s")
    print(f"Average time to merge: {avg_merge_time:.2f} s")
    print(f"Average queue length: {avg_queue_length:.2f} vehicles")
    print(f"Merge success rate: {merge_rate * 100:.2f}%")
    print(f"Collision rate: {collision_rate * 100:.2f}%")
    print(f"Speed variance: {speed_variance:.2f} (m/s)^2")
    print(f"Total vehicles entered: {total_entered}")
    print(f"Total successfully merged: {total_merged}")
    print(f"Total collisions: {total_collisions}")

    return {
        "avg_speed": avg_speed,
        "mean_travel_time": mean_travel_time,
        "avg_time_to_merge": avg_merge_time,
        "avg_queue_length": avg_queue_length,
        "merge_success_rate": merge_rate,
        "collision_rate": collision_rate,
        "speed_variance": speed_variance,
    }

def plot_training_stats(callback: EvaluationCallback):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(callback.eval_timesteps, callback.rewards, label="Avg Reward")
    plt.title("Average Reward over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(callback.eval_timesteps, callback.collision_rates, label="Collision Rate", color="red")
    plt.title("Collision Rate over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Rate")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(callback.eval_timesteps, callback.avg_speeds, label="Average Speed", color="green")
    plt.title("Average Speed over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(callback.eval_timesteps, callback.speed_vars, label="Speed Variance", color="orange")
    plt.title("Speed Variance over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Variance")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(callback.eval_timesteps, callback.merge_success_rates, label="Merge Success Rate", color="purple")
    plt.title("Merge Success Rate over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_stats.png")
    plt.show()

# --- Train the agent ---
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log="./dqn_multimerge_tb/",
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="dqn_multimerge"
)

eval_callback = EvaluationCallback(make_env, eval_freq=100, n_eval_episodes=5)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
plot_training_stats(eval_callback)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- Final evaluation after training ---
final_stats = evaluate_policy(model, make_env, n_episodes=10)
print("\nFinal Evaluation (10 episodes):")
print(f" - Average Reward       : {final_stats['avg_reward']:.2f}")
print(f" - Average Speed        : {final_stats['avg_speed']:.2f} m/s")
print(f" - Collision Rate       : {final_stats['collision_rate']:.2%}")
print(f" - Speed Variance       : {final_stats['speed_variance']:.2f}")
print(f" - Merge Rate       : {final_stats['merge_success_rate']:.2f}")
# --- Record a demo video ---
def record_video(model, video_length=300, filename="dqn_multimerge_demo.mp4"):
    def make_render_env():
        return gym.make(
            "zippermerge-v0",
            render_mode="rgb_array",
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
