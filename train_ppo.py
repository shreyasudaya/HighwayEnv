import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

def make_env():
    env = gym.make("zippermerge-v0")  # register your env as 'lane_drop_merge-v0'
    return env

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        input_dim = observation_space.shape[0]
        output_dim = action_space.n  # 4 actions: acc, dec, keep, LC
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.net(x)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value

from stable_baselines3.common.policies import ActorCriticPolicy

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
            activation_fn=nn.Tanh,
        )

def train():
    env = make_env()
    eval_env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        policy=CustomActorCriticPolicy,
        learning_rate=1e-3,
        batch_size=8192,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_lanedrop_tensorboard/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=5_000_000, callback=eval_callback)

if __name__ == "__main__":
    train()
