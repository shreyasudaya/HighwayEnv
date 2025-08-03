# MAPPO Training Script with CTDE for MultiAgentLaneDropMergeEnv

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from typing import List

# Example environment
import highway_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Actor-Critic Network Definitions ========
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)  # make it (1, obs_dim) if (obs_dim,)
        logits = self.model(obs)
        return logits.squeeze(0)  # return shape (act_dim)

class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(global_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_obs):
        return self.model(global_obs)

# ======== Helper Functions ========
def select_actions(actors, obs_n):
    actions = []
    dists = []
    for i, actor in enumerate(actors):
        logits = actor(obs_n[i])
        dist = Categorical(logits=logits)
        action = dist.sample()
        actions.append(action)
        dists.append(dist)
    return actions, dists

# ======== Training Loop ========
def train(env_id='multizippermerge-v0', n_agents=3, total_episodes=1000):
    env = gym.make(env_id, render_mode=None)

    obs_sample = env.reset(seed=0)[0]
    obs_dim = obs_sample[0].shape[0]
    n_agents = len(obs_sample)
    act_dim = env.action_space[0].n
    global_obs_dim = obs_dim * n_agents

    actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
    critic = CentralizedCritic(global_obs_dim).to(device)

    actor_opts = [optim.Adam(actor.parameters(), lr=3e-4) for actor in actors]
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

    gamma = 0.99

    for ep in range(total_episodes):
        obs_tuple = env.reset(seed=ep)[0]
        obs_n = [torch.tensor(obs_tuple[i], dtype=torch.float32, device=device) for i in range(n_agents)]

        ep_rewards = [0.0 for _ in range(n_agents)]

        for step in range(20-+0):
            actions, dists = select_actions(actors, obs_n)
            action_dict = {f"agent_{i}": actions[i].item() for i in range(n_agents)}

            next_obs_tuple, reward_dict, terminated, truncated, _ = env.step(action_dict)
            done = all(terminated.values()) or all(truncated.values())

            next_obs_n = [torch.tensor(next_obs_tuple[i], dtype=torch.float32, device=device) for i in range(n_agents)]
            global_obs = torch.cat(obs_n, dim=-1).unsqueeze(0)  # shape: (1, global_obs_dim)
            global_next_obs = torch.cat(next_obs_n, dim=-1).unsqueeze(0)

            rewards = [reward_dict[f"agent_{i}"] for i in range(n_agents)]
            ep_rewards = [ep_rewards[i] + rewards[i] for i in range(n_agents)]
            reward_tensor = torch.tensor([sum(rewards)], dtype=torch.float32, device=device)

            # Centralized critic update
            value = critic(global_obs)
            next_value = critic(global_next_obs).detach()
            target = reward_tensor + gamma * next_value
            critic_loss = nn.functional.mse_loss(value, target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor updates (policy gradient with advantage from centralized critic)
            advantage = (target - value).detach()
            for i in range(n_agents):
                log_prob = dists[i].log_prob(actions[i])
                actor_loss = -log_prob * advantage
                actor_opts[i].zero_grad()
                actor_loss.backward()
                actor_opts[i].step()

            if done:
                break
            obs_n = next_obs_n

        print(f"Episode {ep} | Rewards: {[round(r, 2) for r in ep_rewards]}")

if __name__ == '__main__':
    train()