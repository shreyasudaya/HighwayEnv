import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import highway_env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_AGENTS = 3
TOTAL_TIMESTEPS = 2_000_000
EPISODE_LENGTH = 300
UPDATE_EVERY = 2000
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
ENV_ID = "multizippermerge-v0"

# Create log dir
run_name = f"MAPPO_{ENV_ID}_{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")

# Multi-agent-aware MLPs
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy = nn.Linear(128, output_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

def make_env():
    env = gym.make(ENV_ID, render_mode=None)
    return env

# Environment
env = make_env()
obs_space = env.observation_space[0]
act_space = env.action_space[0]
obs_dim = np.prod(obs_space.shape)
action_dim = act_space.n

model = ActorCritic(obs_dim, action_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Buffers
obs_buf = []
action_buf = []
logprob_buf = []
reward_buf = []
done_buf = []
value_buf = []

obs, _ = env.reset()
episode_rewards = [0 for _ in range(NUM_AGENTS)]
timestep = 0

while timestep < TOTAL_TIMESTEPS:
    for _ in range(UPDATE_EVERY):
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).view(NUM_AGENTS, -1).to(device)
        logits, value = model(obs_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        next_obs, rewards, terminated, truncated, infos = env.step(actions.cpu().numpy())
        done = [a or b for a, b in zip(terminated, truncated)]

        obs_buf.append(obs_tensor.cpu().numpy())
        action_buf.append(actions.cpu().numpy())
        logprob_buf.append(dist.log_prob(actions).detach().cpu().numpy())
        reward_buf.append(rewards)
        done_buf.append(done)
        value_buf.append(value.squeeze(-1).detach().cpu().numpy())

        for i in range(NUM_AGENTS):
            episode_rewards[i] += rewards[i]

        obs = next_obs
        timestep += 1

        if all(done):
            for i, ep_r in enumerate(episode_rewards):
                writer.add_scalar(f"agent_{i}/episode_reward", ep_r, timestep)
            obs, _ = env.reset()
            episode_rewards = [0 for _ in range(NUM_AGENTS)]

    # Compute advantages and targets
    with torch.no_grad():
        next_obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).view(NUM_AGENTS, -1).to(device)
        _, next_value = model(next_obs_tensor)
        next_value = next_value.squeeze(-1).cpu().numpy()

    advantages = np.zeros_like(reward_buf)
    lastgaelam = np.zeros(NUM_AGENTS)
    for t in reversed(range(len(reward_buf))):
        if t == len(reward_buf) - 1:
            nextnonterminal = 1.0 - np.array(done_buf[t])
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - np.array(done_buf[t + 1])
            nextvalues = value_buf[t + 1]
        delta = np.array(reward_buf[t]) + GAMMA * nextvalues * nextnonterminal - value_buf[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

    returns = np.array(advantages) + np.array(value_buf)

    # Flatten buffers
    obs_train = torch.tensor(np.array(obs_buf).reshape(-1, obs_dim), dtype=torch.float32).to(device)
    actions_train = torch.tensor(np.array(action_buf).reshape(-1), dtype=torch.int64).to(device)
    old_logprobs_train = torch.tensor(np.array(logprob_buf).reshape(-1), dtype=torch.float32).to(device)
    returns_train = torch.tensor(returns.reshape(-1), dtype=torch.float32).to(device)
    advantages_train = torch.tensor(advantages.reshape(-1), dtype=torch.float32).to(device)

    # PPO update
    for _ in range(4):
        logits, value = model(obs_train)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions_train)
        entropy = dist.entropy().mean()

        ratio = (logprobs - old_logprobs_train).exp()
        surr1 = ratio * advantages_train
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_train
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((returns_train - value.squeeze(-1)) ** 2).mean()

        loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar("loss/actor", actor_loss.item(), timestep)
    writer.add_scalar("loss/critic", critic_loss.item(), timestep)
    writer.add_scalar("loss/total", loss.item(), timestep)

    # Reset buffers
    obs_buf.clear()
    action_buf.clear()
    logprob_buf.clear()
    reward_buf.clear()
    done_buf.clear()
    value_buf.clear()

writer.close()
print("Training complete.")