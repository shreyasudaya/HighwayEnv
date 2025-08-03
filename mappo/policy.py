import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits


class Critic(nn.Module):
    def __init__(self, central_obs_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(central_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


class MAPPOPolicy:
    def __init__(self, config):
        obs_dim = config["obs_space"].shape[0]
        act_dim = config["act_space"].n
        central_obs_dim = config["central_obs_space"].shape[0]

        self.actor = Actor(obs_dim, act_dim).to(config["device"])
        self.critic = Critic(central_obs_dim).to(config["device"])

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=config["lr"]
        )
        self.device = config["device"]
        self.act_dim = act_dim

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def evaluate(self, obs_batch, actions):
        logits = self.actor(obs_batch)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def evaluate_values(self, central_obs):
        return self.critic(central_obs).squeeze(-1)
