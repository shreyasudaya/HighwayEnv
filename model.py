import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedMAPPO(nn.Module):
    def __init__(self, obs_space, act_space, agent_count, hidden_dim=128):
        super(CentralizedMAPPO, self).__init__()
        self.agent_count = agent_count
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.n

        # Actor (decentralized, per-agent)
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.act_dim)
            ) for _ in range(agent_count)
        ])

        # Critic (centralized, shared)
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim * agent_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, obs_tensor, mask=None):
        # obs_tensor: shape (n_agents, obs_dim)
        logits = torch.stack([actor(obs_tensor[i]) for i, actor in enumerate(self.actors)])
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Critic uses concatenated observations (centralized)
        with torch.no_grad():
            shared_obs = obs_tensor.view(1, -1)  # shape: (1, obs_dim * n_agents)
            value = self.critic(shared_obs)

        return actions, log_probs, value.squeeze()

    def evaluate_actions(self, obs_tensor, actions):
        # obs_tensor: (n_agents, obs_dim)
        logits = torch.stack([actor(obs_tensor[i]) for i, actor in enumerate(self.actors)])
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        shared_obs = obs_tensor.view(1, -1)
        value = self.critic(shared_obs)
        return log_probs, entropy, value.squeeze()
