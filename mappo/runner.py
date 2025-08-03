import torch
import torch.nn.functional as F

class MAPPOTrainer:
    def __init__(self, policy, buffer, config):
        self.policy = policy
        self.buffer = buffer
        self.config = config

    def update(self):
        obs, actions, rewards, dones, next_obs = self.buffer.get_batches()

        # Centralized value estimation
        num_agents = self.config["num_agents"]
        obs_reshaped = obs.view(-1, num_agents, obs.shape[-1])
        central_obs = obs_reshaped.reshape(obs.shape[0], -1).to(self.policy.device)

        actions = actions.to(self.policy.device)
        rewards = rewards.to(self.policy.device)

        # Compute value targets
        with torch.no_grad():
            values = self.policy.evaluate_values(central_obs)
            advantages = rewards - values

        log_probs, entropy = self.policy.evaluate(obs.to(self.policy.device), actions)

        # PPO losses
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = F.mse_loss(values, rewards)
        entropy_loss = -entropy.mean()

        loss = policy_loss + self.config["value_loss_coef"] * value_loss + self.config["entropy_coef"] * entropy_loss

        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
            self.config["max_grad_norm"]
        )
        self.policy.optimizer.step()

        self.buffer.clear()
        print(f"Policy loss: {policy_loss.item():.3f}, Value loss: {value_loss.item():.3f}, Entropy: {entropy.mean().item():.3f}")
