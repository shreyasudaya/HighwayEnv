import torch

class SharedReplayBuffer:
    def __init__(self, config, obs_shape, act_shape):
        self.device = config["device"]
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.next_obs_buf = []

    def insert(self, obs, actions, rewards, dones, next_obs):
        self.obs_buf.append(obs)
        self.act_buf.append(actions)
        self.rew_buf.append(rewards)
        self.done_buf.append(dones)
        self.next_obs_buf.append(next_obs)

    def get_batches(self):
        obs = torch.tensor([list(o.values()) for o in self.obs_buf], dtype=torch.float32).view(-1, 7)
        act = torch.tensor([list(a.values()) for a in self.act_buf], dtype=torch.int64).view(-1)
        rew = torch.tensor([list(r.values()) for r in self.rew_buf], dtype=torch.float32).view(-1)
        done = torch.tensor([list(d.values()) for d in self.done_buf], dtype=torch.float32).view(-1)
        next_obs = torch.tensor([list(n.values()) for n in self.next_obs_buf], dtype=torch.float32).view(-1, 7)
        return obs, act, rew, done, next_obs

    def clear(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.next_obs_buf.clear()