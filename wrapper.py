import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
import highway_env

class MultiAgentEnvWrapper:
    def __init__(self, env_id="multizippermerge-v0"):
        self.env = gym.make(env_id)
        self.num_agents = len(self.env.unwrapped.controlled_vehicles)
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        obs_sample = self.env.reset()[0][0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs, _ = self.env.reset()
        return {f"agent_{i}": obs[i] for i in range(self.num_agents)}

    def step(self, action_dict):
        actions = [action_dict[f"agent_{i}"] for i in range(self.num_agents)]
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        obs_dict = {f"agent_{i}": obs[i] for i in range(self.num_agents)}
        rew_dict = {f"agent_{i}": rewards[i] for i in range(self.num_agents)}
        done_dict = {f"agent_{i}": terminateds[i] for i in range(self.num_agents)}
        trunc_dict = {f"agent_{i}": truncateds[i] for i in range(self.num_agents)}

        all_done = all(terminateds) or all(truncateds)
        return obs_dict, rew_dict, done_dict, trunc_dict, all_done

    def render(self):
        self.env.render()
