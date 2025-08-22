# agents/train_ppo.py
"""
Skeleton: trains an SB3 PPO agent on a *single-agent* wrapper of the grid env.
This file is provided as a basis if you want to later train learned low-level policies.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.grid_env import MultiAgentGridEnv
import gym
import numpy as np

class SingleAgentWrapper(gym.Env):
    """
    Wrap the multi-agent env as single-agent by controlling only agent_0,
    while other agents follow scripted policy (e.g., random).
    """
    def __init__(self):
        self.env = MultiAgentGridEnv(n_agents=2)
        self.agent_id = "agent_0"
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        return obs[self.agent_id]

    def step(self, action):
        # other agent does random
        actions = {aid: 0 for aid in self.env.agent_ids}
        actions[self.agent_id] = action
        # simple scripted behavior for other agent
        for aid in self.env.agent_ids:
            if aid != self.agent_id:
                actions[aid] = self.env.rng.randint(0,4)
        obs, rewards, done, info = self.env.step(actions)
        return obs[self.agent_id], rewards[self.agent_id], done, info

def train(total_timesteps=200_000):
    env = DummyVecEnv([lambda: SingleAgentWrapper()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_agent0")
    print("Saved ppo_agent0.zip")

if __name__ == "__main__":
    train(50000)
