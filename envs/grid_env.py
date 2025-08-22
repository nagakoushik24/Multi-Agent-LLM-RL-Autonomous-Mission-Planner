import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import imageio
import os

class MultiAgentGridEnv(gym.Env):
    """
    Simple multi-agent grid:
    - grid of size HxW
    - cells: 0 free, 1 obstacle, 2 goal (shared), 3 agent
    - agents start at different cells
    Action space (per agent): 0=stay,1=up,2=down,3=left,4=right
    Observation (per agent): local view size 5x5 flattened + own coords + goal coords + other agents coords
    This is a simple demo env (not PettingZoo). Actions passed as dict {agent_id:action}
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, height=11, width=11, n_agents=2, obstacle_prob=0.18, max_steps=200, seed=None):
        super().__init__()
        self.h = height
        self.w = width
        self.n_agents = n_agents
        self.obstacle_prob = obstacle_prob
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        # action space for each agent
        self.agent_ids = [f"agent_{i}" for i in range(n_agents)]
        self.action_space = spaces.Discrete(5)
        # observation size: flattened 5x5 local + own r,c + goal r,c + (n_agents-1)*2
        local = 5*5
        self.obs_size = local + 2 + 2 + (n_agents-1)*2
        self.observation_space = spaces.Box(low=-100, high=100, shape=(self.obs_size,), dtype=np.float32)
        self.reset()

    def _new_grid(self):
        grid = np.zeros((self.h, self.w), dtype=np.int32)
        for i in range(self.h):
            for j in range(self.w):
                if self.rng.random() < self.obstacle_prob:
                    grid[i,j] = 1
        return grid

    def reset(self):
        self.steps = 0
        self.grid = self._new_grid()
        # pick goal cell on free cell
        while True:
            gr = self.rng.randrange(self.h)
            gc = self.rng.randrange(self.w)
            if self.grid[gr,gc] == 0:
                self.grid[gr,gc] = 2
                self.goal = (gr,gc)
                break
        # place agents
        self.agent_pos = {}
        for aid in self.agent_ids:
            while True:
                r = self.rng.randrange(self.h)
                c = self.rng.randrange(self.w)
                if self.grid[r,c] == 0 and (r,c) != self.goal and (r,c) not in self.agent_pos.values():
                    self.agent_pos[aid] = (r,c)
                    break
        return self._get_obs()

    def _get_local(self, pos, size=5):
        pad = size//2
        big = np.ones((self.h + 2*pad, self.w + 2*pad), dtype=np.int32)
        big[pad:pad+self.h, pad:pad+self.w] = self.grid
        r, c = pos
        r += pad; c += pad
        view = big[r-pad:r+pad+1, c-pad:c+pad+1]
        return view.flatten()

    def _get_obs(self):
        obs = {}
        for aid in self.agent_ids:
            own = self.agent_pos[aid]
            local = self._get_local(own).astype(np.float32)
            own_coords = np.array(own, dtype=np.float32)
            goal_coords = np.array(self.goal, dtype=np.float32)
            others = []
            for oid in self.agent_ids:
                if oid == aid: continue
                others.extend(list(self.agent_pos[oid]))
            others = np.array(others, dtype=np.float32) if others else np.array([], dtype=np.float32)
            vec = np.concatenate([local, own_coords, goal_coords, others])
            obs[aid] = vec
        return obs

    def step(self, actions):
        """
        actions: dict {agent_id: action_int}
        returns obs, rewards, done, info
        """
        self.steps += 1
        rewards = {aid: -0.01 for aid in self.agent_ids}  # small step penalty
        info = {}
        done = False
        # apply actions (simultaneous)
        new_positions = {}
        for aid, act in actions.items():
            r,c = self.agent_pos[aid]
            if act == 1: nr, nc = r-1, c
            elif act == 2: nr, nc = r+1, c
            elif act == 3: nr, nc = r, c-1
            elif act == 4: nr, nc = r, c+1
            else: nr, nc = r, c
            # bounds check
            if not (0 <= nr < self.h and 0 <= nc < self.w): nr, nc = r, c
            # obstacle check
            if self.grid[nr,nc] == 1:
                nr, nc = r, c
            new_positions[aid] = (nr, nc)
        # collision resolution: if two agents intend same cell, they both stay
        dests = {}
        for aid, pos in new_positions.items():
            dests.setdefault(pos, []).append(aid)
        for pos, a_list in dests.items():
            if len(a_list) > 1:
                # all stay in place
                for aid in a_list:
                    new_positions[aid] = self.agent_pos[aid]
        self.agent_pos = new_positions
        # rewards for reaching goal
        for aid, pos in self.agent_pos.items():
            if pos == self.goal:
                rewards[aid] += 10.0
                done = True
        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), rewards, done, info

    def render(self, scale=20):
        canvas = np.ones((self.h*scale, self.w*scale, 3), dtype=np.uint8) * 255
        for i in range(self.h):
            for j in range(self.w):
                val = self.grid[i,j]
                color = (200,200,200) if val==1 else (255,255,255)
                if val==2:
                    color = (0,200,0)
                canvas[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = color
        for aid, pos in self.agent_pos.items():
            r,c = pos
            canvas[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = (0,0,200)
        return canvas

    def save_render(self, path='frame.png'):
        arr = self.render()
        imageio.imwrite(path, arr)
