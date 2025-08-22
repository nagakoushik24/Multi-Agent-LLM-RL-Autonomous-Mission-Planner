# agents/scripted_agent.py
from utils.a_star import astar
import time

class ScriptedAgent:
    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.env = env
        self.current_path = []
        self.target = None

    def assign_subgoal(self, target_cell):
        start = self.env.agent_pos[self.agent_id]
        path = astar(self.env.grid, start, tuple(target_cell))
        self.current_path = path[1:] if len(path) > 1 else []
        self.target = tuple(target_cell)
        return bool(self.current_path)

    def step(self):
        # if have path, move to next cell
        if self.current_path:
            next_cell = self.current_path.pop(0)
            # compute action to reach next_cell relative to current
            r,c = self.env.agent_pos[self.agent_id]
            nr,nc = next_cell
            if nr == r-1:
                action = 1
            elif nr == r+1:
                action = 2
            elif nc == c-1:
                action = 3
            elif nc == c+1:
                action = 4
            else:
                action = 0
        else:
            action = 0
        return action
