import torch
from torch_ac.utils import DictList
import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0

        self.memory = []

        self.temp = DictList()
        self.reset_temp()

    def reset_temp(self):
        self.temp.ob = []
        self.temp.ac = []
        self.temp.rew = []
        self.temp.mask = []

    def push_experience(self, exps):
        for i in range(len(exps.obs)):
            obs = exps.obs[i]
            action = exps.action[i]
            reward = exps.reward[i]
            mask = exps.mask[i]
            self.push(obs, action, reward, mask)
            
    def push(self, obs, action, reward, mask):
        """Saves a transition."""
        if (not mask):
            if len(self.memory) < self.capacity:
                self.memory.append(self.temp.copy())
            else:
                self.position = self.position if self.position < self.capacity else 0
                if np.random.random() > -0.5:
                    self.memory[self.position] = self.temp.copy()
            self.position += 1
                           
            self.temp.ob = [{"image": obs.image}]
            self.temp.ac = [action]
            self.temp.rew = [reward]
            self.temp.mask = [mask]
        else:
            self.temp.ob.append({"image": obs.image})
            self.temp.ac.append(action)
            self.temp.rew.append(reward)
            self.temp.mask.append(mask)
        
    def sample(self, batch_size):
        batch_idx = np.random.randint(0, len(self.memory), batch_size).tolist()
        return [self.memory[i] for i in batch_idx]

    def __setitem__(self, idx, temp):
        self.memory[idx] = temp.copy()

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)
