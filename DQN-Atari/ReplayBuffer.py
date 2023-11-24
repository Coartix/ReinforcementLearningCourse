# Experience replay buffer
from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, device, size=10000):
        self._maxsize = size
        self._storage = deque(maxlen=self._maxsize)
        self.device = device

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self._storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self._storage, batch_size)
        batch = np.array(batch, dtype=object)  # Convert to NumPy array

        states = np.array([obs[0] for obs in batch])
        actions = np.array([obs[1] for obs in batch])
        rewards = np.array([obs[2] for obs in batch])
        next_states = np.array([obs[3] for obs in batch])
        dones = np.array([obs[4] for obs in batch])

        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        return states, actions, rewards, next_states, dones
