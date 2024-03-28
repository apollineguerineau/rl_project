
import random
from collections import deque

class Memory:

    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.content = deque(maxlen=max_size)

    def __len__(self):
        return len(self.content)

    def add(self, state, action, reward, next_state, done):
        """Add new tuple (s_t, a_t, r_t, s_{t+1}) to memory"""
        self.content.append((state, action, reward, next_state, done))
    
    def sample(self, n):
        """Randomly select n tuples from memory"""
        sample = random.sample(self.content, n)
        return sample