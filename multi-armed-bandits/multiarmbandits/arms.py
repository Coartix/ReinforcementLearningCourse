import numpy as np

class ArmBernoulli:
    def __init__(self, p: float, random_state: int = 0):
        self.mean = p
        self.local_random = np.random.RandomState(random_state)
        
    def sample(self):
        return self.local_random.rand() < self.mean