import numpy as np


class VanillaReplay:
    def __init__(self, limit=1_000_000):
        self.limit = limit
        self.experiences = []

    def push(self, experience):
        self.experiences.append(experience)
        self.experiences = self.experiences[-self.limit:]

    def sample(self, size):
        return np.random.choice(self.experiences, size=size, replace=False)
