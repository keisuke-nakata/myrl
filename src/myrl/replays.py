import random


class VanillaReplay:
    def __init__(self, limit=1_000_000 // 4):
        # limit: divided by action repeat
        self.limit = limit
        self.experiences = []

    def push(self, experience):
        self.experiences.append(experience)
        self.experiences = self.experiences[-self.limit:]

    def sample(self, size):
        return random.choices(self.experiences, k=size)

    def __len__(self):
        return len(self.experiences)
