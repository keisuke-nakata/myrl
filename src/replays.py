import numpy as np

# class BaseReplay:
#     def __init__(self, experiences):
#         self.experiences = experiences
#
#     def memorize(self, experience):
#         pass

# 
# class Experience:
#     def __init__(self, last_observation, action, reward, observation, done):
#         self.last_observation = last_observation
#         self.action = action
#         self.reward = reward
#         self.


class VanillaReplay:
    def __init__(self, limit=1_000_000):
        self.limit = limit
        self.experiences = []

    def push(self, experience):
        self.experiences.append(experience)
        self.experiences = self.experiences[-self.limit:]

    def sample(self, size):
        return np.random.choice(self.experiences, size=size, replace=False)
