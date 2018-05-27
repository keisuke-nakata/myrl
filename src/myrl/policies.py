import numpy as np


class Policy:
    pass


class EpsilonGreedy(Policy):
    def __init__(self, action_space, initial_epsilon, final_epsilon, final_exploration_step=0):
        self.action_space = action_space
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_exploration_step = final_exploration_step

    def __call__(self, q_values, step):
        if step > self.final_exploration_step:
            epsilon = self.final_epsilon
        else:  # linear annealing
            epsilon = (self.final_epsilon - self.initial_epsilon) / self.final_exploration_step * step + self.initial_epsilon

        if np.random.uniform() < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action
