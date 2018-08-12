import numpy as np
import chainer
from chainer.dataset.convert import to_device

CPU_ID = -1


class QPolicy:
    def __init__(self, network):
        self.network = network

    def __call__(self, state):
        one_batch_state = np.asarray([state], dtype=np.float32)
        one_batch_state = to_device(self.network._device_id, one_batch_state)
        with chainer.no_backprop_mode():
            q_values = self.network(one_batch_state)
        q_values = to_device(CPU_ID, q_values.array)[0]
        action = np.argmax(q_values)
        return action


class GreedyExplorer:
    def __call__(self, step):
        return False


class LinearAnnealEpsilonGreedyExplorer:
    def __init__(self, initial_epsilon, final_epsilon, final_exploration_step):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_exploration_step = final_exploration_step

    def __call__(self, step):
        is_random = np.random.uniform() < self.get_epsilon(step)
        return is_random

    def get_epsilon(self, step):
        if step > self.final_exploration_step:
            epsilon = self.final_epsilon
        else:  # linear annealing
            epsilon = (self.final_epsilon - self.initial_epsilon) / self.final_exploration_step * step + self.initial_epsilon
        return epsilon

# class Greedy(Policy):
#     def __init__(self, action_space):
#         self.action_space = action_space
#
#     def __call__(self, q_values, step):
#         action = np.argmax(q_values)
#         is_random = False
#         return action, is_random
#
#     def get_epsilon(self, *args, **kwargs):
#         return 0.0
#
#
# class EpsilonGreedy(Policy):
#     def __init__(self, action_space, initial_epsilon, final_epsilon, final_exploration_step=0):
#         self.action_space = action_space
#         self.initial_epsilon = initial_epsilon
#         self.final_epsilon = final_epsilon
#         self.final_exploration_step = final_exploration_step
#
#     def __call__(self, q_values, step):
#         is_random = np.random.uniform() < self.get_epsilon(step)
#         if is_random:
#             action = self.action_space.sample()
#         else:
#             action = np.argmax(q_values)
#         return action, is_random
#
#     def get_epsilon(self, step):
#         if step > self.final_exploration_step:
#             epsilon = self.final_epsilon
#         else:  # linear annealing
#             epsilon = (self.final_epsilon - self.initial_epsilon) / self.final_exploration_step * step + self.initial_epsilon
#         return epsilon
