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
        return False, 0.0


class LinearAnnealEpsilonGreedyExplorer:
    def __init__(self, initial_epsilon, final_epsilon, final_exploration_step):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_exploration_step = final_exploration_step

    def __call__(self, step):
        epsilon = self.get_epsilon(step)
        is_random = np.random.uniform() < epsilon
        return is_random, epsilon

    def get_epsilon(self, step):
        if step > self.final_exploration_step:
            epsilon = self.final_epsilon
        else:  # linear annealing
            epsilon = (self.final_epsilon - self.initial_epsilon) / self.final_exploration_step * step + self.initial_epsilon
        return epsilon
