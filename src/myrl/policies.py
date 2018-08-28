import logging

import numpy as np
import chainer
from chainer.dataset.convert import to_device
from chainer.serializers import load_hdf5

CPU_ID = -1
logger = logging.getLogger(__name__)


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
        return action, q_values

    def load_parameters(self, path):
        load_hdf5(path, self.network)
        logger.info(f'load parameters from {path}')


class GreedyExplorer:
    def __init__(self, n_warmup_steps=0):
        self.n_warmup_steps = n_warmup_steps

    def __call__(self, step):
        if step < self.n_warmup_steps:
            is_random = True
            epsilon = 1.0
            warming_up = True
        else:
            is_random = False
            epsilon = 0.0
            warming_up = False
        return is_random, epsilon, warming_up


class EpsilonGreedyExplorer(GreedyExplorer):
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def __call__(self, step):
        is_random, epsilon, warming_up = super().__call__(step)
        if not warming_up:
            epsilon = self.get_epsilon(step)
            is_random = np.random.uniform() < epsilon
        return is_random, epsilon, warming_up

    def get_epsilon(self, step):
        return self.epsilon


class LinearAnnealEpsilonGreedyExplorer(EpsilonGreedyExplorer):
    def __init__(self, initial_epsilon, final_epsilon, final_exploration_step, *args, **kwargs):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_exploration_step = final_exploration_step
        super().__init__(epsilon=None, *args, **kwargs)

    def get_epsilon(self, step):
        if step > self.final_exploration_step:
            epsilon = self.final_epsilon
        else:  # linear annealing
            epsilon = (self.final_epsilon - self.initial_epsilon) / self.final_exploration_step * step + self.initial_epsilon
        return epsilon
