import logging
from collections import namedtuple

import numpy as np
import chainer
from chainer.dataset.convert import to_device
from chainer.serializers import load_hdf5

CPU_ID = -1
logger = logging.getLogger(__name__)

ExplorationInfo = namedtuple('ExplorationInfo', ['is_random', 'epsilon', 'warming_up'])


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

    def __repr__(self):
        return f'<{self.__class__.__name__}(network={repr(self.network)}'


class GreedyExplorer:
    def __init__(self, n_warmup_steps=0):
        self.n_warmup_steps = n_warmup_steps

    def __call__(self, step):
        if step < self.n_warmup_steps:  # warming_up
            is_random = True
            epsilon = 1.0
            warming_up = True
        else:
            is_random = False
            epsilon = 0.0
            warming_up = False
        return ExplorationInfo(is_random, epsilon, warming_up)

    def __repr__(self):
        return f'<{self.__class__.__name__}(n_warmup_steps={self.n_warmup_steps}'


class EpsilonGreedyExplorer(GreedyExplorer):
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def __call__(self, step):
        exploration_info = super().__call__(step)
        if exploration_info.warming_up:
            return exploration_info
        epsilon = self.get_epsilon(step)
        is_random = np.random.uniform() < epsilon
        warming_up = False
        return ExplorationInfo(is_random, epsilon, warming_up)

    def get_epsilon(self, step):
        return self.epsilon

    def __repr__(self):
        return f'<{self.__class__.__name__}(epsilon={self.epsilon}, n_warmup_steps={self.n_warmup_steps})>'


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

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}'
            f'(initial_epsilon={self.initial_epsilon}, final_epsilon={self.final_epsilon}, final_exploration_step={self.final_exploration_step}, '
            f'n_warmup_steps={self.n_warmup_steps})>'
        )
