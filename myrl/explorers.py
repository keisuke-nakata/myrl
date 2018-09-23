import logging
from collections import namedtuple
import sys

import numpy as np

logger = logging.getLogger(__name__)

ExplorationInfo = namedtuple('ExplorationInfo', ['is_random', 'epsilon', 'warming_up'])


def build_explorer(explorer_config):
    Explorer = getattr(sys.modules[__name__], explorer_config['class'])
    explorer = Explorer(**explorer_config['params'])
    logger.info(f'built explorer {explorer}.')
    return explorer


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
            f'(initial_epsilon={self.initial_epsilon}, final_epsilon={self.final_epsilon}, '
            f'final_exploration_step={self.final_exploration_step}, n_warmup_steps={self.n_warmup_steps})>')
