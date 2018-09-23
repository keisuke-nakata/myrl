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

    def __repr__(self):
        return f'<{self.__class__.__name__}(network={repr(self.network)}'
