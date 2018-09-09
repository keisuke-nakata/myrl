import logging
import sys

import chainer
import chainer.links as L
import chainer.functions as F

logger = logging.getLogger(__name__)


def build_network(n_actions, network_config):
    Network = getattr(sys.modules[__name__], network_config['class'])
    if issubclass(Network, DuelingCNN):
        network = Network(n_actions, mode=network_config['dueling_mode'])
    else:
        network = Network(n_actions)
    logger.info(f'built network {network}.')
    return network


class CNNBase(chainer.Chain):
    """
    described in:
    - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (NIPS version)
    - https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (Nature version)

    This follows the Nature version.
    """
    def __init__(self, feature_size=512):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=4, out_channels=32,
                ksize=(8, 8), stride=4,
                initial_bias=0.1)
            self.conv2 = L.Convolution2D(
                in_channels=32, out_channels=64,
                ksize=(4, 4), stride=2,
                initial_bias=0.1)
            self.conv3 = L.Convolution2D(
                in_channels=64, out_channels=64,
                ksize=(3, 3), stride=1,
                initial_bias=0.1)
            self.f_linear = L.Linear(in_size=64 * 7 * 7, out_size=feature_size, initial_bias=0.1)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.f_linear(x))
        return x


class VanillaCNN(chainer.Chain):
    def __init__(self, n_actions, feature_size=512):
        self.n_actions = n_actions
        self.feature_size = feature_size
        super().__init__()
        with self.init_scope():
            self.cnn_base = CNNBase(feature_size=feature_size)
            self.q_linear = L.Linear(in_size=feature_size, out_size=n_actions)

    def __call__(self, x):
        x = self.cnn_base(x)
        x = self.q_linear(x)
        return x

    def __repr__(self):
        return f'<{self.__class__.__name__}(n_actions={self.n_actions}, feature_size={self.feature_size})>'


class DuelingCNN(chainer.Chain):
    def __init__(self, n_actions, mode, feature_size=512):
        self.n_actions = n_actions
        self.mode = mode
        self.feature_size = feature_size
        super().__init__()
        with self.init_scope():
            self.cnn_base = CNNBase(feature_size=feature_size)
            self.a_linear = L.Linear(in_size=feature_size, out_size=n_actions)
            self.v_linear = L.Linear(in_size=feature_size, out_size=1)

    def __call__(self, x):
        x = self.cnn_base(x)
        v = self.v_linear(x)
        a = self.a_linear(x)
        if self.mode == 'naive':
            v, a = F.broadcast(v, a)
            ret = v + a
        elif self.mode == 'avg':
            a_mean = F.mean(a, axis=1, keepdims=True)
            v, a, a_mean = F.broadcast(v, a, a_mean)
            ret = v + a - a_mean
        elif self.mode == 'max':
            a_max = F.max(a, axis=1, keepdims=True)
            v, a, a_max = F.broadcast(v, a, a_max)
            ret = v + a - a_max
        else:
            raise ValueError(f'Unknown mode {self.mode}')
        return ret

    def __repr__(self):
        return f'<{self.__class__.__name__}(n_actions={self.n_actions}, mode={self.mode}, feature_size={self.feature_size})>'
