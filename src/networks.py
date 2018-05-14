import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F


class VanillaCNN(chainer.Chain):
    """
    described in:
    - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (NIPS version)
    - https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (Nature version)

    We follow the nature one.
    """
    def __init__(self, n_actions):
        super().__init__()
        # Keras implementations use their default initilzier (GlorotUniform),
        # and the original DeepMind implementation also uses it (maybe...). See:
        # https://github.com/torch/nn/blob/master/SpatialConvolution.lua#L34
        # https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/master/dqn/convnet.lua
        initializer = initializers.GlorotUniform()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=4,
                out_channels=32,
                ksize=(8, 8),
                stride=4,
                initialW=initializer)
            self.conv2 = L.Convolution2D(
                in_channels=32,
                out_channels=64,
                ksize=(4, 4),
                stride=2,
                initialW=initializer)
            self.conv2 = L.Convolution2D(
                in_channels=64,
                out_channels=64,
                ksize=(3, 3),
                stride=1,
                initialW=initializer)
            self.linear1 = L.Linear(
                in_size=64 * 7 * 7,
                out_size=512,
                initialW=initializer)
            self.linear2 = L.Linear(
                in_size=512,
                out_size=n_actions,
                initialW=initializer)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
