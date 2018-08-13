import numpy as np
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
        # Original DeepMind implementation looks to use Lua Torch with default weight initialization: https://github.com/deepmind/dqn/blob/master/dqn/convnet.lua
        # Lua Torch's default weight initializer looks like Uniform with endpoint 1/sqrt{fan_in}.
        # (For convolutions) https://github.com/torch/nn/blob/master/SpatialConvolution.lua#L38
        # (For linears) https://github.com/torch/nn/blob/master/Linear.lua#L25
        # HeUniform with scale = sqrt{6} is the one DeepMind did.
        initializer = initializers.HeUniform(scale=np.sqrt(6))
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
            self.conv3 = L.Convolution2D(
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
