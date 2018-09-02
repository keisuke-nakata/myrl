import chainer
import chainer.links as L
import chainer.functions as F


class VanillaCNN(chainer.Chain):
    """
    described in:
    - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (NIPS version)
    - https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (Nature version)

    This follows the Nature version.
    """
    def __init__(self, n_actions):
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
            self.linear1 = L.Linear(in_size=64 * 7 * 7, out_size=512, initial_bias=0.1)
            self.linear2 = L.Linear(in_size=512, out_size=n_actions)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
