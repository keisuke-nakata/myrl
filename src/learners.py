import copy

import chainer
import chainer.functions as F
from chainer.serializers import save_hdf5

from utils import to_gpu_or_npfloat32


class BaseLearner:
    def __init__(self, network, optimizer, gamma=0.99):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma

        self.optimizer.setup(self.network)

        self.n_steps = 0

    def learn(self, experiences):
        batch = self._experiences2batch(experiences)
        self._learn(batch)
        self.n_steps += 1

    def dump_parameters(self, path):
        save_hdf5(filename=path, obj=self.network)

    def _experiences2batch(self, experiences):
        raise NotImplementedError

    def _learn(self, batch):
        raise NotImplementedError


class FittedQLearner(BaseLearner):
    """Fitted Q learning, a.k.a Q learning with target network."""
    def __init__(self, *args, **kwargs):
        """
        target_network_update_freq
            corresponds to the parameter C from Algorithm 1 of the DQN paper.
            (measured in the number of parameter updates)
        """
        self.target_network_update_freq = kwargs.pop('target_network_update_freq', 10_000)
        super().__init__(*args, **kwargs)
        self.target_network = copy.deepcopy(self.network)  # FIXME: self.network.copy(mode='copy')
        # TODO: target_network.to_gpu()?
        # network と target_network は同じデバイスにいないと非常にめんどくさいかも (データのコピーが行ったり来たりするので)

    def _experiences2batch(self, experiences):
        if self.n_steps % self.target_network_update_freq == 0:
            self._update_target_network()
        last_observations, actions, rewards, observations, dones = zip(*experiences)

        batch_x = to_gpu_or_npfloat32(last_observations, device=self.target_network._device_id)
        batch_reward = to_gpu_or_npfloat32(rewards, device=self.target_network._device_id)
        batch_done = to_gpu_or_npfloat32(dones, device=self.target_network._device_id)
        with chainer.no_backprop_mode():
            batch_target_q = self.target_network(batch_x)
            batch_y = batch_reward + self.gamma * batch_done * F.max(batch_target_q, axis=1)

        return (batch_x, batch_y)

    def _learn(self, batch):
        batch_x, batch_y = batch

        prediction_train = model(image_train)

    # Calculate the loss with softmax_cross_entropy
    loss = F.softmax_cross_entropy(prediction_train, target_train)

    # Calculate the gradients in the network
    model.cleargrads()
    loss.backward()

    # Update all the trainable paremters
    optimizer.update()

    def _update_target_network(self):
        self.target_network.copyparams(self.network)
