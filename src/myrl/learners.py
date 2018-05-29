import numpy as np

import chainer
import chainer.functions as F
from chainer.serializers import save_hdf5
from chainer.dataset.convert import to_device


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
        self.target_network = self.network.copy(mode='copy')
        # TODO: target_network.to_gpu()? しなくても、たぶん勝手に同じ device に存在する状態でコピーされるっぽい？要確認
        # network と target_network は同じデバイスにいないと非常にめんどくさいかも (データのコピーが行ったり来たりするので)

    def _experiences2batch(self, experiences):
        if self.n_steps % self.target_network_update_freq == 0:
            self._update_target_network()
        states, actions, rewards, next_states, dones = zip(*experiences)

        batch_x = to_device(self.target_network._device_id, np.asarray(states, dtype=np.float32))
        batch_action = to_device(self.target_network._device_id, np.asarray(actions, dtype=np.int32))
        batch_next_x = to_device(self.target_network._device_id, np.asarray(next_states, dtype=np.float32))
        batch_reward = to_device(self.target_network._device_id, np.asarray(rewards, dtype=np.float32))
        batch_done = to_device(self.target_network._device_id, np.asarray(dones, dtype=np.float32))
        with chainer.no_backprop_mode():
            batch_target_q = self.target_network(batch_next_x)
            batch_y = batch_reward + self.gamma * batch_done * F.max(batch_target_q, axis=1)

        return (batch_x, batch_y, batch_action)

    def _learn(self, batch):
        batch_x, batch_y, batch_action = batch
        batch_q = F.select_item(self.network(batch_x), batch_action)
        loss = F.mean_squared_error(batch_q, batch_y)

        self.network.cleargrads()
        loss.backward()
        self.optimizer.update()

    def _update_target_network(self):
        self.target_network.copyparams(self.network)
