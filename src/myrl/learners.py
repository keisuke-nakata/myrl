import logging

import numpy as np

import chainer
import chainer.functions as F
from chainer.serializers import save_hdf5
from chainer.dataset.convert import to_device

from .utils import Timer


logger = logging.getLogger(__name__)

CPU_ID = -1


class BaseLearner:
    def __init__(self, network, optimizer, gamma=0.99, logging_freq=2000):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.logging_freq = logging_freq

        self.optimizer.setup(self.network)

        self.losses = []
        self.td_errors = []
        self.total_updates = 0
        self.timer = Timer()
        self.timer.start()

    def learn(self, experiences):
        batch = self._experiences2batch(experiences)
        loss, td_error = self._learn(batch)
        self.losses.append(loss)
        self.td_errors.append(td_error)

        self.total_updates += 1
        if self.logging_freq != 0 and self.total_updates % self.logging_freq == 0:
            self.timer.lap()
            n = len(self.losses)
            logger.info(
                f'finished {n} updates with avg loss {sum(self.losses) / n}, td_error {sum(self.td_errors) / n} in {self.timer.laptime_str} '
                f'({self.total_updates / self.timer.laptime:.2f} fps) '
                f'(total_updates {self.total_steps:,}, total_time {self.timer.elapsed_str})')
            self.losses = []
            self.td_errors = []
        return loss, td_error

    def dump_parameters(self, path):
        save_hdf5(filename=path, obj=self.network)
        logger.info(f'dump parameters into {path}')

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
        self.target_network_update_soft = kwargs.pop('target_network_update_soft', None)
        super().__init__(*args, **kwargs)
        self.target_network = self.network.copy(mode='copy')
        # TODO: target_network.to_gpu()? しなくても、たぶん勝手に同じ device に存在する状態でコピーされるっぽい？要確認
        # network と target_network は同じデバイスにいないと非常にめんどくさいかも (データのコピーが行ったり来たりするので)

    def _experiences2batch(self, experiences):
        if self.total_updates % self.target_network_update_freq == 0:
            self._sync_target_network()
        states, actions, rewards, next_states, dones = zip(*experiences)

        batch_x = to_device(self.target_network._device_id, np.asarray(states, dtype=np.float32))
        batch_action = to_device(self.target_network._device_id, np.asarray(actions, dtype=np.int32))
        batch_next_x = to_device(self.target_network._device_id, np.asarray(next_states, dtype=np.float32))
        batch_reward = to_device(self.target_network._device_id, np.asarray(rewards, dtype=np.float32))
        batch_done = to_device(self.target_network._device_id, np.asarray(dones, dtype=np.float32))
        with chainer.no_backprop_mode():
            batch_target_q = self.target_network(batch_next_x)
            batch_y = batch_reward + self.gamma * (1 - batch_done) * F.max(batch_target_q, axis=1)

        return (batch_x, batch_y, batch_action)

    def _learn(self, batch):
        batch_x, batch_y, batch_action = batch
        batch_q = F.select_item(self.network(batch_x), batch_action)
        loss = F.mean_squared_error(batch_q, batch_y)

        self.network.cleargrads()
        loss.backward()
        self.optimizer.update()

        with chainer.no_backprop_mode():
            td_error = F.mean_absolute_error(batch_q, batch_y)
        loss_cpu = to_device(CPU_ID, loss.array)
        td_error_cpu = to_device(CPU_ID, td_error.array)
        return loss_cpu, td_error_cpu

    def _sync_target_network(self):
        if self.target_network_update_soft is not None:  # soft update
            raise NotImplementedError
        else:  # hard update
            self.target_network.copyparams(self.network)
            logger.info(f'sync target network at learner updates {self.total_updates}')
