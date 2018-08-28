import logging

import chainer
import chainer.functions as F
from chainer.serializers import save_hdf5
from chainer.dataset.convert import to_device


logger = logging.getLogger(__name__)

CPU_ID = -1


class FittedQLearner:
    """Fitted Q learning, a.k.a Q learning with target network."""
    def __init__(self, network, optimizer, gamma=0.99):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma

        self.optimizer.setup(self.network)
        self.target_network = self.network.copy(mode='copy')  # this copies `_device_id` as well

    def learn(self, batch_state, batch_action, batch_reward, batch_done, batch_next_state):
        batch_state, batch_action, batch_reward, batch_done, batch_next_state = to_device(
            self.network._device_id,
            (batch_state, batch_action, batch_reward, batch_done, batch_next_state))

        with chainer.no_backprop_mode():
            batch_target_q = self.target_network(batch_next_state)
            batch_y = batch_reward + self.gamma * (1 - batch_done) * F.max(batch_target_q, axis=1)

        batch_q = F.select_item(self.network(batch_state), batch_action)
        assert len(batch_q.shape) == 1
        assert len(batch_y.shape) == 1
        assert batch_q.shape[0] == batch_y.shape[0]

        # loss = F.mean(F.huber_loss(batch_q, batch_y, delta=1.0, reduce='no'))
        loss = F.sum(F.huber_loss(batch_q, batch_y, delta=1.0, reduce='no'))

        with chainer.no_backprop_mode():
            td_error = F.mean_absolute_error(batch_q, batch_y)

        self.network.cleargrads()
        loss.backward()
        self.optimizer.update()

        loss_cpu = to_device(CPU_ID, loss.array)
        td_error_cpu = to_device(CPU_ID, td_error.array)
        return loss_cpu, td_error_cpu

    def update_target_network(self, soft=None):
        if soft is not None:
            raise NotImplementedError
        # if self.target_network_update_soft is not None:  # soft update
        #     raise NotImplementedError
        # else:  # hard update
        #     self.target_network.copyparams(self.network)
        #     logger.info(f'sync target network at learner updates {self.total_updates}')
        self.target_network.copyparams(self.network)
        logger.info(f'Updated target network.')

    def dump_parameters(self, path):
        save_hdf5(filename=path, obj=self.network)
        logger.info(f'dump parameters into {path}')
