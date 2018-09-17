import logging
import sys

import numpy as np
import chainer
import chainer.functions as F
from chainer.serializers import save_hdf5
from chainer.dataset.convert import to_device
from chainer import optimizers

logger = logging.getLogger(__name__)

CPU_ID = -1


def build_learner(network, learner_config, gamma=0.99, multi_step_n=None):
    Learner = getattr(sys.modules[__name__], learner_config['class'])
    optimizer = getattr(optimizers, learner_config['optimizer']['class'])(**learner_config['optimizer']['params'])
    if multi_step_n is not None:
        discount = gamma ** multi_step_n
    else:
        discount = gamma
    learner = Learner(network=network, optimizer=optimizer, discount=discount)
    logger.info(f'built learner {learner}.')
    return learner


class FittedQLearner:
    """Fitted Q learning, a.k.a Q learning with target network.

    Parameters
    ----------
    network : chainer.Chain
        network to be trained.
    optimizer : chainer.Optimizer
        optimizer to train the network.
    discount : float [0, 1]
        discount factor. Typically `discount = gamma`. For n-step Q-learning, set `discount = gamma ** multi_step_n`.
    """
    def __init__(self, network, optimizer, discount):
        self.network = network
        self.optimizer = optimizer
        self.discount = discount

        self.optimizer.setup(self.network)
        self.target_network = self.network.copy(mode='copy')  # this copies `_device_id` as well

    def learn(self, batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int):
        batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int = to_device(
            self.network._device_id,
            (batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int))

        batch_state = batch_state_int.astype(np.float32) / 255  # [0, 255] -> [0.0, 1.0]
        batch_next_state = batch_next_state_int.astype(np.float32) / 255  # [0, 255] -> [0.0, 1.0]

        batch_y, batch_q = self._compute_q_y(batch_state, batch_action, batch_reward, batch_done, batch_next_state)
        # assert len(batch_q.shape) == 1
        # assert len(batch_y.shape) == 1
        # assert batch_q.shape[0] == batch_y.shape[0]

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

    def _compute_q_y(self, batch_state, batch_action, batch_reward, batch_done, batch_next_state):
        with chainer.no_backprop_mode():
            batch_target_q = self.target_network(batch_next_state)
            batch_y = batch_reward + self.discount * (1 - batch_done) * F.max(batch_target_q, axis=1)
        batch_q = F.select_item(self.network(batch_state), batch_action)
        return batch_y, batch_q

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

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}'
            f'(network={self.network}, optimizer={self.optimizer.__class__.__name__}, discount={self.discount})>')


class DoubleQLearner(FittedQLearner):
    def _compute_q_y(self, batch_state, batch_action, batch_reward, batch_done, batch_next_state):
        with chainer.no_backprop_mode():
            batch_max = F.argmax(self.network(batch_next_state), axis=1)
            batch_target_q = self.target_network(batch_next_state)
            batch_y = batch_reward + self.discount * (1 - batch_done) * F.select_item(batch_target_q, batch_max)
        batch_q = F.select_item(self.network(batch_state), batch_action)
        return batch_y, batch_q
