from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent


def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


class DQN(agent.AttributeSavingMixin):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        logger (Logger): Logger used
    """

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 average_q_decay=0.999,
                 average_loss_decay=0.99,
                 logger=getLogger(__name__)):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.logger = logger

        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()
        # For backward compatibility
        self.target_q_function = self.target_model
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            target_params = dict(self.target_model.namedparams())
            for param_name, param in self.model.namedparams():
                if target_params[param_name].data is None:
                    raise TypeError(
                        'self.target_model parameter {} is None. Maybe the model params are '
                        'not initialized.\nPlease try to forward dummy input '
                        'beforehand to determine parameter shape of the model.'.format(
                            param_name))
                target_params[param_name].data[:] = param.data
            print('synced target model.')

    def update(self, experiences):
        """Update the model from experiences

        This function is thread-safe.
        Args:
          experiences (list): list of dict that contains
            state: cupy.ndarray or numpy.ndarray
            action: int [0, n_action_types)
            reward: float32
            next_state: cupy.ndarray or numpy.ndarray
            next_legal_actions: list of booleans; True means legal
          gamma (float): discount factor
        Returns:
          None
        """
        exp_batch = {
            'state': self.xp.asarray([phi(elem['state']) for elem in experiences]),
            'action': self.xp.asarray([elem['action'] for elem in experiences]),
            'reward': self.xp.asarray([elem['reward'] for elem in experiences], dtype=np.float32),
            'next_state': self.xp.asarray([phi(elem['next_state']) for elem in experiences]),
            'next_action': self.xp.asarray([elem['next_action'] for elem in experiences]),
            'is_state_terminal': self.xp.asarray([elem['is_state_terminal'] for elem in experiences], dtype=np.float32)
        }
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_next_state = exp_batch['next_state']

            target_next_qout = self.target_model(batch_next_state)
            next_q_max = target_next_qout.max

            batch_rewards = exp_batch['reward']
            batch_terminal = exp_batch['is_state_terminal']

            batch_q_target = F.reshape(batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max, (batch_size, 1))

        loss = F.sum(F.huber_loss(batch_q, batch_q_target, delta=1.0))

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def act(self, obs):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(self.xp.asarray([phi(obs)]))
                q = float(action_value.max.data)
                action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(self.xp.asarray([phi(obs)]))
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        if len(self.replay_buffer) < self.replay_start_size:
            pass
        elif self.t % self.update_interval != 0:
            pass
        else:
            transitions = self.replay_buffer.sample(self.minibatch_size)
            self.update(transitions)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)

        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
        ]
