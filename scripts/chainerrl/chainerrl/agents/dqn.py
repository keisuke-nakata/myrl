from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


class DQN(agent.AttributeSavingMixin, agent.Agent):
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
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`



    q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  update_interval=args.update_interval,
                  phi=phi
    """

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 average_q_decay=0.999,
                 average_loss_decay=0.99,
                 logger=getLogger(__name__),
                 batch_states=batch_states):
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
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.batch_states = batch_states
        update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

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
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

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

        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      batch_states=self.batch_states)
        loss = self._compute_loss(exp_batch, self.gamma)

        # Update stats
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def _compute_target_values(self, exp_batch, gamma):
        batch_next_state = exp_batch['next_state']

        target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          gamma (float): discount factor
        Returns:
          loss
        """
        y, t = self._compute_y_and_t(exp_batch, gamma)
        return F.sum(F.huber_loss(y, t, delta=1.0))

    def act(self, obs):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([obs], self.xp, self.phi))
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
                action_value = self.model(
                    self.batch_states([obs], self.xp, self.phi))
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[
                    0]

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

        self.replay_updater.update_if_necessary(self.t)

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
        if isinstance(self.model, Recurrent):
            self.model.reset_state()
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
        ]
