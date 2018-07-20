import logging
import random
import warnings
import os
import csv
from collections import OrderedDict

import numpy as np
import imageio
import pandas as pd
import chainer
from chainer.dataset.convert import to_device
from chainer.serializers import load_hdf5

from .preprocessors import DoNothingPreprocessor
from .utils import Timer

logger = logging.getLogger(__name__)

CPU_ID = -1


class BaseActor:
    def __init__(
            self,
            env, network, policy,
            n_action_repeat=1, obs_preprocessor=None, n_random_actions_at_reset=(0, 0), n_stack_frames=1,
            result_dir='./'):
        if n_action_repeat != n_stack_frames:
            msg = f'Giving different n_action_repeat and n_stack_frames (given {n_action_repeat} and {n_stack_frames}) cause unexpected behavior.'
            warnings.warn(msg)
            logger.warn(msg)
        self.env = env
        self.network = network
        self.policy = policy
        self.n_action_repeat = n_action_repeat
        if obs_preprocessor is None:
            self.obs_preprocessor = DoNothingPreprocessor()
        else:
            self.obs_preprocessor = obs_preprocessor
        self.n_random_actions_at_reset = n_random_actions_at_reset  # both inclusive
        self.n_stack_frames = n_stack_frames
        self.result_dir = result_dir
        self.history_path = os.path.join(self.result_dir, 'actor_history.csv')

        os.makedirs(self.result_dir, exist_ok=True)

        self.total_steps = 0
        self.total_episodes = 0
        self.is_done = True  # this flag will be also used as "require_reset".

        self.timer = Timer()
        self.timer.start()

    def load_parameters(self, path):
        load_hdf5(path, self.network)
        logger.info(f'load parameters from {path}')

    def _take_action(self, action):
        obs, r, done, info = self.env.step(action)
        self.is_done = done
        return obs, r, done, info

    def _repeat_action(self, action, is_random):
        """与えられた行動を n 回繰り返す。
        ただし途中で done になった場合はそれ以上環境と通信せず、n に足りないぶんを最後の状態をコピーして観測バッファに詰める
        """
        logger.debug(f'repeating {self.n_action_repeat} times: action {action}')
        reward = 0
        for step in range(1, self.n_action_repeat + 1):
            observation, current_reward, done, info = self._take_action(action)
            logger.debug(f'action: {action}, reward: {current_reward}, done: {done}')
            reward += current_reward
            self._push_episode_obs(observation)
            self.episode_imgs.append(observation.copy())
            self.episode_actions.append(action)
            self.episode_actions_is_random.append(is_random)
            if done:
                for _ in range(self.n_action_repeat - step):
                    self._push_episode_obs(observation.copy())
                    logger.debug('The env is done. Pad the last observation as dummies to fill the buffer.')
                break
        # update counters
        self.total_steps += step
        self.episode_steps += step
        self.episode_reward += reward

        logger.debug(f'step {self.episode_steps} of episode {self.total_episodes} (action {action}, reward {reward}, done {done}) total_steps {self.total_steps}')
        return reward, done, step

    def _reset(self):
        """
        `observation` is the raw/preprocessed observation from the env.
        `state` is the input for learners/replays, which may be stacked observations.
        """
        self.timer.lap()

        observation = self.env.reset()
        self.is_done = False

        # counters
        self.total_episodes += 1
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_actions_is_random = []
        self.episode_obses = []
        self.episode_imgs = []
        self.local_experience_buffer = []

        for _ in range(self.n_stack_frames):
            self._push_episode_obs(observation.copy())  # fill episode_obses with the first frame
        self.episode_imgs.append(observation.copy())

        n_random_actions = random.randint(*self.n_random_actions_at_reset)
        logger.debug(f'take random {n_random_actions} action at reset')
        for _ in range(n_random_actions):
            action = self.env.action_space.sample()
            observation, reward, done, info = self._take_action(action)
            logger.debug(f'action: {action}, reward: {reward}, done: {done}')
            self.total_steps += 1
            self.episode_steps += 1
            self.episode_reward += reward
            self.episode_actions.append(action)
            self.episode_actions_is_random.append(True)
            self._push_episode_obs(observation)
            self.episode_imgs.append(observation.copy())

        logger.debug(f'begin episode {self.total_episodes}')

    def _push_episode_obs(self, observation):
        if not hasattr(self, 'last_observation'):
            self.last_observation = None
        processed_observation = self.obs_preprocessor(observation, self.last_observation)
        self.last_observation = observation
        self.episode_obses.append(processed_observation)

    @property
    def state(self):
        if self.n_stack_frames == 0:
            state = self.episode_obses[-1]
        else:
            state = np.concatenate(self.episode_obses[-self.n_stack_frames:], axis=-1)
        return np.transpose(state, (2, 0, 1))  # chainer is channel first

    @property
    def previous_state(self):
        if self.n_stack_frames == 0:
            previous_state = self.episode_obses[-2]
        else:
            previous_state = np.concatenate(self.episode_obses[-self.n_stack_frames * 2:-self.n_stack_frames], axis=-1)
        return np.transpose(previous_state, (2, 0, 1))  # chainer is channel first

    def dump_episode(self):
        if not self.is_done:
            msg = 'trying to dump a non-finished episode'
            warnings.warn(msg)
            logger.warn(msg)
        episode_dir = os.path.join(self.result_dir, f'episode{self.total_episodes:05}')
        os.makedirs(episode_dir, exist_ok=True)
        self.dump_step_history(os.path.join(episode_dir, 'actions.csv'))
        self.dump_episode_gif(os.path.join(episode_dir, 'play.gif'))
        self.dump_summary(os.path.join(episode_dir, 'summary.txt'))

    def dump_step_history(self, path):
        assert self.episode_steps == len(self.episode_actions) == len(self.episode_actions_is_random)
        d = OrderedDict([('episode_actions', self.episode_actions), ('episode_actions_is_random', self.episode_actions_is_random)])
        pd.DataFrame(d).to_csv(path, index=False)

    def dump_episode_gif(self, path):
        imageio.mimwrite(path, self.episode_imgs, fps=60)
        logger.info('dump an episode at {}'.format(path))

    def dump_summary(self, path):
        with open(path, 'w') as f:
            print(f'episode {self.total_episodes}', file=f)
            print(f'episode_step {self.episode_steps}', file=f)
            print(f'reward {self.episode_reward}', file=f)
            print(f'time {self.timer.laptime} ({self.timer.laptime_str})', file=f)
            print(f'fps {self.episode_steps / self.timer.laptime}', file=f)
            print(f'episilon {self.policy.get_epsilon(self.total_steps)}', file=f)
            print(f'total_steps {self.total_steps}', file=f)
            print(f'total_time {self.timer.elapsed} ({self.timer.elapsed_str})', file=f)

    def dump_episode_history(self, episode_seconds):
        if not hasattr(self, '_already_called_dump_episode_history'):
            mode = 'w'
        else:
            mode = 'a'
        with open(self.history_path, mode=mode) as f:
            writer = csv.writer(f)
            if mode == 'w':
                header = ['total_episodes', 'total_steps', 'episode_steps', 'episode_reward', 'episode_seconds', 'episode_fps', 'epsilon']
                writer.writerow(header)
            writer.writerow([
                self.total_episodes, self.total_steps, self.episode_steps, self.episode_reward, episode_seconds,
                self.episode_steps / episode_seconds, self.policy.get_epsilon(self.total_steps)])
        self._already_called_dump_episode_history = None  # dummy


class QActor(BaseActor):
    def warmup_act(self, n_steps):
        logger.info(f'warming up {n_steps} steps...')
        self._reset()

        step = 0
        self.timer.lap()
        while step < n_steps:
            action = self.env.action_space.sample()
            reward, done, current_step = self._repeat_action(action, is_random=True)
            experience = (self.previous_state, np.int32(action), reward, self.state, done)
            self.local_experience_buffer.append(experience)
            if done:
                self.timer.lap()
                yield self.local_experience_buffer
                logger.info(
                    f'finished warmup episode {self.total_episodes} '
                    f'with reward {self.episode_reward}, step {self.episode_steps} in {self.timer.laptime_str} '
                    f'({self.episode_steps / self.timer.laptime:.2f} fps) '
                    f'(total_steps {self.total_steps:,}, total_time {self.timer.elapsed_str})')
                self._reset()
            step += current_step
        # restore counter and env
        self.total_steps = 0
        self.total_episodes = 0
        self.is_done = True  # force "require_reset"
        logger.info(f'warming up {n_steps} steps... done ({step} steps).')
        yield self.local_experience_buffer

    def act(self):
        if self.is_done:
            self._reset()

        # get action
        one_batch_state = np.asarray([self.state], dtype=np.float32)  # [state]: add (dummy) batch dim
        one_batch_state = to_device(self.network._device_id, one_batch_state)
        with chainer.no_backprop_mode():
            q_values = to_device(CPU_ID, self.network(one_batch_state).array)[0]
        action, is_random = self.policy(q_values, self.total_steps)

        # interact with env
        reward, done, current_step = self._repeat_action(action, is_random=is_random)

        experience = (self.previous_state, np.int32(action), reward, self.state, done)

        if done:
            self.timer.lap()
            logger.info(
                f'finished episode {self.total_episodes} '
                f'with reward {self.episode_reward}, step {self.episode_steps} in {self.timer.laptime_str} '
                f'({self.episode_steps / self.timer.laptime:.2f} fps) '
                f'(epsilon {self.policy.get_epsilon(self.total_steps):.4}, total_steps {self.total_steps:,}, total_time {self.timer.elapsed_str})')
            self.dump_episode_history(self.timer.laptime)

        return experience, done

    def act_episode(self):
        while True:
            experience, done = self.act()
            self.local_experience_buffer.append(experience)
            if done:
                return self.local_experience_buffer
