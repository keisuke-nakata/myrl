import logging
import random
import warnings
import os

import numpy as np
import chainer
from chainer.dataset.convert import to_device
# from ..replays import VanillaReplay

from .preprocessors import DoNothingPreprocessor
from .utils import Timer

logger = logging.getLogger(__name__)

CPU_ID = -1


class BaseActor:
    def __init__(
            self,
            env, network, policy, global_replay,
            n_action_repeat=1, obs_preprocessor=None, n_random_actions_at_reset=(0, 0), n_stack_frames=1,
            render_episode_freq=10, render_dir=None):
        if n_action_repeat != n_stack_frames:
            msg = 'Giving different n_action_repeat and n_stack_frames (given {} and {}) cause unexpected behavior.'.format(n_action_repeat, n_stack_frames)
            warnings.warn(msg)
            logger.warn(msg)
        self.env = env
        self.network = network
        self.policy = policy
        self.global_replay = global_replay
        self.n_action_repeat = n_action_repeat
        if obs_preprocessor is None:
            self.obs_preprocessor = DoNothingPreprocessor()
        else:
            self.obs_preprocessor = obs_preprocessor
        self.n_random_actions_at_reset = n_random_actions_at_reset  # both inclusive
        self.n_stack_frames = n_stack_frames
        self.render_episode_freq = render_episode_freq
        self.render_dir = render_dir

        # self.local_buffer = VanillaReplay()
        self.total_steps = 0
        self.total_episodes = 0
        self.require_reset = True
        self.rendering_mode = False

        self.timer = Timer()
        self.timer.start()

    def load_parameters(self, parameters):
        raise NotImplementedError

    # def act(self, local_buffer_size=1, n_steps=None):
    #     """
    #     Parameters
    #     ----------
    #     local_buffer_size : int, optional
    #         corresponds to "B" in the Ape-X paper (Algorithm 1)
    #     n_steps : int, optional
    #         corresponds to "T" in the Ape-X paper (Algorithm 1)
    #
    #     Returns
    #     -------
    #     steps : int
    #         #steps (actions) the actor tried.
    #     episodes : int
    #         #episodes the actor tried. Un-terminated episodes will be included.
    #     """
    #     raise NotImplementedError

    def warmup(self, n_steps):
        logger.info('warming up {} steps...'.format(n_steps))
        self._reset()

        step = 0
        self.timer.lap()
        while step < n_steps:
            action = self.env.action_space.sample()
            reward, done, current_step = self._repeat_action(action)
            self.global_replay.push((self.previous_state, np.int32(action), np.sign(reward), self.state, done))
            if done:
                self.timer.lap()
                logger.info('episode finished with reward {} in {} (total_time {})'.format(self.episode_reward, self.timer.laptime_str, self.timer.elapsed_str))
                self._reset()
            step += current_step
        # restore counter and env
        self.total_steps = 0
        self.total_episodes = 0
        self.require_reset = True
        logger.info('warming up {} steps... done ({} steps).'.format(n_steps, step))

    def _repeat_action(self, action):
        """与えられた行動を n 回繰り返す。
        ただし途中で done になった場合はそれ以上環境と通信せず、n に足りないぶんを最後の状態をコピーして観測バッファに詰める
        """
        logger.debug('repeating {} times: action {} times'.format(self.n_action_repeat, action))
        reward = 0
        for step in range(1, self.n_action_repeat + 1):
            observation, current_reward, done, info = self.env.step(action)
            logger.debug('action: {}, reward: {}, done: {}'.format(action, current_reward, done))
            reward += current_reward
            self._push_episode_obses(observation)
            self.episode_actions.append(action)
            if self.rendering_mode:
                img = self.env.render(mode='rgb_array')
            else:
                img = None
            self.episode_imgs.append(img)
            if done:
                for _ in range(self.n_action_repeat - step):
                    self._push_episode_obses(observation.copy())
                    logger.debug('The env is done. Pad the last observation as dummies to fill the buffer.')
                self.require_reset = True
                break
        # update counters
        self.total_steps += step
        self.episode_steps += step
        self.episode_reward += reward

        logger.debug('step {} of episode {} (action {}, reward {}, done {}) total_steps {}'.format(
            self.episode_steps, self.total_episodes, action, reward, done, self.total_steps))
        return reward, done, step

    def _reset(self):
        """
        `observation` is the raw/preprocessed observation from the env.
        `state` is the input for learners/replays, which may be stacked observations.
        """
        observation = self.env.reset()
        self.require_reset = False
        if self.rendering_mode:
            msg = 'env reset with rendering mode'
        else:
            msg = 'env reset without rendering mode'
        logger.debug(msg)

        # counters
        self.total_episodes += 1
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_imgs = []
        self.episode_obses = []

        for _ in range(self.n_stack_frames):
            self._push_episode_obses(observation.copy())  # fill episode_obses with the first frame

        if self.rendering_mode:
            img = self.env.render(mode='rgb_array')
        else:
            img = None
        self.episode_imgs.append(img)

        n_random_actions = random.randint(*self.n_random_actions_at_reset)
        logger.debug('take random {} action at reset'.format(n_random_actions))
        for _ in range(n_random_actions):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            logger.debug('action: {}, reward: {}, done: {}'.format(action, reward, done))
            self.total_steps += 1
            self.episode_steps += 1
            self.episode_reward += reward
            self.episode_actions.append(action)
            self._push_episode_obses(observation)
            if self.rendering_mode:
                img = self.env.render(mode='rgb_array')
            else:
                img = None
            self.episode_imgs.append(img)

        logger.debug('begin episode {}'.format(self.total_episodes))

    def _push_episode_obses(self, observation):
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

    def dump_episode(self, render_dir):
        os.makedirs(render_dir, exist_ok=True)
        for step, img in enumerate(self.episode_imgs):
            with open(os.path.join(render_dir, str(step) + '.txt'), 'w') as f:
                print(step, file=f)
        logger.info('dump an episode at {}'.format(render_dir))


class QActor(BaseActor):
    def act(self):
        if self.require_reset:
            if self.render_episode_freq > 0:
                self.rendering_mode = self.total_episodes % self.render_episode_freq == 0
            else:
                self.rendering_mode = False
            self._reset()

        # get action
        one_batch_state = np.asarray([self.state], dtype=np.float32)  # [state]: add (dummy) batch dim
        one_batch_state = to_device(self.network._device_id, one_batch_state)
        with chainer.no_backprop_mode():
            q_values = to_device(CPU_ID, self.network(one_batch_state).array)[0]
        action = self.policy(q_values, self.total_steps)

        # interact with env
        reward, done, current_step = self._repeat_action(action)

        # push experience to replay buffer
        # self.local_buffer.push((self._last_state, action, reward, self.episode_steps))
        # if len(self.local_buffer) >= local_buffer_size:  # TODO
        #     self.local_buffer.forward_to_replay()  # TODO
        self.global_replay.push((self.previous_state, np.int32(action), np.sign(reward), self.state, done))

        if done:
            self.timer.lap()
            logger.info('finished episode {} with reward {}, step {} in {} (total_steps {}, epsilon {}, total_time {})'.format(
                self.total_episodes, self.episode_reward, self.episode_steps, self.timer.laptime_str,
                self.total_steps, self.policy.get_epsilon(self.total_steps), self.timer.elapsed_str))
            if self.rendering_mode:
                render_dir = os.path.join(self.render_dir, 'episode{}'.format(self.total_episodes))
                self.dump_episode(render_dir)
