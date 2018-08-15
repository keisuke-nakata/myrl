import logging
import random

import numpy as np
import imageio

logger = logging.getLogger(__name__)

NOOP_ACTION = 0


class Actor:
    def __init__(self, env, policy, explorer, obs_preprocessor, n_noop_at_reset=(0, 30), n_stack_frames=4, n_action_repeat=4):
        self.env = env
        self.policy = policy
        self.explorer = explorer
        self.obs_preprocessor = obs_preprocessor
        self.n_noop_at_reset = n_noop_at_reset  # both inclusive
        self.n_stack_frames = n_stack_frames
        self.n_action_repeat = n_action_repeat

        self.is_done = True  # this flag will be also used as "require_reset".

    def _reset(self):
        """this method is the only one which calls `env.reset()`, and one of the two method which calls `env.step()` (the other is `self._repeat_action()`)"""
        self.episode_obses = []
        self.episode_processed_obses = []

        # reset
        observation = self.env.reset()
        self.is_done = False
        self.episode_obses.append(observation)

        # noop action at reset
        # because states consist of 4 skipped raw frames, some initial frames may be ignored in order to construct valid states.
        done = False
        n_random_actions = random.randint(*self.n_noop_at_reset)
        for _ in range(n_random_actions):
            observation, reward, done, info = self.env.step(NOOP_ACTION)
            self.episode_obses.append(observation)
        assert not done

        n_obses = len(self.episode_obses)
        assert n_obses == n_random_actions + 1
        if n_obses == 1:  # just reset, no noop were performed
            self.episode_processed_obses.append(self.obs_preprocessor(self.episode_obses[0], None))
        elif n_obses < self.n_action_repeat:  # some NOOP(s) were performed, but short to skip 4
            self.episode_processed_obses.append(self.obs_preprocessor(self.episode_obses[-1], self.episode_obses[-2]))
        else:  # some NOOPs were performed, sufficient to skip 4
            offset = n_obses % self.n_action_repeat
            for i in range(offset, n_obses, self.n_action_repeat):
                self.episode_processed_obses.append(self.obs_preprocessor(self.episode_obses[offset + 3], self.episode_obses[offset + 2]))

    def _repeat_action(self, action):
        """This method is one of the two method which calls `env.step()` (the other is `self._reset()`)"""
        assert not self.is_done
        reward = 0
        for repeat in range(1, self.n_action_repeat + 1):
            observation, current_reward, done, info = self.env.step(action)
            self.episode_obses.append(observation)
            reward += current_reward
            if done:
                break
        self.episode_processed_obses.append(self.obs_preprocessor(self.episode_obses[-1], self.episode_obses[-2]))
        return reward, done

    def act(self, step):
        if self.is_done:
            self._reset()
        state = self.state
        is_random, epsilon = self.explorer(step)
        if is_random:
            action = self.env.action_space.sample()
        else:
            action = self.policy(state)
        reward, done = self._repeat_action(action)
        self.is_done = done
        return state, action, reward, done, is_random, epsilon

    @property
    def state(self):
        if self.n_stack_frames == 0:
            state = self.episode_processed_obses[-1]
        else:
            state = self.episode_processed_obses[-self.n_stack_frames:]
            if len(state) < self.n_stack_frames:  # need padding
                state = [state[0].copy() for _ in range(self.n_stack_frames - len(state))] + state
            state = np.concatenate(state, axis=-1)
        return np.transpose(state.copy(), (2, 0, 1))  # chainer is channel first

    def dump_episode_gif(self, path):
        imageio.mimwrite(path, self.episode_obses, fps=60)
        logger.info(f'dump episode at {path}.')

    def load_parameters(self, path):
        self.policy.load_parameters(path)
