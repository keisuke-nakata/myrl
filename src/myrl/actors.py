import logging
import random

import numpy as np
import imageio

from .replays import Experience


logger = logging.getLogger(__name__)

NOOP_ACTION = 0


class Actor:
    def __init__(self, env, policy, explorer, obs_preprocessor, reward_preprocessor=None, n_noop_at_reset=(0, 30), n_stack_frames=4, n_action_repeat=4):
        self.env = env
        self.policy = policy
        self.explorer = explorer
        self.obs_preprocessor = obs_preprocessor
        self.reward_preprocessor = reward_preprocessor
        self.n_noop_at_reset = n_noop_at_reset  # both inclusive
        self.n_stack_frames = n_stack_frames
        self.n_action_repeat = n_action_repeat

        self.is_done = True  # this flag will be also used as "require_reset".
        self.action_meanings = self.env.unwrapped.get_action_meanings()

    def _reset(self):
        """this method is the only one which calls `env.reset()`, and one of the two method which calls `env.step()` (the other is `self._repeat_action()`)"""
        self.episode_obses = []
        self.episode_processed_obses = []

        # reset
        observation = self.env.reset()
        self.is_done = False
        self.episode_obses.append(observation)

        # noop action at reset
        done = False
        n_random_actions = random.randint(*self.n_noop_at_reset)
        for _ in range(n_random_actions):
            observation, reward, done, info = self.env.step(NOOP_ACTION)
            self.episode_obses.append(observation)
            if done:
                observation = self.env.reset()
                self.episode_obses = [observation]
                done = False
        assert not done

        n_obses = len(self.episode_obses)
        assert n_obses > 0
        for i in range((n_obses - 1) % self.n_action_repeat, n_obses, self.n_action_repeat):
            if i == 0:
                last_obs = None
            else:
                last_obs = self.episode_obses[i - 1]
            self.episode_processed_obses.append(self.obs_preprocessor(self.episode_obses[i], last_obs))

    def _repeat_action(self, action):
        """This method is one of the two method which calls `env.step()` (the other is `self._reset()`)"""
        assert not self.is_done
        reward = 0
        for repeat in range(self.n_action_repeat):
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
        exploration_info = self.explorer(step)
        if exploration_info.is_random:
            action = self.env.action_space.sample()
            q_values = [np.nan] * self.env.action_space.n
        else:
            action, q_values = self.policy(state)
        reward, done = self._repeat_action(action)
        reward = self.reward_preprocessor(reward)
        self.is_done = done
        action_meaning = self.action_meanings[action]
        state_int = np.round(state * 255).astype(np.uint8)  # [0.0, 1.0] -> [0, 255]
        experience = Experience(state_int, action, reward, done)
        return experience, exploration_info, q_values, action_meaning

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

    def dump_episode_anim(self, path):
        imageio.mimwrite(path, self.episode_obses, fps=60)
        logger.info(f'dump episode animation at {path}.')

    def load_parameters(self, path):
        self.policy.load_parameters(path)
