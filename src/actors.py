import random

import numpy as np
import chainer
from chainer.dataset.convert import to_device
# from ..replays import VanillaReplay

from preprocessors import DoNothingPreprocessor


CPU_ID = -1


class BaseActor:
    def __init__(self, env, network, policy, global_replay, n_action_repeat=1, obs_preprocessor=None, n_random_actions_at_reset=(0, 0), n_stack_frames=1):
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

        self.obs_buf = []
        self._last_observation = None
        # self.local_buffer = VanillaReplay()
        self.total_steps = 0
        self.total_episodes = 0
        self.current_episode_steps = 0

    def load_parameters(self, parameters):
        raise NotImplementedError

    def act(self, local_buffer_size=1, n_steps=None):
        """
        Parameters
        ----------
        local_buffer_size : int, optional
            corresponds to "B" in the Ape-X paper (Algorithm 1)
        n_steps : int, optional
            corresponds to "T" in the Ape-X paper (Algorithm 1)

        Returns
        -------
        steps : int
            #steps (actions) the actor tried.
        episodes : int
            #episodes the actor tried. Un-terminated episodes will be included.
        """
        raise NotImplementedError

    def _reset(self):
        """
        `observation` is the raw/preprocessed observation from the env.
        `state` is the input for learners/replays, which may be stacked observations.
        """
        # observation = self.obs_preprocessor(self.env.reset())
        observation = self.env.reset()
        for _ in range(self.n_stack_frames):
            self._push_obs_buf(observation.copy())  # fill obs_buf first

        for _ in range(random.randint(*self.n_random_actions_at_reset)):
            # self._push_obs_buf(self.obs_preprocessor(self.env.step(self.env.action_space.sample())[0]))
            self._push_obs_buf(self.env.step(self.env.action_space.sample())[0])

        self.total_episodes += 1
        self.current_episode_steps = 0
        return self.state

    def _push_obs_buf(self, observation):
        processed_observation = self.obs_preprocessor(observation, self._last_observation)
        self.obs_buf.append(processed_observation)
        self._last_observation = observation
        self.obs_buf = self.obs_buf[-self.n_stack_frames:]

    @property
    def state(self):
        if self.n_stack_frames == 0:
            return self.obs_buf[-1]
        else:
            return np.concatenate(self.obs_buf, axis=-1)


class QActor(BaseActor):
    def act(self, local_buffer_size=1, n_steps=1):
        if not hasattr(self, '_last_state'):  # first interaction
            self._last_state = self._reset()
        for step in range(n_steps):
            q_values, action, state, reward, done = self._step(self._last_state)
            # self.local_buffer.push((self._last_state, action, reward, self.current_episode_steps))
            # if len(self.local_buffer) >= local_buffer_size:  # TODO
            #     self.local_buffer.forward_to_replay()  # TODO
            self.global_replay.push((self._last_state, action, reward, state, done))
            if done:
                self._last_state = self._reset()
            else:
                self._last_state = state

    def _step(self, state):
        one_batch_state = np.asarray([state], dtype=np.float32)  # [state]: add batch dim
        one_batch_state = to_device(self.network._device_id, one_batch_state)
        with chainer.no_backprop_mode():
            q_values = to_device(CPU_ID, self.network(one_batch_state)).array[0]
        action = self.policy(q_values, self.total_steps)

        reward = 0
        for _ in range(self.n_action_repeat):
            observation, current_reward, done, info = self.env.step(action)
            # self._push_obs_buf(self.obs_preprocessor(observation))
            self._push_obs_buf(observation)
            reward += current_reward
            if done:
                break

        self.total_steps += 1
        self.current_episode_steps += 1
        return q_values, action, self.state, reward, done
