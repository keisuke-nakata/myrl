import numpy as np
import chainer
from chainer.cuda import to_cpu, to_gpu
# from ..replays import VanillaReplay

from utils import to_gpu_or_npfloat32


class BaseActor:
    def __init__(self, env, network, policy, global_replay):
        self.env = env
        self.network = network
        self.policy = policy
        self.global_replay = global_replay

        # self.local_buffer = VanillaReplay()
        self.total_steps = 0
        self.total_episodes = 0
        self.current_episode_steps = 0

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

    def load_parameters(self, parameters):
        raise NotImplementedError


class QActor(BaseActor):
    def act(self, local_buffer_size=1, n_steps=1):
        if not hasattr(self, '_last_observation'):  # first interaction
            self._last_observation = self._reset()
        for step in range(n_steps):
            q_values, action, observation, reward, done, info = self._step(self._last_observation)
            # self.local_buffer.push((self._last_observation, action, reward, self.current_episode_steps))
            # if len(self.local_buffer) >= local_buffer_size:  # TODO
            #     self.local_buffer.forward_to_replay()  # TODO
            self.global_replay.push((self._last_observation, action, reward, observation, done))
            if done:
                self._last_observation = self._reset()
            else:
                self._last_observation = observation

    def _reset(self):
        observation = self.env.reset()
        self.total_episodes += 1
        self.current_episode_steps = 0
        return observation

    def _step(self, observation):
        observation = to_gpu_or_npfloat32([observation], device=self.network._device_id)
        with chainer.no_backprop_mode():
            q_values = to_cpu(self.network(observation)).array[0]
        action = self.policy(q_values, self.total_steps)
        new_observation, reward, done, info = self.env.step(action)
        self.total_steps += 1
        self.current_episode_steps += 1
        return q_values, action, new_observation, reward, done, info
