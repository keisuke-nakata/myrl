class BaseActor:
    def __init__(self, env, network, policy):
        self.env = env
        self.network = network
        self.policy = policy

        self.local_buffer = Replay()
        self.total_steps = 0
        self.total_episodes = 0
        self.current_episode_steps = 0

    def act(self, local_buffer_size=1, n_steps=None):
        """

        Parameters
        ----------
        local_buffer_size : int, optional
            B
        n_steps : int, optional
            T

        Returns
        -------
        steps : int
            #steps (actions) the actor tried.
        episodes : int
            #episodes the actor tried. Un-terminated episodes will be included.
        """
        raise NotImplementedError


class QActor(BaseActor):
    def act(self, local_buffer_size=1, n_steps=1):
        if not hasattr(self, '_last_observation'):  # first interaction
            self._last_observation = self._reset()
        for step in range(n_steps):
            q_values, action, observation, reward, is_done, info = self._step(self._last_observation)
            self.local_buffer.push((self._last_observation, action, reward, self.current_episode_steps))
            if len(self.local_buffer) >= local_buffer_size:  # TODO
                self.local_buffer.forward_to_replay()  # TODO
            if is_done:
                self._last_observation = self._reset()
            else:
                self._last_observation = observation

    def _reset(self):
        observation = self.env.reset()
        self.total_episodes += 1
        self.current_episode_steps = 0
        return observation

    def _step(self, observation):
        q_values = self.network(observation)
        action = self.policy(q_values, self.total_steps)
        new_observation, reward, is_done, info = self.env.step(action)
        self.total_steps += 1
        self.current_episode_steps += 1
        return q_values, action, new_observation, reward, is_done, info
