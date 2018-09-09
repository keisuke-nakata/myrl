# import random
import logging
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state_int', 'action', 'reward', 'done'])  # always store states as dtype==uint8.


class VanillaReplay:
    def __init__(self, limit=1_000_000):
        self.limit = limit

        self.head = 0
        self.replay = [None] * self.limit
        self.full = False

    def push(self, experience):
        self.replay[self.head] = experience
        self.head += 1
        if self.head >= self.limit:
            self.head = 0
            if not self.full:
                logger.info('filled replay memory.')
                self.full = True

    # def mpush(self, experiences):
    #     self.replay.extend(experiences)
    #     self.replay = self.replay[-self.limit:]

    def sample(self, size):
        if self.full:
            end = self.limit
        else:
            end = len(self) - 1
        idxs = np.random.randint(end, size=size)  # `np.random.randint` is 5x faster than `np.random.choice` or `random.choices`.
        taboo = self.limit - 1 if self.head == 0 else self.head - 1  # current head points to the *next* index
        exps_with_next = []
        for i in range(len(idxs)):
            while idxs[i] == taboo:
                idxs[i] = np.random.randint(end)
            idx = idxs[i]
            experience = self.replay[idx]
            next_experience = self.replay[(idx + 1) % self.limit]
            exps_with_next.append((experience.state_int, experience.action, experience.reward, experience.done, next_experience.state_int))
        return exps_with_next

    def batch_sample(self, size):
        experiences_with_next = self.sample(size)
        states_int, actions, rewards, dones, next_states_int = zip(*experiences_with_next)

        batch_state_int = np.array(states_int, dtype=np.uint8)
        batch_action = np.array(actions, dtype=np.int8)
        batch_reward = np.array(rewards, dtype=np.float32)
        batch_done = np.array(dones, dtype=np.int8)
        batch_next_state_int = np.array(next_states_int, dtype=np.uint8)

        return batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int

    def __len__(self):
        return self.head if not self.full else self.limit

    def __repr__(self):
        return f'<{self.__class__.__name__}(limit={self.limit}, gamma={self.gamma}, multi_step_n={self.multi_step_n})>'
