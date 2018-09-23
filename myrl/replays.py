import logging
from collections import namedtuple
import sys

import numpy as np

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state_int', 'action', 'reward', 'done'])  # always store states as dtype==uint8.


def build_replay(replay_config, gamma=None, multi_step_n=None):
    Replay = getattr(sys.modules[__name__], replay_config['class'])
    kwargs = {'limit': replay_config['limit']}
    if issubclass(Replay, MultiStepReplay):
        kwargs['gamma'] = gamma
        kwargs['multi_step_n'] = multi_step_n
    replay = Replay(**kwargs)
    logger.info(f'built replay {replay}.')
    return replay


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

    def sample(self, size):
        if self.full:
            end = self.limit
        else:
            end = len(self) - 1
        # `np.random.randint` is 5x faster than `np.random.choice` or `random.choices`.
        idxs = np.random.randint(end, size=size)
        taboo = self.limit - 1 if self.head == 0 else self.head - 1  # current head points to the *next* index # FIXME: modulo is simpler
        exps_with_next = []
        for i in range(len(idxs)):
            while idxs[i] == taboo:
                idxs[i] = np.random.randint(end)
            idx = idxs[i]
            experience = self.replay[idx]
            next_experience = self.replay[(idx + 1) % self.limit]
            exps_with_next.append((
                experience.state_int, experience.action, experience.reward, experience.done, next_experience.state_int))
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
        return f'<{self.__class__.__name__}(limit={self.limit})>'


class MultiStepReplay(VanillaReplay):
    def __init__(self, limit=1_000_000, gamma=0.99, multi_step_n=1):
        self.limit = limit
        self.gamma = gamma
        self.multi_step_n = multi_step_n  # 1 is ordinal Q-learning

        self.head = 0
        self.replay = [None] * self.limit
        self.full = False

    def sample(self, size):
        if self.full:
            end = self.limit
        else:
            end = len(self) - 1
        taboos = []
        for i in range(1, self.multi_step_n + 1):
            idx = (self.head - i) % self.limit  # current head points to the *next* index
            if self.replay[idx].done:
                break
            taboos.append(idx)

        exps_with_next = []
        # `np.random.randint` is 5x faster than `np.random.choice` or `random.choices`.
        idxs = np.random.randint(end, size=size)
        for i in range(len(idxs)):
            while idxs[i] in taboos:
                idxs[i] = np.random.randint(end)
            idx = idxs[i]

            # accumulate n-step reward
            reward = 0
            for t in range(self.multi_step_n):
                target_idx = (idx + t) % self.limit
                target_exp = self.replay[target_idx]
                reward += (self.gamma ** t) * target_exp.reward
                if target_exp.done:
                    next_exp = target_exp
                    break
            else:  # this block runs only when for-loop finished without `break`.
                next_exp = self.replay[(target_idx + 1) % self.limit]

            start_exp = self.replay[idx]
            # Case: target_exp.done = False (for-loop is not breaked)
            #  EXP_i+0    EXP_i+1    EXP_i+2    ...    EXP_i+n-1    EXP_i+n
            #  ^^^^^^^                                 ^^^^^^^^^    ^^^^^^^
            # start_exp                                target_exp   next_exp
            #
            # Case: target_exp.done = True (for-loop is breaked)
            #  EXP_i+0    EXP_i+1    EXP_i+2    ...    EXP_i+n-1
            #  ^^^^^^^                                 ^^^^^^^^^
            # start_exp                           target_exp/next_exp
            # , where `n-1` is `self.multi_step_n - 1` or the terminal state index found during accumulating reward.
            # NOTE: When `target_exp.done` is True, `next_exp.state_int` is same as the one of `target_exp`.
            # However this is not a problem because next state is not used to calculate the terminal-state's Q-value target signal.
            exps_with_next.append((start_exp.state_int, start_exp.action, reward, target_exp.done, next_exp.state_int))
        return exps_with_next

    def __repr__(self):
        return f'<{self.__class__.__name__}(limit={self.limit}, gamma={self.gamma}, multi_step_n={self.multi_step_n})>'
