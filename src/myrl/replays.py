# import random
import pickle
import logging
import ctypes as C
import base64
import multiprocessing
from collections import namedtuple

import redis
import numpy as np

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
# ExperienceIntState = namedtuple('ExperienceIntState', ['state', 'action', 'reward', 'done'])


class VanillaReplay:
    def __init__(self, limit=1_000_000):
        self.limit = limit

        self.head = 0
        self.replay = [None] * self.limit
        self.full = False

    def push(self, experience):
        # state_int = np.round(experience.state * 255).astype(np.uint8)  # [0.0, 1.0] -> [0, 255]
        # state_int = np.round((experience.state + 1) * 127.5).astype(np.uint8)  # [-1.0, 1.0] -> [0, 255]
        # experience_int_state = ExperienceIntState(state_int, experience.action, experience.reward, experience.done)
        # self.replay[self.head] = experience_int_state
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
            experience_int_state = self.replay[idx]
            next_experience_int_state = self.replay[(idx + 1) % self.limit]
            # state = (experience_int_state.state / 255).astype(np.float32)  # [0, 255] -> [0.0, 1.0]
            # next_state = (next_experience_int_state.state / 255).astype(np.float32)  # [0, 255] -> [0.0, 1.0]
            # state = (experience_int_state.state / 127.5 - 1.0).astype(np.float32)  # [0, 255] -> [-1.0, 1.0]
            # next_state = (next_experience_int_state.state / 127.5 - 1.0).astype(np.float32)  # [0, 255] -> [-1.0, 1.0]
            # exps_with_next.append((state, experience_int_state.action, experience_int_state.reward, experience_int_state.done, next_state))
            exps_with_next.append((experience_int_state.state, experience_int_state.action, experience_int_state.reward, experience_int_state.done, next_experience_int_state.state))
        return exps_with_next

    def batch_sample(self, size):
        experiences_with_next = self.sample(size)
        states, actions, rewards, dones, next_states = zip(*experiences_with_next)

        # batch_state = np.array(states, dtype=np.float32)
        batch_state = np.array(states, dtype=np.uint8)
        batch_action = np.array(actions, dtype=np.int8)
        batch_reward = np.array(rewards, dtype=np.float32)
        batch_done = np.array(dones, dtype=np.int8)
        # batch_next_state = np.array(next_states, dtype=np.float32)
        batch_next_state = np.array(next_states, dtype=np.uint8)

        return batch_state, batch_action, batch_reward, batch_done, batch_next_state

    def __len__(self):
        return self.head if not self.full else self.limit


class RedisReplay:
    def __init__(self, limit=1_000_000 // 4, host='localhost', port=6379, db=0, listname='replay'):
        # limit: divided by action repeat
        self.limit = limit
        self.listname = listname
        self._redis = redis.StrictRedis(host=host, port=port, db=db)
        try:
            self._redis.ping()
        except redis.exceptions.ConnectionError:
            logger.exception('Failed to connect Redis server. Check host/port/status of redis server.')
            raise
        else:
            logger.info('success to ping redis server')

    def push(self, experience):
        self._redis.lpush(self.listname, pickle.dumps(experience))
        self._redis.ltrim(self.listname, 0, self.limit)

    def mpush(self, experiences):
        self._redis.lpush(self.listname, *[pickle.dumps(e) for e in experiences])
        self._redis.ltrim(self.listname, 0, self.limit)

    def sample(self, size):
        idxs = np.random.randint(low=0, high=len(self), size=size)
        exps = [pickle.loads(self._redis.lindex(self.listname, idx)) for idx in idxs]
        return exps

    def __len__(self):
        return self._redis.llen(self.listname)

    def flush(self):
        return self._redis.flushall()


class SharedReplay:
    class AtariExperience(C.Structure):
        _shape = (4, 84, 84)  # FIXME: magic number
        _dtype = np.uint8  # FIXME: magic number
        _size = len(base64.b64encode(np.empty(_shape, dtype=_dtype)))
        _fields_ = [
            ('state', C.c_char * _size),
            ('action', C.c_int),
            ('reward', C.c_double),
            ('next_state', C.c_char * _size),
            ('done', C.c_bool), ]

    def __init__(self, limit=1_000_000 // 4):
        # limit: divided by action repeat
        self.limit = limit
        self._dtype = self.AtariExperience._dtype
        self._shape = self.AtariExperience._shape
        self._len = 0

        self._queue = None

    def _push(self, experience, memory, head):
        """This private method does not ensure concurrent update.
        The caller of this method should ensure that."""
        state, action, reward, next_state, done = experience
        data = (
            base64.b64encode(np.ascontiguousarray(state)),
            int(action),
            float(reward),
            base64.b64encode(np.ascontiguousarray(next_state)),
            bool(done), )
        memory[head.value] = data
        if self._len < self.limit:
            self._len = head.value + 1
            if self.is_filled:
                logger.info('Memory is filled.')
        head.value = (head.value + 1) % self.limit

    def mpush(self, experiences, lock, memory, head):
        with lock:
            for exp in experiences:
                self._push(exp, memory, head)

    def start_prefetch(self, size, lock, memory, head, n_prefetches=2):
        self._queue = multiprocessing.Queue(n_prefetches)
        self._prefetch_process = multiprocessing.Process(target=self._prefetch, args=(size, lock, memory, head))
        self._prefetch_process.start()

    def _prefetch(self, size, lock, memory, head):
        while True:
            with lock:
                high = self.limit if self.is_filled else head.value
                idxs = np.random.randint(0, high=high, size=size)
                data = [
                    (
                        np.frombuffer(base64.decodebytes(memory[idx].state), dtype=self._dtype).reshape(self._shape),
                        memory[idx].action,
                        memory[idx].reward,
                        np.frombuffer(base64.decodebytes(memory[idx].next_state), dtype=self._dtype).reshape(self._shape),
                        memory[idx].done,
                    ) for idx in idxs]
                self._queue.put(data)

    def sample(self):
        if self._queue is None:
            raise ValueError('Prefetch process has not started. Call `start_prefetch()` before `sample()`.')
        return self._queue.get()

    def __len__(self):
        return self._len

    @property
    def is_filled(self):
        return len(self) == self.limit
