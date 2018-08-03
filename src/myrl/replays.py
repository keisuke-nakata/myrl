import random
import pickle
import logging
import ctypes as C
import base64
import multiprocessing

import redis
import numpy as np

logger = logging.getLogger(__name__)


class VanillaReplay:
    def __init__(self, limit=1_000_000 // 4):
        # limit: divided by action repeat
        self.limit = limit
        self.experiences = []

    def push(self, experience):
        self.experiences.append(experience)
        self.experiences = self.experiences[-self.limit:]

    def mpush(self, experiences):
        self.experiences.extend(experiences)
        self.experiences = self.experiences[-self.limit:]

    def sample(self, size):
        return random.choices(self.experiences, k=size)

    def __len__(self):
        return len(self.experiences)


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
