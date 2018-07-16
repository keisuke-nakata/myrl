import random
import pickle

import redis
import numpy as np


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
