from multiprocessing import Process

import redis


# class RedisQueue(object):
#     """Simple Queue with Redis Backend"""
#     def __init__(self, name, namespace='queue', **redis_kwargs):
#         """The default connection parameters are: host='localhost', port=6379, db=0"""
#         self.__db = redis.Redis(**redis_kwargs)
#         self.key = '%s:%s' % (namespace, name)
#
#     def qsize(self):
#         """Return the approximate size of the queue."""
#         return self.__db.llen(self.key)
#
#     def empty(self):
#         """Return True if the queue is empty, False otherwise."""
#         return self.qsize() == 0
#
#     def put(self, item):
#         """Put item into the queue."""
#         self.__db.rpush(self.key, item)
#
#     def get(self, block=True, timeout=None):
#         """Remove and return an item from the queue.
#
#         If optional args block is true and timeout is None (the default), block
#         if necessary until an item is available."""
#         if block:
#             item = self.__db.blpop(self.key, timeout=timeout)
#         else:
#             item = self.__db.lpop(self.key)
#
#         if item:
#             item = item[1]
#         return item
#
#     def get_nowait(self):
#         """Equivalent to get(False)."""
#         return self.get(False)
#
#     def push(self, item):
#
#
#     def __len__(self):
#         return self.__db.scard(self.key)


import time
import numpy as np
import pickle


experience = (self.previous_state, np.int32(action), reward, self.state, done)


class Learner:
    def learn(self):
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        for _ in range(30):
            # val = r.lrange('mylist', -3, -1)
            val = r.lrange('mylist', 0, -1)
            print(f'learn {val}')
            time.sleep(1)


class Actor:
    def act(self):
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        i = 0
        for _ in range(30):
            previous_state = np.
            r.lpush('mylist', i)
            r.ltrim('mylist', 0, 9)
            print(f'act {i}')
            time.sleep(0.5)
            i += 1


class Trainer:

    def build(self):
        self.learner = Learner()
        self.actor = Actor()

    def train(self):
        proc_learner = Process(target=self.learner.learn)
        proc_actor = Process(target=self.actor.act)

        proc_learner.start()
        proc_actor.start()

        proc_learner.join()
        proc_actor.join()


def main():
    trainer = Trainer()
    trainer.build()
    trainer.train()
    # r = redis.StrictRedis(host='localhost', port=6379, db=0)

# def main():
#     r = redis.StrictRedis(host='localhost', port=6379, db=0)
#     r.lpush('mylist', 'aaa')
#     r.lpush('mylist', 'bbb')
#     r.llen(name



if __name__ == '__main__':
    main()
