# from multiprocessing.managers import BaseManager
# from queue import Queue
# queue = Queue()
# class QueueManager(BaseManager): pass
# QueueManager.register('get_queue', callable=lambda:queue)
# m = QueueManager(address=('', 50000), authkey=b'abracadabra')
# s = m.get_server()
# s.serve_forever()

import time
import threading
from multiprocessing import Process
from multiprocessing.managers import BaseManager

# from queue import Queue
# queue = Queue()
# class QueueManager(BaseManager): pass
# QueueManager.register('get_queue', callable=lambda:queue)
class ListManager(BaseManager):
    def setup(self):
        self.lst = []

# QueueManager.register('get_queue', callable=lambda:queue)
ListManager.register('get_lst', callable=
m = QueueManager(address=('', 50000), authkey=b'abracadabra')
s = m.get_server()
stop_timer = threading.Timer(1, lambda:s.stop_event.set())
QueueManager.register('stop', callable=lambda:stop_timer.start())


def f(s):
    s.serve_forever()

p = Process(target=f, args=(s, ))
p.start()

m = QueueManager(address=('', 50000), authkey=b'abracadabra')
m.connect()
queue = m.get_queue()
# queue.put('hahaha')


def put(queue):
    for i in range(3):
        queue.put(i)
        time.sleep(1)

p2 = Process(target=put, args=(queue, ))
p2.start()


def put2(queue):
    for i in range(3):
        queue.put(i * 1000)
        time.sleep(0.8)

p4 = Process(target=put2, args=(queue, ))
p4.start()


m2 = QueueManager(address=('', 50000), authkey=b'abracadabra')
m2.connect()
queue2 = m2.get_queue()
# print(queue2.get())


def get(queue):
    for _ in range(6):
        print(queue.get())


p3 = Process(target=get, args=(queue2, ))
p3.start()

p2.join()
p3.join()
p4.join()



# queue.put('hi')
# print(queue2.get())
# queue.put('fu')
# queue.put('he')
# print(queue2.get())
# print(queue2.get())

# s.shutdown()
m.stop()

print('before join')
p.join()
print('after join')
# del p
# del m
# del s  # このをしないと、このプロセス自体が終わらない (謎)

print('enjoy')
import os
os._exit(0)
