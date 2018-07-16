from multiprocessing import Process, Manager
from multiprocessing.managers import SyncManager
import time


def f(d, l):
    # d[1] = '1'
    # d['2'] = 2
    # d[0.25] = None
    # l.reverse()
    for i in range(10):
        l.append(i)
        print(f'I am f. append {i}')
        # time.sleep(1)


def g(d, l):
    # d[3] = '3'
    # d[10.0] = ('yes', 'we', 'can')
    # l.append(100)
    for i in range(10):
        l.append(i * 10)
        print(f'I am g. append {i * 10}')
        # time.sleep(1)


if __name__ == '__main__':
    manager = SyncManager()
    manager.__enter__()
    d = manager.dict()
    l = manager.list()

    p = Process(target=f, args=(d, l))
    p2 = Process(target=g, args=(d, l))
    p.start()
    p2.start()
    p.join()
    p2.join()

    print(d)
    print(l)
    manager.__exit__(None, None, None)
    # with Manager() as manager:
    #     d = manager.dict()
    #     l = manager.list()
    #
    #     p = Process(target=f, args=(d, l))
    #     p2 = Process(target=g, args=(d, l))
    #     p.start()
    #     p2.start()
    #     p.join()
    #     p2.join()
    #
    #     print(d)
    #     print(l)
