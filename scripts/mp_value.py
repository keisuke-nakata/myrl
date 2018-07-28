import time
import multiprocessing
from multiprocessing.sharedctypes import Value, Array
import ctypes as C
import base64

import numpy as np


class Hoge(C.Structure):
    _fields_ = [
        ('x', C.c_char * 96),
        ('x2', C.c_char * 8),
        ('y', C.c_int),
        ('z', C.c_double)]


def append(array, head):
    for i in range(15):
        data = (
            base64.b64encode(np.ones((3, 3)) * i), base64.b64encode(np.ones((2, 2), dtype=np.uint8) * i), i, i / 10)
        array[head.value] = data
        print('append data {}'.format(data))
        with head.get_lock():
            head.value = (head.value + 1) % 10
        time.sleep(1)


def sample(array, head):
    time.sleep(3)
    for i in range(15):
        data = array[:2]
        print(data)
        for d in data:
            print(np.frombuffer(base64.decodebytes(d.x)))
            print(np.frombuffer(base64.decodebytes(d.x2), dtype=np.uint8))
            print(d.y)
            print(d.z)
        time.sleep(1)


if __name__ == '__main__':
    array = Array(Hoge, 10, lock=True)
    head = Value('i', 0, lock=True)

    a_proc = multiprocessing.Process(target=append, args=(array, head))
    s_proc = multiprocessing.Process(target=sample, args=(array, head))

    a_proc.start()
    s_proc.start()

    a_proc.join()
    s_proc.join()
