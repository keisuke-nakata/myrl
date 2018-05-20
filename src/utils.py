import numpy as np

from chainer.cuda import to_gpu


def to_gpu_or_npfloat32(array, device):
    ret = np.array(array, dtype=np.float32)
    if device is not None:  # gpu mode
        ret = to_gpu(ret)
    return ret
