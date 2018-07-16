import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


class DoNothingPreprocessor:
    def __call__(self, observation, last_observation):
        return observation


class MaxWithPrevious:
    """This was necessary to remove flickering that is present in games where some objects appear
    only in even frames while other objects appear only in odd frames,
    an artefact caused by the limited number of sprites Atari 2600 can display at once.
    """
    def __call__(self, observation, last_observation):
        if last_observation is not None:
            ret = np.maximum(observation, last_observation)
            return ret
        return observation


class GrayScale:
    def __call__(self, observation, last_observation):
        ret = rgb2gray(observation)  # will be ((210, 160), dtype=np.float64)
        return ret


class Rescale:
    def __call__(self, observation, last_observation):
        ret = resize(observation, output_shape=(110, 84, 1))  # 210x160 -> 110x84x1
        return ret[13:-13, :, :]  # crop center (84, 84, 1)


class Float32:
    def __call__(self, observation, last_observation):
        ret = np.asarray(observation, dtype=np.float32)
        return ret

#
# class UInt8:
#     def __call__(self, observation, last_observation):
#         ret = (observation * 255).astype(np.uint8)
#         return ret
# 

class AtariPreprocessor:
    def __init__(self):
        self.max_with_previous = MaxWithPrevious()
        self.grayscale = GrayScale()
        self.rescale = Rescale()
        self.float32 = Float32()

    def __call__(self, observation, last_observation):
        observation = self.max_with_previous(observation, last_observation)
        observation = self.grayscale(observation, last_observation)
        observation = self.rescale(observation, last_observation)
        observation = self.float32(observation, last_observation)
        return observation
