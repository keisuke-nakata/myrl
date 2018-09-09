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
    """DeepMind says they extract the Y channel a.k.a. luminance in their Nature version paper.
    NIPS version paper just says they used grayscale.
    The doc of `skimage.color.rgb2gray` says it extracts the luminance, so we use it here.
    https://github.com/deepmind/dqn/blob/master/dqn/Scale.lua#L24
    """
    def __call__(self, observation):
        ret = rgb2gray(observation)  # will be ((210, 160), dtype=np.float64), scales [0, 1]
        return ret


class Rescale:
    """DeepMind says they "rescaling and cropping" an 84x84 region in their NIPS version paper,
    however they just say "rescaling" in the Nature version.
    We adopt the rescaling, since almost all implementations seems to use rescaling as preprocessing.
    https://github.com/deepmind/dqn/blob/master/dqn/Scale.lua#L25
    """
    def __call__(self, observation):
        ret = resize(observation, output_shape=(84, 84, 1), mode='constant', anti_aliasing=True, order=1)  # 210x160 -> 84x84x1, scales [0, 1]. order=1 means "bilinear".
        # ret = ret * 2 - 1.0  # scales [-1, 1]
        return ret


class Float32:
    def __call__(self, observation):
        ret = np.asarray(observation, dtype=np.float32)
        return ret


class UInt8:
    def __call__(self, observation):
        ret = (observation * 255).astype(np.uint8)
        return ret


class AtariPreprocessor:
    def __init__(self):
        """actor -> memory"""
        self.max_with_previous = MaxWithPrevious()
        self.grayscale = GrayScale()
        self.rescale = Rescale()
        # self.float32 = Float32()
        # self.uint8 = UInt8()

    def __call__(self, observation, last_observation):
        observation = self.max_with_previous(observation, last_observation)
        observation = self.grayscale(observation)
        observation = self.rescale(observation)
        # observation = self.float32(observation)
        # observation = self.uint8(observation)
        return observation

    def __repr__(self):
        return f'<{self.__class__.__name__}>'
