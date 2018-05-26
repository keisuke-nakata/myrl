import numpy as np
from scipy.misc import imresize


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


class GrayScaleAndRescale:
    def __call__(self, observation, last_observation):
        raise ValueError('Gym の obs の RGB がどう規格化されているのか確認  あとサイズも確認 本当に 210x160なのか？縦横ひっくり返ってないか？')
        ret = imresize(observation, size=(110, 84), interp='bilinear', mode='L')
        # ↑ size の指定順大丈夫か？mode='L'だとintになって返ってきたりしないか？
        return ret[13:-13, :, :]  # crop center (84, 84)


class AtraiProprocess:
    def __init__(self):
        self.max_with_previous = MaxWithPrevious()
        self.grayscale_and_rescale = GrayScaleAndRescale()

    def __call__(self, observation, last_observation):
        observation = self.max_with_previous(observation, last_observation)
        observation = self.grayscale_and_rescale(observation, last_observation)
        return observation
