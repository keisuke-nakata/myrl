import numpy as np
import gym


def setup_env(env_id, clip=False, suddendeath=False, life_episode=False):
    env = gym.make(env_id)
    env._max_episode_steps = 10000
    if clip:
        env = RewardClippingWrapper(env)
    if suddendeath:
        env = SuddenDeathWrapper(env)
    if life_episode:
        env = LifeEpisodeWrapper(env)
    return env


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class SuddenDeathWrapper(gym.Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['original_done'] = done
        sudden_death_done = reward != 0
        return observation, reward, sudden_death_done, info


class LifeEpisodeWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        self.lives = -1
        self.done = False
        self.lost_life = False
        super().__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        lives = info['ale.lives']
        self.lost_life = lives < self.lives
        self.lives = lives
        self.done = done
        return observation, reward, self.lost_life or done, info

    def reset(self, **kwargs):
        if not self.done and self.lost_life:
            observation, reward, done, info = self.env.step(0)
            assert self.lives == info['ale.lives']
            assert not self.done
            self.lives = info['ale.lives']
            self.done = False
            self.lost_life = False
        else:
            observation = self.env.reset()
            self.lives = -1
            self.done = False
            self.lost_life = False
        return observation
