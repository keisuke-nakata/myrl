import gym


class SuddenDeathWrapper(gym.Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['original_done'] = done
        sudden_death_done = reward != 0
        return observation, reward, sudden_death_done, info
