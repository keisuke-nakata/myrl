import gym

from agents import BaseAgent
from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import QLearner
from ..policies import EpsilonGreedy  # TODO


class VanillaDQNAgent(BaseAgent):
    def build(self, config):
        self.config = config

        env = gym.make(config['env']['env'])

        self.actor = QActor(
            env=env,
            network=VanillaCNN(self.config['env']['n_actions']),
            policy=EpsilonGreedy(action_space=env.action_space, **self.config['policy']))
        self.learner = QLearner(network=VanillaCNN(self.config['env']['n_actions']))
        self.replay = Replay()  # TODO

    def train(self):
        """同期更新なので単なるループでOK"""
        total_steps = 0
        total_episodes = 0
        while self.actor.total_steps < self.config['n_total_steps']:
            steps, episodes = self.actor.act()
            self.learner.learn()

            total_steps += steps
            total_episodes += episodes
