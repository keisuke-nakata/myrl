import gym

from agents import BaseAgent
from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import QLearner
from ..policies import EpsilonGreedy
from ..replays import VanillaReplay


class VanillaDQNAgent(BaseAgent):
    def build(self, config):
        self.config = config
        self.network = VanillaCNN(n_actions)

        env = gym.make(config['env']['env'])
        n_actions = env.action_space.n

        self.replay = VanillaReplay()
        self.actor = QActor(
            env=env,
            network=self.network,
            policy=EpsilonGreedy(action_space=env.action_space, **self.config['policy']),
            global_replay=self.replay)
        self.learner = QLearner(network=self.network, gamma=self.config['learner']['gamma'])

    def train(self):
        """同期更新なので単なるループでOK"""
        total_steps = 0
        total_episodes = 0

        while total_steps < self.config['warmup_steps']:
            self.actor.act()
            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes

        while total_steps < self.config['n_total_steps']:
            self.actor.act()
            experiences = self.replay.sample(size=self.config['learner']['batch_size'])
            self.learner.learn(experiences)

            # sync-ing parameters is not necessary because the `network` instance is shared in actor and leaner
            # self.actor.load_parameters(self.learner.dump_parameters())

            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes
