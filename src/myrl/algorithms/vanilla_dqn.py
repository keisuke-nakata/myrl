import logging

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import FittedQLearner
from ..policies import EpsilonGreedy
from ..replays import VanillaReplay
from ..preprocessors import AtariPreprocessor


logger = logging.getLogger(__name__)


class VanillaDQNAgent:
    def build(self, config, env):
        self.config = config
        self.env = env

        self.n_actions = env.action_space.n

        self.network = VanillaCNN(self.n_actions)  # will be shared among actor and learner
        self.policy = EpsilonGreedy(action_space=env.action_space, **self.config['policy']['params'])

        n_action_repeat = self.config['actor']['n_action_repeat']

        self.replay = VanillaReplay(limit=self.config['replay']['limit'] // n_action_repeat)
        self.obs_preprocessor = AtariPreprocessor()
        self.actor = QActor(
            env=env,
            network=self.network,
            policy=self.policy,
            global_replay=self.replay,
            n_action_repeat=n_action_repeat,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=tuple(self.config['actor']['n_random_actions_at_reset']),
            n_stack_frames=self.config['n_stack_frames'])
        optimizer = getattr(optimizers, self.config['optimizer']['optimizer'])(**self.config['optimizer']['params'])
        self.learner = FittedQLearner(
            network=self.network,
            optimizer=optimizer,
            gamma=self.config['learner']['gamma'],
            target_network_update_freq=self.config['learner']['target_network_update_freq'])
        # self.renderer =

        logger.debug('build a VanillaDQNAgent, {}'.format(env))

    def train(self):
        """同期更新なので単なるループでOK"""
        total_steps = 0
        total_episodes = 0

        self.actor.warmup(self.config['n_warmup_steps'])

        while total_steps < self.config['n_total_steps']:
            logger.debug('episode {}, step {}'.format(total_episodes, total_steps))
            self.actor.act()
            experiences = self.replay.sample(size=self.config['learner']['batch_size'])
            self.learner.learn(experiences)

            # sync-ing parameters is not necessary because the `network` instance is shared in actor and leaner
            # self.actor.load_parameters(self.learner.dump_parameters())

            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes
