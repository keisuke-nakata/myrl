import logging
import os
from multiprocessing import Process

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import FittedQLearner
from ..policies import EpsilonGreedy, Greedy
from ..replays import RedisReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env


logger = logging.getLogger(__name__)


class VanillaDQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.config = config
        self.env_id = env_id

        self.env = setup_env(env_id, clip=False)
        self.clipped_env = setup_env(env_id, clip=True)

        self.n_actions = self.env.action_space.n

        self.network = VanillaCNN(self.n_actions)  # will be shared among actor and learner
        if device >= 0:
            self.network.to_gpu(device)
        logger.info(f'built an agent with device {device}.')

        self.policy = EpsilonGreedy(action_space=self.clipped_env.action_space, **self.config['policy']['params'])
        self.greedy_policy = Greedy(action_space=self.env.action_space)

        n_action_repeat = self.config['actor']['n_action_repeat']
        self.render_episode_freq = self.config['history']['render_episode_freq']

        # self.replay = VanillaReplay(limit=self.config['replay']['limit'] // n_action_repeat)
        # self.dummy_replay = VanillaReplay(limit=10)
        self.obs_preprocessor = AtariPreprocessor()
        self.greedy_actor = QActor(
            env=self.env,
            network=self.network,
            policy=self.greedy_policy,
            n_action_repeat=1,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=(0, 0),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=os.path.join(self.config['result_dir'], 'greedy'))
        self.actor = QActor(
            env=self.clipped_env,
            network=self.network,
            policy=self.policy,
            n_action_repeat=n_action_repeat,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=tuple(self.config['actor']['n_random_actions_at_reset']),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=self.config['result_dir'])
        optimizer = getattr(optimizers, self.config['optimizer']['optimizer'])(**self.config['optimizer']['params'])
        self.learner = FittedQLearner(
            network=self.network,
            optimizer=optimizer,
            gamma=self.config['learner']['gamma'],
            target_network_update_freq=self.config['learner']['target_network_update_freq'],
            logging_freq=self.config['learner']['logging_freq'])

        logger.debug(f'build a VanillaDQNAgent, {self.clipped_env}')

    def train(self):
        """同期更新なので単なるループでOK"""
        total_steps = 0
        total_episodes = 0

        # warmup
        for exps in self.actor.warmup_act(self.config['n_warmup_steps']):
            self.replay.mpush(exps)

        episode_losses = []
        episode_td_errors = []
        while total_steps < self.config['n_total_steps'] or total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {total_episodes}, step {total_steps}')
            # actor
            experience, done = self.actor.act()
            self.replay.push(experience)

            # learner
            experiences = self.replay.sample(size=self.config['learner']['batch_size'])
            loss, td_error = self.learner.learn(experiences)
            episode_losses.append(loss)
            episode_td_errors.append(td_error)

            # sync-ing parameters is not necessary because the `network` instance is shared in actor and leaner
            # self.actor.load_parameters(self.learner.dump_parameters())

            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes

            # dump episode and greedy actor's play
            if done and total_episodes % self.render_episode_freq == 0:
                self.actor.dump_episode()
                logger.info('greedy actor is playing...')
                self.greedy_actor.total_episodes = total_episodes - 1
                self.greedy_actor.total_steps = total_steps
                self.greedy_actor.act_episode()
                self.greedy_actor.dump_episode()
                logger.info('greedy actor is playing... done.')

            if done:
                logger.info(
                    f'episode {total_episodes}: '
                    f'memory length {len(self.replay):,}, '
                    f'avg loss {sum(episode_losses) / len(episode_losses):.4}, '
                    f'avg TD error {sum(episode_td_errors) / len(episode_td_errors):.4}')
                episode_losses = []
                episode_td_errors = []
                # FIXME: loss, td_error をファイルにも書き出す
