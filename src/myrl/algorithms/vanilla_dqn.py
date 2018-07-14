import logging
import os

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import FittedQLearner
from ..policies import EpsilonGreedy, Greedy
from ..replays import VanillaReplay
from ..preprocessors import AtariPreprocessor


logger = logging.getLogger(__name__)


class VanillaDQNAgent:
    def build(self, config, env, device=-1):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.config = config
        self.env = env

        self.n_actions = env.action_space.n

        self.network = VanillaCNN(self.n_actions)  # will be shared among actor and learner
        if device >= 0:
            self.network.to_gpu(device)
        logger.info(f'built an agent with device {device}.')

        self.policy = EpsilonGreedy(action_space=env.action_space, **self.config['policy']['params'])
        self.greedy_policy = Greedy(action_space=env.action_space)

        n_action_repeat = self.config['actor']['n_action_repeat']
        self.render_episode_freq = self.config['history']['render_episode_freq']

        self.replay = VanillaReplay(limit=self.config['replay']['limit'] // n_action_repeat)
        self.dummy_replay = VanillaReplay(limit=10)
        self.obs_preprocessor = AtariPreprocessor()
        self.greedy_actor = QActor(
            env=env,
            network=self.network,
            policy=self.greedy_policy,
            global_replay=self.dummy_replay,
            n_action_repeat=1,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=(0, 0),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=os.path.join(self.config['result_dir'], 'greedy'),
            render_episode_freq=1)
        self.actor = QActor(
            env=env,
            network=self.network,
            policy=self.policy,
            global_replay=self.replay,
            n_action_repeat=n_action_repeat,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=tuple(self.config['actor']['n_random_actions_at_reset']),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=self.config['result_dir'],
            render_episode_freq=self.render_episode_freq)
        optimizer = getattr(optimizers, self.config['optimizer']['optimizer'])(**self.config['optimizer']['params'])
        self.learner = FittedQLearner(
            network=self.network,
            optimizer=optimizer,
            gamma=self.config['learner']['gamma'],
            target_network_update_freq=self.config['learner']['target_network_update_freq'])

        logger.debug(f'build a VanillaDQNAgent, {env}')

    def train(self):
        """同期更新なので単なるループでOK"""
        total_steps = 0
        total_episodes = 0

        self.actor.warmup(self.config['n_warmup_steps'])

        episode_losses = []
        episode_td_errors = []
        while total_steps < self.config['n_total_steps'] or total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {total_episodes}, step {total_steps}')
            q_values, action, is_random, reward, current_step, done = self.actor.act()
            experiences = self.replay.sample(size=self.config['learner']['batch_size'])

            loss, td_error = self.learner.learn(experiences)
            episode_losses.append(loss)
            episode_td_errors.append(td_error)

            # sync-ing parameters is not necessary because the `network` instance is shared in actor and leaner
            # self.actor.load_parameters(self.learner.dump_parameters())

            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes

            # greedy actor's play
            if done and total_episodes % self.render_episode_freq == 0:
                logger.info('greedy actor is playing...')
                self.greedy_actor.total_episodes = total_episodes - 1
                self.greedy_actor.total_steps = total_steps
                while True:
                    greedy_q_values, greedy_action, greedy_is_random, greedy_reward, greedy_current_step, greedy_done = self.greedy_actor.act()
                    if greedy_done:  # greedy_actor will render the episode automatically
                        break
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
