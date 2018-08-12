import logging
import os
import csv

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import Actor
from ..learners import FittedQLearner
from ..policies import QPolicy, LinearAnnealEpsilonGreedyExplorer, GreedyExplorer
from ..replays import VanillaReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env

logger = logging.getLogger(__name__)


"""
用語の決まり：
frame: もとのゲームにおけるフレームのこと。FPS を出す際は、フレームスキップ (action_repeat) が入っていたとしても、そのぶんを含めて数えることとする
step: 何回行動を選択したか。つまり、step の回数は、frame の回数をフレームスキップ (action_repeat) で割った値となる。
"""


class VanillaDQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.config = config
        self.env_id = env_id
        self.device = device

        self.n_actions = setup_env(env_id).action_space.n
        self.n_action_repeat = self.config['actor']['n_action_repeat']
        self.render_episode_freq = self.config['history']['render_episode_freq']

        self.network = VanillaCNN(self.n_actions)
        if self.device >= 0:
            self.network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')

        self.policy = QPolicy(self.networ)
        # TODO: reward clipping を忘れずに！！
        self.actor = Actor(
            setup_env(env_id, clip=False), self.policy,
            LinearAnnealEpsilonGreedyExplorer(self.config['explorer']['params']),
            AtariPreprocessor(),
            n_noop_at_reset=(0, 30), n_stack_frames=4, n_action_repeat=4)
        self.test_actor = Actor(
            setup_env(env_id, clip=False), self.policy,
            GreedyExplorer(),
            AtariPreprocessor(),
            n_noop_at_reset=(0, 0), n_stack_frames=4, n_action_repeat=4)

        # self.actor = self._build_actor(self.network, env_id, greedy=False)
        # self.greedy_actor = self._build_actor(self.network, env_id, greedy=True)
        self.learner = self._build_learner(self.network, self.config['learner'])

        self.replay = VanillaReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)

    # def _build_actor(self, network, env_id, greedy=False):
    #     if greedy:
    #         env = setup_env(env_id, clip=False)
    #         policy = Greedy(action_space=env.action_space)
    #         n_action_repeat = 1
    #         n_random_actions_at_reset = (0, 0)
    #         result_dir = os.path.join(self.config['result_dir'], 'greedy')
    #     else:
    #         env = setup_env(env_id, clip=True)
    #         policy = EpsilonGreedy(action_space=env.action_space, **self.config['policy']['params'])
    #         n_action_repeat = self.n_action_repeat
    #         n_random_actions_at_reset = tuple(self.config['actor']['n_random_actions_at_reset'])
    #         result_dir = os.path.join(self.config['result_dir'], 'actor')
    #     obs_preprocessor = AtariPreprocessor()
    #
    #     actor = QActor(
    #         env=env,
    #         network=network,
    #         policy=policy,
    #         n_action_repeat=n_action_repeat,
    #         obs_preprocessor=obs_preprocessor,
    #         n_random_actions_at_reset=n_random_actions_at_reset,
    #         n_stack_frames=self.config['n_stack_frames'],
    #         result_dir=result_dir)
    #
    #     logger.info(f'built {"a greedy " if greedy else "an "}actor.')
    #     return actor

    def _build_learner(self, network, learner_config):
        optimizer = getattr(optimizers, learner_config['optimizer']['class'])(**learner_config['optimizer']['params'])

        learner = FittedQLearner(
            network=network,
            optimizer=optimizer,
            gamma=learner_config['gamma'],
            target_network_update_freq=learner_config['target_network_update_freq'], )
        logger.info(f'built a learner.')
        return learner

    # def warmup(self, n_warmup_steps):
    #     for exps in self.actor.warmup_act(n_warmup_steps):
    #         if len(exps) > 0:
    #             self.replay.mpush(exps)

    def train(self):
        # self.warmup(self.config['n_warmup_steps'])

        batch_size = self.config['learner']['batch_size']
        learner_logging_freq = self.config['learner']['logging_freq']
        learner_result_path = os.path.join(self.config['result_dir'], 'learner_history.csv')
        with open(learner_result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'loss', 'td_error'])

        n_steps = 1
        n_episodes = 1
        n_warmup_steps = self.config['n_warmup_steps']
        updates = []
        losses = []
        td_errors = []
        if n_warmup_steps > 0:
            logger.info(f'warming up {n_warmup_steps} steps...')
        while n_steps <= self.config['n_total_steps'] and n_episodes <= self.config['n_total_episodes']:
            logger.debug(f'episode {self.actor.total_episodes}, step {self.actor.total_steps}')

            # actor
            state, action, reward, next_state, done, is_random = self.actor.act()
            experience = (state, action, reward, next_state, done)
            self.replay.push(experience)

            # learner
            if n_steps >= n_warmup_steps:
                experiences = self.replay.sample(batch_size)
                loss, td_error = self.learner.learn(experiences)
                updates.append(self.learner.total_updates)
                losses.append(loss)
                td_errors.append(td_error)
            if learner_logging_freq != 0 and self.learner.total_updates % learner_logging_freq == 0:
                with open(learner_result_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(list(zip(*[updates, losses, td_errors])))
                self.learner.timer.lap()
                n = len(losses)
                logger.info(
                    f'finished {n} updates with avg loss {sum(losses) / n:.5f}, td_error {sum(td_errors) / n:.5f} in {self.learner.timer.laptime_str} '
                    f'({n / self.learner.timer.laptime:.2f} batches per seconds) '
                    f'(total_updates {self.learner.total_updates:,}, total_time {self.learner.timer.elapsed_str})')
                updates = []
                losses = []
                td_errors = []

            # dump episode and greedy actor's play
            if done and self.actor.total_episodes % self.render_episode_freq == 0:
                logger.info(f'Memory length at episode {self.actor.total_episodes}: {len(self.replay)}')
                self.actor.dump_episode()

                logger.info('greedy actor is playing...')
                self.greedy_actor.total_episodes = self.actor.total_episodes - 1
                self.greedy_actor.total_steps = self.actor.total_steps
                self.greedy_actor.act_episode()
                self.greedy_actor.dump_episode()
                logger.info('greedy actor is playing... done.')
