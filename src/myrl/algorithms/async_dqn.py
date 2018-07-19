import logging
import os
import multiprocessing

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import FittedQLearner
from ..policies import EpsilonGreedy, Greedy
from ..replays import RedisReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env


logger = logging.getLogger(__name__)


class AsyncDQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.config = config
        self.env_id = env_id

        self.env = setup_env(env_id, clip=False)
        self.clipped_env = setup_env(env_id, clip=True)

        self.n_actions = self.env.action_space.n

        self.learner_network = VanillaCNN(self.n_actions)
        self.actor_network = self.learner_network.copy(mode='copy')
        if device >= 0:
            self.learner_network.to_gpu(device)
        logger.info(f'built an learner_network with device {device}.')

        self.network_dump_path = os.path.join(self.config['result_dir'], 'network.hdf5')

        self.policy = EpsilonGreedy(action_space=self.clipped_env.action_space, **self.config['policy']['params'])
        self.greedy_policy = Greedy(action_space=self.env.action_space)

        self.n_action_repeat = self.config['actor']['n_action_repeat']
        self.render_episode_freq = self.config['history']['render_episode_freq']

        self.replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)
        self.replay.flush()
        self.obs_preprocessor = AtariPreprocessor()
        self.greedy_actor = QActor(
            env=self.env,
            network=self.actor_network,
            policy=self.greedy_policy,
            n_action_repeat=1,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=(0, 0),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=os.path.join(self.config['result_dir'], 'greedy'),
            render_episode_freq=1)
        self.actor = QActor(
            env=self.clipped_env,
            network=self.actor_network,
            policy=self.policy,
            n_action_repeat=self.n_action_repeat,
            obs_preprocessor=self.obs_preprocessor,
            n_random_actions_at_reset=tuple(self.config['actor']['n_random_actions_at_reset']),
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=self.config['result_dir'],
            render_episode_freq=self.render_episode_freq)
        optimizer = getattr(optimizers, self.config['optimizer']['optimizer'])(**self.config['optimizer']['params'])
        self.learner = FittedQLearner(
            network=self.learner_network,
            optimizer=optimizer,
            gamma=self.config['learner']['gamma'],
            target_network_update_freq=self.config['learner']['target_network_update_freq'],
            logging_freq=self.config['learner']['logging_freq'])

        logger.debug(f'build a VanillaDQNAgent, {self.clipped_env}')

    def _act(self):
        total_steps = 0
        total_episodes = 0

        replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)

        while total_steps < self.config['n_total_steps'] or total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {total_episodes}, step {total_steps}')
            # actor
            # q_values, action, is_random, reward, current_step, done = self.actor.act()
            experience, done = self.actor.act()
            replay.push(experience)

            total_steps = self.actor.total_steps
            total_episodes = self.actor.total_episodes

            # greedy actor's play
            if done and total_episodes % self.render_episode_freq == 0:
                logger.info('greedy actor is playing...')
                self.greedy_actor.total_episodes = total_episodes - 1
                self.greedy_actor.total_steps = total_steps
                while True:
                    # greedy_q_values, greedy_action, greedy_is_random, greedy_reward, greedy_current_step, greedy_done = self.greedy_actor.act()
                    greedy_experience, greedy_done = self.greedy_actor.act()
                    if greedy_done:  # greedy_actor will render the episode automatically
                        break
                logger.info('greedy actor is playing... done.')
            if self.config['actor']['load_freq'] != 0 and total_steps % self.config['actor']['load_freq'] == 0:
                self.actor.load_parameters(self.network_dump_path)

    def _learn(self):
        replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)
        self.learner.dump_parameters(self.network_dump_path)  # initial parameter

        while True:
            experiences = replay.sample(size=self.config['learner']['batch_size'])
            self.learner.learn(experiences)
            if self.config['learner']['dump_freq'] != 0 and self.learner.total_updates % self.config['learner']['dump_freq'] == 0:
                self.learner.dump_parameters(self.network_dump_path)

    def train(self):
        # warmup
        for exps in self.actor.warmup_act(self.config['n_warmup_steps']):
            self.replay.mpush(exps)

        proc_learner = multiprocessing.Process(target=self._learn)
        proc_actor = multiprocessing.Process(target=self._act)

        proc_learner.start()
        proc_actor.start()

        proc_actor.join()
        proc_learner.terminate()
