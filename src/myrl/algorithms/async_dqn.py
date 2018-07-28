import logging
import os
import multiprocessing
from multiprocessing.sharedctypes import Value, Array

from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import QActor
from ..learners import FittedQLearner
from ..policies import EpsilonGreedy, Greedy
from ..replays import SharedReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env
from ..utils import report_error

logger = logging.getLogger(__name__)


class AsyncDQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.config = config
        self.env_id = env_id
        self.device = device

        self._env = setup_env(env_id, clip=False)  # for refering information
        self.n_actions = self._env.action_space.n

        self.network_dump_path = os.path.join(self.config['result_dir'], 'network.hdf5')

        self.n_action_repeat = self.config['actor']['n_action_repeat']
        self.render_episode_freq = self.config['history']['render_episode_freq']

        self.replay = SharedReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)

    @report_error(logger)
    def _warmup(self, n_warmup_steps, memory, head):
        warmup_actor = self._build_actor(self.env_id)
        for exps in warmup_actor.warmup_act(n_warmup_steps):
            if len(exps) > 0:
                self.replay.mpush(exps, memory, head)

    def _build_actor(self, env_id, greedy=False):
        if greedy:
            env = setup_env(env_id, clip=False)
            policy = Greedy(action_space=env.action_space)
            n_action_repeat = 1
            n_random_actions_at_reset = (0, 0)
            result_dir = os.path.join(self.config['result_dir'], 'greedy')
        else:
            env = setup_env(env_id, clip=True)
            policy = EpsilonGreedy(action_space=env.action_space, **self.config['policy']['params'])
            n_action_repeat = self.n_action_repeat
            n_random_actions_at_reset = tuple(self.config['actor']['n_random_actions_at_reset'])
            result_dir = os.path.join(self.config['result_dir'], 'actor')
        actor_network = VanillaCNN(self.n_actions)
        if self.device >= 0:
            actor_network.to_gpu(self.device)
        logger.info(f'built an actor_network with device {self.device}.')
        obs_preprocessor = AtariPreprocessor()

        actor = QActor(
            env=env,
            network=actor_network,
            policy=policy,
            n_action_repeat=n_action_repeat,
            obs_preprocessor=obs_preprocessor,
            n_random_actions_at_reset=n_random_actions_at_reset,
            n_stack_frames=self.config['n_stack_frames'],
            result_dir=result_dir)

        logger.info(f'built an {"greedy " if greedy else ""}actor.')
        return actor

    def _build_learner(self):
        learner_network = VanillaCNN(self.n_actions)
        if self.device >= 0:
            learner_network.to_gpu(self.device)
        logger.info(f'built an learner_network with device {self.device}.')
        optimizer = getattr(optimizers, self.config['optimizer']['optimizer'])(**self.config['optimizer']['params'])

        learner = FittedQLearner(
            network=learner_network,
            optimizer=optimizer,
            gamma=self.config['learner']['gamma'],
            target_network_update_freq=self.config['learner']['target_network_update_freq'],
            logging_freq=self.config['learner']['logging_freq'])
        logger.info(f'built a learner.')
        return learner

    @report_error(logger)
    def _act(self, lock, memory, head):
        actor = self._build_actor(self.env_id, greedy=False)
        greedy_actor = self._build_actor(self.env_id, greedy=True)
        load_freq = self.config['actor']['load_freq']
        next_load_step = 0

        while actor.total_steps < self.config['n_total_steps'] and actor.total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {actor.total_episodes}, step {actor.total_steps}')
            # actor
            experience, done = actor.act()
            self.replay.push(experience, memory, head)

            if done:
                logger.info(f'Memory length at episode {actor.total_episodes}: {len(self.replay)}')

            # dump episode and greedy actor's play
            if done and actor.total_episodes % self.render_episode_freq == 0:
                actor.dump_episode()

                logger.info('greedy actor is playing...')
                with lock:
                    greedy_actor.load_parameters(self.network_dump_path)
                greedy_actor.total_episodes = actor.total_episodes - 1
                greedy_actor.total_steps = actor.total_steps
                greedy_actor.act_episode()
                greedy_actor.dump_episode()
                logger.info('greedy actor is playing... done.')
            if load_freq != 0 and actor.total_steps >= next_load_step:
                with lock:
                    actor.load_parameters(self.network_dump_path)
                next_load_step += load_freq

    @report_error(logger)
    def _learn(self, lock, memory, head):
        learner = self._build_learner()
        dump_freq = self.config['learner']['dump_freq']
        with lock:
            learner.dump_parameters(self.network_dump_path)  # initial parameter

        while True:
            experiences = self.replay.sample(size=self.config['learner']['batch_size'], memory=memory, head=head)
            learner.learn(experiences)
            if dump_freq != 0 and learner.total_updates % dump_freq == 0:
                with lock:
                    learner.dump_parameters(self.network_dump_path)

    def train(self):
        logger.info('Constructing a shared memory...')
        memory = Array(SharedReplay.AtariExperience, self.replay.limit, lock=True)
        head = Value('i', 0, lock=True)
        logger.info('Constructing a shared memory... done.')

        warmup_proc = multiprocessing.Process(target=self._warmup, args=(self.config['n_warmup_steps'], memory, head))
        warmup_proc.start()
        warmup_proc.join()

        lock = multiprocessing.Lock()
        proc_learner = multiprocessing.Process(target=self._learn, args=(lock, memory, head))
        proc_actor = multiprocessing.Process(target=self._act, args=(lock, memory, head))

        proc_learner.start()
        proc_actor.start()

        proc_actor.join()
        proc_learner.terminate()

        # self._learn(lock, memory, head)
