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

# multiprocessing.set_start_method('forkserver')
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

        RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat).flush()
        # replay.flush()
        logger.info('flushed redis server data.')

        lock = multiprocessing.Lock()
        self.proc_learner = multiprocessing.Process(target=self._learn, args=(lock, ))
        self.proc_actor = multiprocessing.Process(target=self._act, args=(lock, ))

    def _warmup(self, n_warmup_steps):
        warmup_actor = self._build_actor(self.env_id)
        replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)
        for exps in warmup_actor.warmup_act(n_warmup_steps):
            if len(exps) > 0:
                replay.mpush(exps)

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

    def _act(self, lock):
        actor = self._build_actor(self.env_id, greedy=False)
        greedy_actor = self._build_actor(self.env_id, greedy=True)
        replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)
        load_freq = self.config['actor']['load_freq']
        next_load_step = 0

        while actor.total_steps < self.config['n_total_steps'] and actor.total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {actor.total_episodes}, step {actor.total_steps}')
            experience, done = actor.act()
            replay.push(experience)

            if done:
                logger.info(f'Memory length at episode {actor.total_episodes}: {len(replay)}')

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

    def _learn(self, lock):
        learner = self._build_learner()
        replay = RedisReplay(limit=self.config['replay']['limit'] // self.n_action_repeat)
        dump_freq = self.config['learner']['dump_freq']
        with lock:
            learner.dump_parameters(self.network_dump_path)  # initial parameter

        while True:
            experiences = replay.sample(size=self.config['learner']['batch_size'])
            learner.learn(experiences)
            if dump_freq != 0 and learner.total_updates % dump_freq == 0:
                with lock:
                    learner.dump_parameters(self.network_dump_path)

    def train(self):
        # warmup
        warmup_proc = multiprocessing.Process(target=self._warmup, args=(self.config['n_warmup_steps'], ))
        warmup_proc.start()
        warmup_proc.join()

        self.proc_learner.start()
        self.proc_actor.start()

        self.proc_actor.join()
        self.proc_learner.terminate()
