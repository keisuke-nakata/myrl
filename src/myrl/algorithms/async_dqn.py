import logging
import os
import multiprocessing
from multiprocessing.sharedctypes import RawValue, RawArray
import csv

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
    def _warmup(self, n_warmup_steps, memory_lock, memory, head):
        warmup_actor = self._build_actor(self.env_id)
        for exps in warmup_actor.warmup_act(n_warmup_steps):
            if len(exps) > 0:
                self.replay.mpush(exps, memory_lock, memory, head)

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
            target_network_update_freq=self.config['learner']['target_network_update_freq'], )
        logger.info(f'built a learner.')
        return learner

    @report_error(logger)
    def _act(self, memory_lock, network_dump_lock, memory, head):
        actor = self._build_actor(self.env_id, greedy=False)
        greedy_actor = self._build_actor(self.env_id, greedy=True)
        load_freq = self.config['actor']['load_freq']
        next_load_step = 0

        experience_buffer = []

        while actor.total_steps < self.config['n_total_steps'] and actor.total_episodes < self.config['n_total_episodes']:
            logger.debug(f'episode {actor.total_episodes}, step {actor.total_steps}')
            experience, done = actor.act()
            experience_buffer.append(experience)

            if done:
                self.replay.mpush(experience_buffer, memory_lock, memory, head)
                logger.info(f'Memory length at episode {actor.total_episodes}: {len(self.replay)}')
                experience_buffer = []

            # dump episode and greedy actor's play
            if done and actor.total_episodes % self.render_episode_freq == 0:
                actor.dump_episode()

                logger.info('greedy actor is playing...')
                with network_dump_lock:
                    greedy_actor.load_parameters(self.network_dump_path)
                greedy_actor.total_episodes = actor.total_episodes - 1
                greedy_actor.total_steps = actor.total_steps
                greedy_actor.act_episode()
                greedy_actor.dump_episode()
                logger.info('greedy actor is playing... done.')
            if load_freq != 0 and actor.total_steps >= next_load_step:
                with network_dump_lock:
                    actor.load_parameters(self.network_dump_path)
                next_load_step += load_freq

    @report_error(logger)
    def _learn(self, memory_lock, network_dump_lock, memory, head):
        learner = self._build_learner()
        logging_freq = self.config['learner']['logging_freq']
        network_dump_freq = self.config['learner']['network_dump_freq']
        result_path = os.path.join(self.config['result_dir'], 'learner_history.csv')
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'loss', 'td_error'])
        with network_dump_lock:
            learner.dump_parameters(self.network_dump_path)  # initial parameter

        updates = []
        losses = []
        td_errors = []
        while True:
            experiences = self.replay.sample(size=self.config['learner']['batch_size'], lock=memory_lock, memory=memory, head=head)
            loss, td_error = learner.learn(experiences)
            updates.append(learner.total_updates)
            losses.append(loss)
            td_errors.append(td_error)
            if logging_freq != 0 and learner.total_updates % logging_freq == 0:
                with open(result_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerowws(list(zip(*[updates, losses, td_errors])))
                learner.timer.lap()
                n = len(losses)
                logger.info(
                    f'finished {n} updates with avg loss {sum(losses) / n:.5f}, td_error {sum(td_errors) / n:.5f} in {learner.timer.laptime_str} '
                    f'({n / learner.timer.laptime:.2f} batches (updates) per seconds) '
                    f'(total_updates {learner.total_updates:,}, total_time {learner.timer.elapsed_str})')
                updates = []
                losses = []
                td_errors = []
            if network_dump_freq != 0 and learner.total_updates % network_dump_freq == 0:
                with network_dump_lock:
                    learner.dump_parameters(self.network_dump_path)

    def train(self):
        logger.info('Constructing shared memory...')
        memory = RawArray(SharedReplay.AtariExperience, self.replay.limit)
        head = RawValue('i', 0)
        logger.info('Constructing shared memory... done.')
        memory_lock = multiprocessing.Lock()

        warmup_proc = multiprocessing.Process(target=self._warmup, args=(self.config['n_warmup_steps'], memory_lock, memory, head))
        warmup_proc.start()
        warmup_proc.join()

        network_dump_lock = multiprocessing.Lock()
        proc_learner = multiprocessing.Process(target=self._learn, args=(memory_lock, network_dump_lock, memory, head))
        proc_actor = multiprocessing.Process(target=self._act, args=(memory_lock, network_dump_lock, memory, head))

        proc_learner.start()
        proc_actor.start()

        proc_actor.join()
        proc_learner.terminate()
