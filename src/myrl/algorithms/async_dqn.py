import logging
import os
import multiprocessing as mp

import numpy as np
from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import Actor
from ..learners import FittedQLearner
from ..policies import QPolicy, LinearAnnealEpsilonGreedyExplorer, GreedyExplorer
from ..replays import VanillaReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env
from ..utils import StandardRecorder

CPU_ID = -1

logger = logging.getLogger(__name__)


"""
用語の決まり：
frame: もとのゲームにおけるフレームのこと。FPS を出す際は、フレームスキップ (action_repeat) が入っていたとしても、そのぶんを含めて数えることとする
step: 何回行動を選択したか。つまり、step の回数は、frame の回数をフレームスキップ (action_repeat) で割った値となる。
"""


def act(n_actions, device, env_id, parameter_path, explorer_config, actor_config, actor_msg_queue, actor_exp_queue, parameter_lock):
    network = VanillaCNN(n_actions)
    if device >= 0:
        network.to_gpu(device)
    logger.info(f'built a network with device {device}.')

    env = setup_env(env_id, clip=False)
    policy = QPolicy(network)
    explorer = LinearAnnealEpsilonGreedyExplorer(**explorer_config['params'])
    preprocessor = AtariPreprocessor()

    actor = Actor(env, policy, explorer, preprocessor, actor_config['n_noop_at_reset'], actor_config['n_stack_frames'], actor_config['n_action_repeat'])
    logger.info(f'built an actor.')

    while True:
        msg = actor_msg_queue.get()  # this blocks if empty
        command = msg[0]
        if command == 'step':
            step = msg[1]
            dump_episode_path = msg[2]

            state, action, reward, done, is_random, epsilon = actor.act(step)
            actor_exp_queue.put((state, action, reward, done, is_random, epsilon))  # infinite queue does not block
            if done and dump_episode_path is not None:
                actor.dump_episode_gif(dump_episode_path)
        elif command == 'load':
            with parameter_lock:
                actor.load_parameters(parameter_path)
        else:
            raise ValueError(f'Unknown message {msg}')


def learn(n_actions, device, parameter_path, learner_config, learner_msg_queue, batch_queue, replay_msg_queue, learner_result_queue, parameter_lock):
    network = VanillaCNN(n_actions)
    if device >= 0:
        network.to_gpu(device)
    logger.info(f'built a network with device {device}.')

    optimizer = getattr(optimizers, learner_config['optimizer']['class'])(**learner_config['optimizer']['params'])

    learner = FittedQLearner(
        network=network,
        optimizer=optimizer,
        gamma=learner_config['gamma']
    )
    logger.info(f'built a learner.')

    while True:
        msg = learner_msg_queue.get()  # this blocks if empty
        command = msg[0]
        if command == 'learn':
            batch = batch_queue.get()  # this blocks if empty
            replay_msg_queue.put(('fetch', None))
            loss, td_error = learner.learn(*batch)
            learner_result_queue.put((loss, td_error))  # infinite queue does not block
        elif command == 'sync':
            learner.update_target_network()
        elif command == 'dump':
            with parameter_lock:
                learner.dump_parameters(parameter_path)
        else:
            raise ValueError(f'Unknown message {msg}')


def replay(limit, batch_size, batch_queue, replay_msg_queue):
    replay = VanillaReplay(limit=limit)
    while True:
        msg = replay_msg_queue.get()  # this blocks if empty
        command = msg[0]
        if command == 'fetch':
            batch = replay.batch_sample(batch_size)
            batch_queue.put(batch)  # infinite queue does not block
        elif command == 'push':
            exp = msg[1]
            replay.push(exp)
        else:
            raise ValueError(f'Unknown message {msg}')


class AsyncDQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        mp.set_start_method('forkserver')

        self.recorder = StandardRecorder(os.path.join(config['result_dir'], 'history.csv'))
        self.recorder.start()

        self.config = config
        self.env_id = env_id
        self.device = device

        self.n_actions = setup_env(env_id).action_space.n

        self.network = VanillaCNN(self.n_actions)
        if self.device >= 0:
            self.network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')

        # policy = QPolicy(self.network)
        # self.test_actor = self._build_actor(env_id, policy, test=True)
        # self.replay = VanillaReplay(limit=self.config['replay']['limit'], device=self.device)

        parameter_path = '!!!!!'

        # queues and locks
        self.actor_msg_queue = mp.Queue()
        self.actor_exp_queue = mp.Queue()
        self.learner_msg_queue = mp.Queue()
        self.batch_queue = mp.Queue()
        self.replay_msg_queue = mp.Queue()
        self.learner_result_queue = mp.Queue()
        self.parameter_lock = mp.Lock()

        # act process
        self.act_process = mp.Process(target=act, args=(
            self.n_actions, CPU_ID, env_id, parameter_path, self.config['explorer'], self.config['actor'],
            self.actor_msg_queue, self.actor_exp_queue, self.parameter_lock))
        self.act_process.start()

        # learn process
        self.learn_process = mp.Process(target=learn, args=(
            self.n_actions, device, parameter_path, self.config['learner'],
            self.learner_msg_queue, self.batch_queue, self.replay_msg_queue, self.learner_result_queue, self.parameter_lock))
        self.learn_process.start()

        # replay procerss
        self.replay_process = mp.Process(target=replay, args=(
            self.config['replay']['limit'], self.config['learner']['batch_size'],
            self.batch_queue, self.replay_msg_queue))
        self.replay_process.start()

    def _build_actor(self, env_id, policy, test=False):
        env = setup_env(env_id, clip=False)
        preprocessor = AtariPreprocessor()
        if test:
            n_noop_at_reset = (0, 0)
            explorer = GreedyExplorer()
        else:
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            explorer = LinearAnnealEpsilonGreedyExplorer(**self.config['explorer']['params'])

        actor = Actor(env, policy, explorer, preprocessor, n_noop_at_reset, self.config['actor']['n_stack_frames'], self.config['actor']['n_action_repeat'])
        logger.info(f'built {"a test " if test else "an "}actor.')
        return actor

    def _build_learner(self, network, learner_config):
        optimizer = getattr(optimizers, learner_config['optimizer']['class'])(**learner_config['optimizer']['params'])
        learner = FittedQLearner(
            network=network,
            optimizer=optimizer,
            gamma=learner_config['gamma']
        )
        logger.info(f'built a learner.')
        return learner

    def train(self):
        self.recorder.begin_episode()
        learn_freq_step = self.config['learner']['learn_freq_step']

        n_warmup_steps = self.config['n_warmup_steps']
        if n_warmup_steps > 0:
            warming_up = True
            logger.info(f'warming up {n_warmup_steps} steps...')
        else:
            warming_up = False
        assert warming_up  # requires warming_up in order to start replay_process fetching

        # run actor at first to fill the queue
        for future_step in range(1, learn_freq_step + 1):
            step = 0 if warming_up else future_step  # 0 means epsilon = 1
            self.actor_msg_queue.put(('step', step, None))

        n_steps = 1
        n_episodes = 1
        n_episode_steps = 1
        while n_steps <= self.config['n_total_steps'] and n_episodes <= self.config['n_total_episodes']:
            if warming_up and n_steps > n_warmup_steps:  # end of the warming-up
                warming_up = False
                logger.info(f'warming up {n_warmup_steps} steps... done.')
                # prefetch a batch at end of the warming-up.
                # succeeding fetching will be fired from learn_process.
                self.replay_msg_queue.put(('fetch', None))
                self.learner_msg_queue.put(('learn', None))  # to fill the queue first

            # trigger next learner step
            if not warming_up:
                self.learner_msg_queue.put(('learn', None))

            # trigger next actor steps
            for future_step in range(n_steps + learn_freq_step, n_steps + learn_freq_step * 2):
                if warming_up:
                    step = 0  # 0 means epsilon = 1
                    dump_episode_path = None
                else:
                    step = future_step
                    if n_episodes % self.config['test_freq_episode'] == 0:
                        # result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:05}')
                        # dump_episode_path = '!!!!!'  # FIXME
                        dump_episode_path = None
                    else:
                        dump_episode_path = None
                self.actor_msg_queue.put(('step', step, dump_episode_path))

            # get current results and push them to recorder/replay
            if not warming_up:
                loss, td_error = self.learner_result_queue.get()
            else:
                loss, td_error = float('nan'), float('nan')
            for current_step in range(n_steps, n_steps + learn_freq_step):
                if not warming_up and current_step % self.config['learner']['target_network_update_freq_step'] == 0:
                    self.learner_msg_queue.put(('sync', None))
                if not warming_up and current_step % self.config['actor_leaner_network_sync_freq_step'] == 0:
                    self.learner_msg_queue.put(('dump', None))
                    self.actor_msg_queue.put(('load', None))

                state, action, reward, done, is_random, epsilon = self.actor_exp_queue.get()
                reward = np.sign(reward)

                # replay
                experience = (state, action, reward, done)
                self.replay_msg_queue.put(('push', experience))

                # recorder
                record_loss = loss if current_step == n_steps else float('nan')
                record_td_error = td_error if current_step == n_steps else float('nan')
                self.recorder.record(
                    total_step=current_step, episode=n_episodes, episode_step=n_episode_steps,
                    reward=reward, action=action, is_random=is_random, epsilon=epsilon, loss=record_loss, td_error=record_td_error)

                # if episode is done...
                if done:
                    self.recorder.end_episode()
                    self.recorder.dump_episodewise_csv()
                    logger.info(self.recorder.dump_episodewise_str())

                    # testing
                    # if n_episodes % self.config['test_freq_episode'] == 0:
                    #     result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:05}')
                    #     os.makedirs(result_episode_dir, exist_ok=True)
                    #     self.actor.dump_episode_gif(os.path.join(result_episode_dir, 'play.gif'))
                    #     self.recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'step_history.csv'))
                    #     with open(os.path.join(result_episode_dir, 'summary.txt'), 'w') as f:
                    #         print(self.recorder.dump_episodewise_str(), file=f)
                    #
                    #     # test_actor's play
                    #     logger.info(f'(episode {n_episodes}) test_actor is playing...')
                    #     self.recorder.begin_episode()
                    #     n_test_episode_steps = 1
                    #     while True:
                    #         state, action, reward, done, is_random, epsilon = self.test_actor.act(n_steps)
                    #         self.recorder.record(
                    #             total_step=n_steps, episode=n_episodes, episode_step=n_test_episode_steps,
                    #             reward=reward, action=action, is_random=is_random, epsilon=epsilon, loss=0.0, td_error=0.0)
                    #         n_test_episode_steps += 1
                    #         if done:
                    #             break
                    #     self.recorder.end_episode()
                    #     logger.info(self.recorder.dump_episodewise_str())
                    #     self.test_actor.dump_episode_gif(os.path.join(result_episode_dir, 'test_play.gif'))
                    #     self.recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'test_step_history.csv'))
                    #     with open(os.path.join(result_episode_dir, 'test_summary.txt'), 'w') as f:
                    #         print(self.recorder.dump_episodewise_str(), file=f)

                    self.recorder.begin_episode()
                    n_episodes += 1
                    n_episode_steps = 1
                else:  # if episode continues
                    n_episode_steps += 1

            n_steps += learn_freq_step
