import logging
import os
import multiprocessing as mp
import threading as th
import queue

import numpy as np

from ..networks import build_network
from ..actors import Actor
from ..learners import build_learner
from ..policies import QPolicy
from ..explorers import build_explorer
from ..replays import build_replay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env, identity
from ..utils import StandardRecorder, visualize, plot_graph

logger = logging.getLogger(__name__)


"""
用語の決まり：
frame: もとのゲームにおけるフレームのこと。FPS を出す際は、フレームスキップ (action_repeat) が入っていたとしても、そのぶんを含めて数えることとする
step: 何回行動を選択したか。つまり、step の回数は、frame の回数をフレームスキップ (action_repeat) で割った値となる。
"""


class AsyncDQNAgent:
    def __init__(self, config, env_id, device=0):
        """device: -1 (CPU), 0 (GPU)"""
        self.config = config
        self.env_id = env_id
        self.device = device

    def build(self):
        self.recorder = StandardRecorder(os.path.join(self.config['result_dir'], 'history.csv'))
        self.recorder.start()

        self.eval_recorder = StandardRecorder(os.path.join(self.config['result_dir'], 'eval_history.csv'))
        self.eval_recorder.template = '(eval) ' + self.eval_recorder.template
        self.eval_recorder.start()

        self.n_actions = setup_env(self.env_id).action_space.n
        self.parameter_path = os.path.join(self.config['result_dir'], 'weight.hdf5')

    def _build_actor(self, env_id, policy, eval_=False):
        env = setup_env(env_id, life_episode=not eval_)
        odb_preprocessor = AtariPreprocessor()
        if eval_:
            reward_preprocessor = identity
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            explorer = build_explorer(self.config['eval_explorer'])
        else:
            reward_preprocessor = np.sign
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            explorer = build_explorer(self.config['explorer'])

        actor = Actor(
            env, policy, explorer, odb_preprocessor, reward_preprocessor,
            n_noop_at_reset, self.config['actor']['n_stack_frames'], self.config['actor']['n_action_repeat'])
        logger.info(f'built {"eval " if eval_ else ""}actor {actor}.')
        return actor

    def act(self, actor_record_queue, actor_replay_queue, parameter_lock):
        network = build_network(self.n_actions, self.config['network'])
        if self.device >= 0:
            network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')
        policy = QPolicy(network)
        actor = self._build_actor(self.env_id, policy)

        with parameter_lock:
            actor.load_parameters(self.parameter_path)

        n_steps = 0
        n_episodes = 0
        next_eval_step = self.config['eval_freq_step']
        while True:  # child actor process runs eternally
            n_episodes += 1
            n_episode_steps = 0
            step_loop = True

            while step_loop:
                n_steps += 1
                n_episode_steps += 1

                experience, exploration_info, q_values, action_meaning = actor.act(n_steps)

                action_q = q_values[experience.action]
                actor_record_queue.put(  # this blocks
                    (n_steps, n_episodes, n_episode_steps, experience, exploration_info, action_meaning, action_q))
                actor_replay_queue.put(experience)

                if not exploration_info.warming_up and n_steps % self.config['actor']['parameter_load_freq_step'] == 0:
                    with parameter_lock:
                        actor.load_parameters(self.parameter_path)

                # if episode is done...
                if experience.done:
                    step_loop = False

                    # evaluate performance
                    if not exploration_info.warming_up and n_steps >= next_eval_step:
                        result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:06}')
                        os.makedirs(result_episode_dir, exist_ok=True)

                        # save actor's play.
                        actor.dump_episode_anim(os.path.join(result_episode_dir, 'play.mp4'))

                        next_eval_step += self.config['eval_freq_step']

    def learn(self, learner_record_queue, learner_replay_queue, parameter_lock, ready_to_learn_event):
        network = build_network(self.n_actions, self.config['network'])
        if self.device >= 0:
            network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')

        multi_step_n = self.config.get('multi_step_n', None)
        learner = build_learner(network, self.config['learner'], gamma=self.config['gamma'], multi_step_n=multi_step_n)
        learner.dump_parameters(self.parameter_path)  # to sync the first parameter of learener's network with actor's one
        parameter_lock.release()

        learner_local_queue = queue.Queue(self.config['learner_replay_queue_size'])

        def fetch_batch(learner_replay_queue, learner_local_queue):
            """fetches data from multiprcessing queue and put them into local queue"""
            while True:
                data = learner_replay_queue.get()
                learner_local_queue.put(data)
        fetch_thread = th.Thread(target=fetch_batch, args=(learner_replay_queue, learner_local_queue))

        ready_to_learn_event.wait()

        fetch_thread.start()
        n_updates = 0
        while True:
            n_updates += 1
            if n_updates % self.config['learner']['target_network_update_freq_update'] == 0:
                learner.update_target_network(soft=None)
            batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int = learner_local_queue.get()  # this blocks
            loss, td_error = learner.learn(batch_state_int, batch_action, batch_reward, batch_done, batch_next_state_int)
            learner_record_queue.put((loss, td_error))  # this blocks
            if n_updates % self.config['learner']['parameter_dump_freq_update'] == 0:
                with parameter_lock:
                    learner.dump_parameters(self.parameter_path)
        fetch_thread.join()

    def replay(self, actor_replay_queue, learner_replay_queue, ready_to_learn_event):
        multi_step_n = self.config.get('multi_step_n', None)
        replay = build_replay(self.config['replay'], gamma=self.config['gamma'], multi_step_n=multi_step_n)

        while True:
            for _ in range(self.config['learner']['learn_freq_step']):
                try:
                    experience = actor_replay_queue.get(block=False)  # this does not block
                except queue.Empty:
                    break
                else:
                    replay.push(experience)

            if ready_to_learn_event.is_set() and not learner_replay_queue.full():
                batch_state, batch_action, batch_reward, batch_done, batch_next_state = replay.batch_sample(
                    self.config['learner']['batch_size'])
                learner_replay_queue.put((batch_state, batch_action, batch_reward, batch_done, batch_next_state))

    def train(self):
        # prepare for multiprocessing
        actor_record_queue = mp.Queue(self.config['actor_record_queue_size'])
        actor_replay_queue = mp.Queue(self.config['actor_replay_queue_size'])
        learner_record_queue = mp.Queue(self.config['learner_record_queue_size'])
        learner_replay_queue = mp.Queue(self.config['learner_replay_queue_size'])
        parameter_lock = mp.Lock()
        ready_to_learn_event = mp.Event()

        learn_process = mp.Process(
            target=self.learn, args=(learner_record_queue, learner_replay_queue, parameter_lock, ready_to_learn_event))
        parameter_lock.acquire()  # this lock is released when learn_process is built
        learn_process.start()
        act_process = mp.Process(
            target=self.act, args=(actor_record_queue, actor_replay_queue, parameter_lock))
        act_process.start()
        replay_process = mp.Process(
            target=self.replay, args=(actor_replay_queue, learner_replay_queue, ready_to_learn_event))
        replay_process.start()

        # build eval_actor
        network = build_network(self.n_actions, self.config['network'])
        plot_graph(network, os.path.join(self.config['result_dir'], 'graph.png'))
        if self.device >= 0:
            network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')
        policy = QPolicy(network)
        eval_actor = self._build_actor(self.env_id, policy, eval_=True)

        n_steps = 0
        n_episodes = 0
        next_eval_step = self.config['eval_freq_step']

        episode_loop = True
        while episode_loop:
            self.recorder.begin_episode()
            n_episodes += 1
            n_episode_steps = 0
            step_loop = True

            while step_loop:
                n_steps += 1
                n_episode_steps += 1

                (actor_n_steps, actor_n_episodes, actor_n_episode_steps,
                    experience, exploration_info, action_meaning, action_q) = actor_record_queue.get()
                assert n_steps == actor_n_steps
                assert n_episodes == actor_n_episodes
                assert n_episode_steps == actor_n_episode_steps

                loss, td_error = float('nan'), float('nan')
                if not exploration_info.warming_up:
                    ready_to_learn_event.set()
                    if n_steps % self.config['learner']['learn_freq_step'] == 0:
                        loss, td_error = learner_record_queue.get()

                # recorder
                self.recorder.record(
                    total_step=n_steps, episode=n_episodes, episode_step=n_episode_steps,
                    reward=experience.reward, action=experience.action, action_meaning=action_meaning,
                    is_random=exploration_info.is_random, epsilon=exploration_info.epsilon,
                    action_q=action_q, loss=loss, td_error=td_error)

                # if episode is done...
                if experience.done:
                    step_loop = False

                    self.recorder.end_episode()
                    self.recorder.dump_episodewise_csv()
                    msg = self.recorder.dump_episodewise_str()
                    if exploration_info.warming_up:
                        msg = '(warmup) ' + msg
                    logger.info(msg)

                    # evaluate performance
                    if not exploration_info.warming_up and n_steps >= next_eval_step:
                        result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:06}')
                        os.makedirs(result_episode_dir, exist_ok=True)

                        # save actor's play.
                        self.recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'step_history.csv'))
                        with open(os.path.join(result_episode_dir, 'summary.txt'), 'w') as f:
                            print(self.recorder.dump_episodewise_str(), file=f)
                        visualize(self.recorder.episodewise_csv_path, title=self.env_id)

                        # eval_actor's play and save it.
                        logger.info(f'(episode {n_episodes}) eval_actor is playing...')
                        self.eval_recorder.begin_episode()
                        with parameter_lock:
                            eval_actor.load_parameters(self.parameter_path)
                        n_eval_steps = n_steps
                        n_eval_episode_steps = 0
                        while True:
                            n_eval_steps += 1
                            n_eval_episode_steps += 1
                            eval_experience, eval_exploration_info, q_values, action_meaning = eval_actor.act(n_steps)
                            self.eval_recorder.record(
                                total_step=n_eval_steps, episode=n_episodes, episode_step=n_eval_episode_steps,
                                reward=eval_experience.reward, action=eval_experience.action, action_meaning=action_meaning,
                                is_random=eval_exploration_info.is_random, epsilon=eval_exploration_info.epsilon,
                                action_q=q_values[eval_experience.action], loss=float('nan'), td_error=float('nan'))
                            if eval_experience.done:
                                break
                        self.eval_recorder.end_episode()
                        self.eval_recorder.dump_episodewise_csv()
                        logger.info(self.eval_recorder.dump_episodewise_str())
                        eval_actor.dump_episode_anim(os.path.join(result_episode_dir, 'eval_play.mp4'))
                        self.eval_recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'eval_step_history.csv'))
                        with open(os.path.join(result_episode_dir, 'eval_summary.txt'), 'w') as f:
                            print(self.eval_recorder.dump_episodewise_str(), file=f)
                        visualize(self.eval_recorder.episodewise_csv_path, title=self.env_id)

                        next_eval_step += self.config['eval_freq_step']

                if n_steps >= self.config['n_total_steps'] or n_episodes >= self.config['n_total_episodes']:
                    episode_loop = False
                # preemptive timer...
                if self.recorder.timer.elapsed > self.config['total_seconds']:
                    logger.info('Timeup. Training end.')
                    episode_loop = False

        act_process.terminate()
        learn_process.terminate()
        replay_process.terminate()

        return self.recorder.episodewise_csv_path, self.eval_recorder.episodewise_csv_path
