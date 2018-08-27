import logging
import os

import numpy as np
from chainer import optimizers

from ..networks import VanillaCNN
from ..actors import Actor
from ..learners import FittedQLearner
from ..policies import QPolicy, LinearAnnealEpsilonGreedyExplorer, EpsilonGreedyExplorer
from ..replays import VanillaReplay
from ..preprocessors import AtariPreprocessor
from ..env_wrappers import setup_env
from ..utils import StandardRecorder, visualize

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
        self.recorder = StandardRecorder(os.path.join(config['result_dir'], 'history.csv'))
        self.recorder.start()

        self.test_recorder = StandardRecorder(os.path.join(config['result_dir'], 'test_history.csv'))
        self.test_recorder.template = '(test) ' + self.test_recorder.template
        self.test_recorder.start()

        self.config = config
        self.env_id = env_id
        self.device = device

        self.n_actions = setup_env(env_id).action_space.n

        self.network = VanillaCNN(self.n_actions)
        if self.device >= 0:
            self.network.to_gpu(self.device)
        logger.info(f'built a network with device {self.device}.')

        policy = QPolicy(self.network)
        self.actor = self._build_actor(env_id, policy)
        self.test_actor = self._build_actor(env_id, policy, test=True)
        self.learner = self._build_learner(self.network, self.config['learner'])
        self.replay = VanillaReplay(limit=self.config['replay']['limit'])

    def _build_actor(self, env_id, policy, test=False):
        env = setup_env(env_id, clip=False, life_episode=not test)
        preprocessor = AtariPreprocessor()
        if test:
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            # explorer = GreedyExplorer()
            explorer = EpsilonGreedyExplorer(**self.config['explorer']['test']['params'])
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
        n_steps = 1
        n_episodes = 1
        n_episode_steps = 1
        next_test_step = self.config['test_freq_step']
        n_warmup_steps = self.config['n_warmup_steps']
        if n_warmup_steps > 0:
            warming_up = True
            logger.info(f'warming up {n_warmup_steps} steps...')
        else:
            warming_up = False
        while n_steps <= self.config['n_total_steps'] and n_episodes <= self.config['n_total_episodes']:
            if warming_up and n_steps > n_warmup_steps:
                warming_up = False
                logger.info(f'warming up {n_warmup_steps} steps... done.')

            # actor
            step = 0 if warming_up else n_steps  # 0 means epsilon = 1
            state, action, reward, done, is_random, epsilon, q_values, action_meaning = self.actor.act(step)
            reward = np.sign(reward)
            experience = (state, action, reward, done)
            self.replay.push(experience)

            # learner
            if not warming_up and n_steps % self.config['learner']['target_network_update_freq_step'] == 0:
                self.learner.update_target_network()
            if not warming_up and n_steps % self.config['learner']['learn_freq_step'] == 0:
                batch_state, batch_action, batch_reward, batch_done, batch_next_state = self.replay.batch_sample(self.config['learner']['batch_size'])
                loss, td_error = self.learner.learn(batch_state, batch_action, batch_reward, batch_done, batch_next_state)
            else:
                loss, td_error = float('nan'), float('nan')

            # recorder
            self.recorder.record(
                total_step=n_steps, episode=n_episodes, episode_step=n_episode_steps,
                reward=reward, action=action, action_meaning=action_meaning, is_random=is_random, epsilon=epsilon,
                action_q=q_values[action], loss=loss, td_error=td_error)

            # if episode is done...
            if done:
                self.recorder.end_episode()
                self.recorder.dump_episodewise_csv()
                msg = self.recorder.dump_episodewise_str()
                if warming_up:
                    msg = '(warmup) ' + msg
                logger.info(msg)

                # testing
                if not warming_up and n_steps >= next_test_step:
                    next_test_step += self.config['test_freq_step']
                    # save actor's play.
                    result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:06}')
                    os.makedirs(result_episode_dir, exist_ok=True)
                    self.actor.dump_episode_anim(os.path.join(result_episode_dir, 'play.mp4'))
                    self.recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'step_history.csv'))
                    with open(os.path.join(result_episode_dir, 'summary.txt'), 'w') as f:
                        print(self.recorder.dump_episodewise_str(), file=f)
                    visualize(self.recorder.episodewise_csv_path)  # TODO: action meaning もほしい

                    # test_actor's play and save it.
                    logger.info(f'(episode {n_episodes}) test_actor is playing...')
                    self.test_recorder.begin_episode()
                    n_test_steps = n_steps
                    n_test_episode_steps = 1
                    while True:
                        state, action, reward, done, is_random, epsilon, q_values, action_meaning = self.test_actor.act(n_steps)
                        self.test_recorder.record(
                            total_step=n_test_steps, episode=n_episodes, episode_step=n_test_episode_steps,
                            reward=reward, action=action, action_meaning=action_meaning, is_random=is_random, epsilon=epsilon,
                            action_q=q_values[action], loss=float('nan'), td_error=float('nan'))
                        n_test_steps += 1
                        n_test_episode_steps += 1
                        if done:
                            break
                    self.test_recorder.end_episode()
                    self.test_recorder.dump_episodewise_csv()
                    logger.info(self.test_recorder.dump_episodewise_str())
                    self.test_actor.dump_episode_anim(os.path.join(result_episode_dir, 'test_play.mp4'))
                    self.test_recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'test_step_history.csv'))
                    with open(os.path.join(result_episode_dir, 'test_summary.txt'), 'w') as f:
                        print(self.test_recorder.dump_episodewise_str(), file=f)
                    visualize(self.test_recorder.episodewise_csv_path)

                self.recorder.begin_episode()
                n_episodes += 1
                n_episode_steps = 1

                # preemptive timer...
                if self.recorder.timer.elapsed > self.config['total_seconds']:
                    logger.info('Timeup. Training end.')
                    break
            else:  # if episode continues
                n_episode_steps += 1

            n_steps += 1

        return self.recorder.episodewise_csv_path, self.test_recorder.episodewise_csv_path
