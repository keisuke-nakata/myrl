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


class DQNAgent:
    def build(self, config, env_id, device=0):
        """
        device: -1 (CPU), 0 (GPU)
        """
        self.recorder = StandardRecorder(os.path.join(config['result_dir'], 'history.csv'))
        self.recorder.start()

        self.eval_recorder = StandardRecorder(os.path.join(config['result_dir'], 'eval_history.csv'))
        self.eval_recorder.template = '(eval) ' + self.eval_recorder.template
        self.eval_recorder.start()

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
        self.eval_actor = self._build_actor(env_id, policy, eval_=True)
        self.learner = self._build_learner(self.network, self.config['learner'])
        self.replay = VanillaReplay(limit=self.config['replay']['limit'])

    def _build_actor(self, env_id, policy, eval_=False):
        env = setup_env(env_id, clip=False, life_episode=not eval_)
        preprocessor = AtariPreprocessor()
        if eval_:
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            # explorer = GreedyExplorer()
            explorer = EpsilonGreedyExplorer(**self.config['explorer']['eval']['params'])
        else:
            n_noop_at_reset = self.config['actor']['n_noop_at_reset']
            explorer = LinearAnnealEpsilonGreedyExplorer(**self.config['explorer']['params'])

        actor = Actor(env, policy, explorer, preprocessor, n_noop_at_reset, self.config['actor']['n_stack_frames'], self.config['actor']['n_action_repeat'])
        logger.info(f'built {"eval" if eval_ else ""} actor.')
        return actor

    def _build_learner(self, network, learner_config):
        optimizer = getattr(optimizers, learner_config['optimizer']['class'])(**learner_config['optimizer']['params'])
        learner = FittedQLearner(
            network=network,
            optimizer=optimizer,
            gamma=learner_config['gamma']
        )
        logger.info(f'built learner.')
        return learner

    def train(self):
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

                # actor
                state, action, reward, done, is_random, epsilon, warming_up, q_values, action_meaning = self.actor.act(n_steps)
                reward = np.sign(reward)
                experience = (state, action, reward, done)
                self.replay.push(experience)

                # learner
                if not warming_up and n_steps % self.config['learner']['target_network_update_freq_step'] == 0:
                    self.learner.update_target_network(soft=None)
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
                    step_loop = False

                    self.recorder.end_episode()
                    self.recorder.dump_episodewise_csv()
                    msg = self.recorder.dump_episodewise_str()
                    if warming_up:
                        msg = '(warmup) ' + msg
                    logger.info(msg)

                    # evaluate performance
                    if not warming_up and n_steps >= next_eval_step:
                        result_episode_dir = os.path.join(self.config['result_dir'], f'episode{n_episodes:06}')

                        # save actor's play.
                        os.makedirs(result_episode_dir, exist_ok=True)
                        self.actor.dump_episode_anim(os.path.join(result_episode_dir, 'play.mp4'))
                        self.recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'step_history.csv'))
                        with open(os.path.join(result_episode_dir, 'summary.txt'), 'w') as f:
                            print(self.recorder.dump_episodewise_str(), file=f)
                        visualize(self.recorder.episodewise_csv_path)

                        # eval_actor's play and save it.
                        logger.info(f'(episode {n_episodes}) eval_actor is playing...')
                        self.eval_recorder.begin_episode()
                        n_eval_steps = n_steps
                        n_eval_episode_steps = 0
                        eval_done = False
                        while not eval_done:
                            n_eval_steps += 1
                            n_eval_episode_steps += 1
                            state, action, reward, eval_done, is_random, epsilon, q_values, action_meaning = self.eval_actor.act(n_steps)
                            self.eval_recorder.record(
                                total_step=n_eval_steps, episode=n_episodes, episode_step=n_eval_episode_steps,
                                reward=reward, action=action, action_meaning=action_meaning, is_random=is_random, epsilon=epsilon,
                                action_q=q_values[action], loss=float('nan'), td_error=float('nan'))
                        self.eval_recorder.end_episode()
                        self.eval_recorder.dump_episodewise_csv()
                        logger.info(self.eval_recorder.dump_episodewise_str())
                        self.eval_actor.dump_episode_anim(os.path.join(result_episode_dir, 'eval_play.mp4'))
                        self.eval_recorder.dump_stepwise_csv(os.path.join(result_episode_dir, 'eval_step_history.csv'))
                        with open(os.path.join(result_episode_dir, 'eval_summary.txt'), 'w') as f:
                            print(self.eval_recorder.dump_episodewise_str(), file=f)
                        visualize(self.eval_recorder.episodewise_csv_path)

                        next_eval_step += self.config['eval_freq_step']

                if n_steps >= self.config['n_total_steps'] or n_episodes >= self.config['n_total_episodes']:
                    episode_loop = False
                # preemptive timer...
                if self.recorder.timer.elapsed > self.config['total_seconds']:
                    logger.info('Timeup. Training end.')
                    episode_loop = False

        return self.recorder.episodewise_csv_path, self.eval_recorder.episodewise_csv_path
