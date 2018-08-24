from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os
import random

import gym
gym.undo_logger_setup()  # NOQA
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import imageio

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl.experiments.evaluator import save_agent
# from chainerrl.experiments.evaluator import Evaluator, save_agent
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer


def phi(obs, last_obs):
    if last_obs is None:
        ret = obs
    else:
        ret = np.maximum(obs, last_obs)
    ret = rgb2gray(ret)  # will be ((210, 160), dtype=np.float64), scales [0, 1]
    ret = resize(ret, output_shape=(84, 84, 1), mode='constant', anti_aliasing=True, order=1)  # 210x160 -> 84x84x1, scales [0, 1]. order=1 means "bilinear".
    ret = ret.astype(np.float32)
    return ret


def train_agent(agent, env, steps, outdir, max_episode_len=None,
                step_offset=0, eval_env=None, logger=None, eval_interval=None, eval_n_runs=None, eval_epsilon=None):
    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()
    done = False
    episode_raw_obs = [obs]
    n_random_actions = random.randint(0, 30)
    for _ in range(n_random_actions):
        observation, reward, done, info = env.step(0)
        episode_raw_obs.append(observation)
    assert not done
    if len(episode_raw_obs) == 1:
        last_obs = None
    else:
        last_obs = episode_raw_obs[-2]
    episode_obs = [phi(episode_raw_obs[-1], last_obs)]

    r = 0

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:
            # get current state
            state = episode_obs[-4:]
            while len(state) < 4:
                state = [state[0].copy()] + state
            state = np.transpose(np.concatenate(state, axis=-1), (2, 0, 1))

            # a_t
            action = agent.act_and_train(state, r)
            # o_{t+1}, r_{t+1}
            r = 0
            for _ in range(4):
                obs, tmp_r, done, info = env.step(action)
                episode_raw_obs.append(obs)
                r += tmp_r
                if done:
                    break
            episode_obs.append(phi(episode_raw_obs[-1], episode_raw_obs[-2]))
            r = np.sign(r)

            t += 1
            episode_r += r
            episode_len += 1

            if done or episode_len == max_episode_len or t == steps:
                # get current state
                state = episode_obs[-4:]
                while len(state) < 4:
                    state = [state[0].copy()] + state
                state = np.transpose(np.concatenate(state, axis=-1), (2, 0, 1))

                agent.stop_episode_and_train(state, r, done=done)
                epsilon = agent.explorer.compute_epsilon(agent.t)
                logger.info('outdir:{} step:{:,} episode:{} R:{} episode_step:{} epsilon:{}'.format(outdir, t, episode_idx, episode_r, episode_len, epsilon))
                logger.info('statistics:%s', agent.get_statistics())

                if t % eval_interval == 0:
                    imageio.mimwrite(os.path.join(outdir, 'episode{}.mp4'.format(episode_idx)), episode_raw_obs, fps=60)

                    # eval run
                    for i in range(eval_n_runs):
                        eval_episode_raw_obs, eval_episode_r, eval_episode_len = run_eval_episode(agent, eval_env, eval_epsilon)
                        logger.info('(eval) R:{} episode_step:{}'.format(eval_episode_r, eval_episode_len))
                        imageio.mimwrite(os.path.join(outdir, 'eval_episode{}_{}.mp4'.format(episode_idx, i)), eval_episode_raw_obs, fps=60)

                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                done = False
                episode_raw_obs = [obs]
                n_random_actions = random.randint(0, 30)
                for _ in range(n_random_actions):
                    observation, reward, done, info = env.step(0)
                    episode_raw_obs.append(observation)
                assert not done
                if len(episode_raw_obs) == 1:
                    last_obs = None
                else:
                    last_obs = episode_raw_obs[-2]
                episode_obs = [phi(episode_raw_obs[-1], last_obs)]
                r = 0

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')


def run_eval_episode(agent, eval_env, eval_epsilon):
    eval_episode_r = 0
    eval_episode_len = 0
    obs = eval_env.reset()
    done = False
    eval_episode_raw_obs = [obs]
    n_random_actions = random.randint(0, 30)
    for _ in range(n_random_actions):
        observation, reward, done, info = eval_env.step(0)
        eval_episode_raw_obs.append(observation)
        eval_episode_r += reward
        eval_episode_len += 1
    assert not done
    if len(eval_episode_raw_obs) == 1:
        last_obs = None
    else:
        last_obs = eval_episode_raw_obs[-2]
    eval_episode_obs = [phi(eval_episode_raw_obs[-1], last_obs)]

    while True:
        if np.random.random() < eval_epsilon:
            action = eval_env.action_space.sample()
        else:
            # get current state
            state = eval_episode_obs[-4:]
            while len(state) < 4:
                state = [state[0].copy()] + state
            state = np.transpose(np.concatenate(state, axis=-1), (2, 0, 1))

            action = agent.act(state)

        for _ in range(4):
            obs, tmp_r, done, info = eval_env.step(action)
            eval_episode_raw_obs.append(obs)
            eval_episode_r += tmp_r
            eval_episode_len += 1
            if done:
                break
        eval_episode_obs.append(phi(eval_episode_raw_obs[-1], eval_episode_raw_obs[-2]))

        if done:
            break
    return eval_episode_raw_obs, eval_episode_r, eval_episode_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='chainerrl-results-envfix',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--max-episode-len', type=int,
                        default=5 * 60 * 60 // 4,  # 5 minutes with 60/4 fps
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--no-monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        env = gym.make(args.env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = links.Sequence(
        links.NatureDQNHead(activation=F.relu),
        L.Linear(512, n_actions),
        DiscreteActionValue)

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    agent = agents.DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                       explorer=explorer, replay_start_size=args.replay_start_size,
                       target_update_interval=args.target_update_interval,
                       update_interval=args.update_interval)

    if args.load:
        agent.load(args.load)
    else:
        logger = logging.getLogger(__name__)

        os.makedirs(args.outdir, exist_ok=True)

        train_agent(
            agent, env, args.steps, args.outdir,
            max_episode_len=args.max_episode_len,
            step_offset=0,
            eval_env=eval_env,
            logger=logger,
            eval_interval=args.eval_interval, eval_n_runs=args.eval_n_runs, eval_epsilon=args.eval_epsilon)


if __name__ == '__main__':
    main()
