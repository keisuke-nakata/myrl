from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
gym.undo_logger_setup()  # NOQA
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl.experiments.evaluator import Evaluator, save_agent
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.misc.makedirs import makedirs
from chainerrl import replay_buffer

import atari_wrappers


def train_agent(agent, env, steps, outdir, max_episode_len=None,
                step_offset=0, evaluator=None,
                step_hooks=[], logger=None):
    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()
    r = 0

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:

            # a_t
            action = agent.act_and_train(obs, r)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1

            for hook in step_hooks:
                hook(env, agent, t)

            if done or episode_len == max_episode_len or t == steps:
                agent.stop_episode_and_train(obs, r, done=done)
                epsilon = agent.explorer.compute_epsilon(agent.t)
                logger.info('outdir:{} step:{:,} episode:{} R:{} episode_step:{} epsilon:{}'.format(outdir, t, episode_idx, episode_r, episode_len, epsilon))
                logger.info('statistics:%s', agent.get_statistics())
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(t=t, episodes=episode_idx + 1)
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                r = 0

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='chainerrl-results-unrolled2',
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

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        atari_env = atari_wrappers.make_atari(args.env)
        env = atari_wrappers.wrap_deepmind(atari_env, episode_life=not test, clip_rewards=not test)
        env.seed(int(env_seed))
        if not args.no_monitor:
            env = gym.wrappers.Monitor(env, args.outdir, mode='evaluation' if test else 'training')
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
        # In testing DQN, randomly select 5% of actions
        eval_explorer = explorers.ConstantEpsilonGreedy(args.eval_epsilon, lambda: np.random.randint(n_actions))

        logger = logging.getLogger(__name__)

        makedirs(args.outdir, exist_ok=True)

        if eval_env is None:
            eval_env = env

        eval_max_episode_len = args.max_episode_len

        evaluator = Evaluator(agent=agent,
                              n_runs=args.eval_n_runs,
                              eval_interval=args.eval_interval, outdir=args.outdir,
                              max_episode_len=eval_max_episode_len,
                              explorer=eval_explorer,
                              env=eval_env,
                              step_offset=0,
                              save_best_so_far_agent=False,
                              logger=logger,
                              )

        train_agent(
            agent, env, args.steps, args.outdir,
            max_episode_len=args.max_episode_len,
            step_offset=0,
            evaluator=evaluator,
            step_hooks=[],
            logger=logger)


if __name__ == '__main__':
    main()
