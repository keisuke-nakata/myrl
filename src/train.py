import sys

import gym
import toml

from myrl import algorithms


def train(config, env_id):
    env = gym.make(env_id)

    agent = algorithms.vanilla_dqn.VanillaDQNAgent()
    agent.build(env, config)
    agent.train()


if __name__ == '__main__':
    config = toml.load(sys.argv[1])
    env_id = sys.argv[2]  # Pong-v4
    train(config, env_id)
