import sys
import logging
import logging.config
import datetime as dt
import os

import gym
import toml

from myrl import algorithms


logger = logging.getLogger(__name__)

result_dir = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(result_dir, exist_ok=True)


def train(config, env_id):
    env = gym.make(env_id)

    agent = algorithms.vanilla_dqn.VanillaDQNAgent()
    agent.build(env, config)
    agent.train()


if __name__ == '__main__':
    logging_config = toml.load('logging.toml')
    logging_config['handlers']['file']['filename'] = logging_config['handlers']['file']['filename'].format(result_dir=result_dir)
    logging.config.dictConfig(logging_config)

    config = toml.load(sys.argv[1])
    env_id = sys.argv[2]  # Pong-v4

    logger.info('training start')
    train(config, env_id)
    logger.info('training end')
