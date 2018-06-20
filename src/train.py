import logging
import logging.config
import datetime as dt
import os
import traceback

import click
import gym
import toml

from myrl import algorithms
from myrl.env_wrappers import SuddenDeathWrapper, RewardClippingWrapper


logger = logging.getLogger(__name__)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('env_id')
def train(config_path, env_id):
    """
    CONFIG_PATH: config filepath, e.g.: configs/vanilla_dqn.toml\n
    ENV_ID: OpenAI Gym environment id, e.g.: Pong-v4
    """
    config = toml.load(config_path)
    _env = gym.make(env_id)  # Pong-v4

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = config['result_dir'].format(env_id=_env.spec.id, now=now)
    config['result_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)

    logging_config = toml.load('logging.toml')
    log_filename = logging_config['handlers']['file']['filename'].format(result_dir=result_dir)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging_config['handlers']['file']['filename'] = log_filename
    logging.config.dictConfig(logging_config)

    try:
        env = RewardClippingWrapper(_env)
        # env = SuddenDeathWrapper(env)

        agent = algorithms.vanilla_dqn.VanillaDQNAgent()
        agent.build(config, env)
        logger.info('training start')
        agent.train()
        logger.info('training end')
    except:  # noqa
        logger.exception('train failed')
        with open(os.path.join(result_dir, 'error.txt'), 'a') as f:
            traceback.print_exc(file=f)


if __name__ == '__main__':
    train()
