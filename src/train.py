import logging
import logging.config
import datetime as dt
import os

import click
import gym
import toml

from myrl import algorithms
from myrl.env_wrappers import SuddenDeathWrapper


logger = logging.getLogger(__name__)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('env_id')
def train(config_path, env_id):
    """
    CONFIG_PATH: config filepath, e.g.: configs/vanilla_dqn.toml\n
    ENV_ID: OpenAI Gym environment id, e.g.: Pong-v4
    """
    result_dir = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(result_dir, exist_ok=True)

    logging_config = toml.load('logging.toml')
    logging_config['handlers']['file']['filename'] = logging_config['handlers']['file']['filename'].format(result_dir=result_dir)
    logging.config.dictConfig(logging_config)

    try:
        config = toml.load(config_path)
        config = set_result_dir(config, result_dir)

        _env = gym.make(env_id)  # Pong-v4
        env = SuddenDeathWrapper(_env)

        agent = algorithms.vanilla_dqn.VanillaDQNAgent()
        agent.build(config, env)
        logger.info('training start')
        agent.train()
        logger.info('training end')
    except:  # noqa
        logger.exception('train failed')


def set_result_dir(nested_dict, result_dir):
    for k, v in nested_dict.items():
        if isinstance(v, str):
            nested_dict[k] = v.format(result_dir=result_dir)
        elif isinstance(v, dict):
            nested_dict[k] = set_result_dir(v, result_dir)
    return nested_dict


if __name__ == '__main__':
    train()
