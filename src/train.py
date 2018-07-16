import logging
import logging.config
import datetime as dt
import os
import traceback

import click
import toml

from myrl import algorithms
from visualize import _visualize


logger = logging.getLogger(__name__)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('env_id')
@click.argument('agent_name')
@click.option('--device', default=0, show_default=True, help='device id. -1: CPU, >=0: GPU (s).')
def train(config_path, env_id, agent_name, device):
    """
    CONFIG_PATH: config filepath, e.g.: configs/vanilla_dqn.toml\n
    ENV_ID: OpenAI Gym environment id, e.g.: PongNoFrameskip-v4\n
    AGENT_NAME: Agent class name, e.g.: VanillaDQNAgent
    """
    config = toml.load(config_path)

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = config['result_dir'].format(env_id=env_id, now=now)
    config['result_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'config.toml'), 'w') as f:
        toml.dump(config, f)

    logging_config = toml.load('logging.toml')
    log_filename = logging_config['handlers']['file']['filename'].format(result_dir=result_dir)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging_config['handlers']['file']['filename'] = log_filename
    logging.config.dictConfig(logging_config)

    try:
        if agent_name == 'VanillaDQNAgent':
            agent = algorithms.vanilla_dqn.VanillaDQNAgent()
        elif agent_name == 'AsyncDQNAgent':
            agent = algorithms.async_dqn.AsyncDQNAgent()
        else:
            raise ValueError(f'Unknown agent: {agent_name}')
        agent.build(config, env_id, device)
        logger.info('training start')
        agent.train()
        logger.info('training end')
    except:  # noqa
        logger.exception('train failed')
        with open(os.path.join(result_dir, 'error.txt'), 'a') as f:
            traceback.print_exc(file=f)
    finally:
        _visualize(os.path.join(result_dir, 'history.csv'), os.path.join(result_dir, 'history.png'), window=10)
        _visualize(os.path.join(result_dir, 'greedy', 'history.csv'), os.path.join(result_dir, 'greedy', 'history.png'), window=10)


if __name__ == '__main__':
    train()
