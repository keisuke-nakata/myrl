import logging
import logging.config
import datetime as dt
import os
import traceback

import click
import toml

from myrl.agents import build_agent
from myrl.utils import visualize


logger = logging.getLogger(__name__)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('env_id')
@click.option('--device', default=0, show_default=True, help='device id. -1: CPU, >=0: GPU (s).')
def train(config_path, env_id, device):
    """
    CONFIG_PATH: config filepath, e.g.: myrl/configs/vanilla_dqn.toml\n
    ENV_ID: OpenAI Gym environment id, e.g.: PongNoFrameskip-v4\n
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
        agent = build_agent(config, env_id, device)
        logger.info('training start')
        csv_path, test_csv_path = agent.train()
        logger.info('training end')
    except:  # noqa
        logger.exception('train failed')
        with open(os.path.join(result_dir, 'error.txt'), 'a') as f:
            traceback.print_exc(file=f)
    else:
        visualize(csv_path)
        visualize(test_csv_path)


if __name__ == '__main__':
    train()
