import logging

from . import dqn, async_dqn

logger = logging.getLogger(__name__)


def build_agent(config, env_id, device):
    agent_name = config['agent_name']
    if agent_name == 'DQNAgent':
        agent = dqn.DQNAgent(config, env_id, device)
    elif agent_name == 'AsyncDQNAgent':
        agent = async_dqn.AsyncDQNAgent(config, env_id, device)
    else:
        raise ValueError(f'Unknown agent: {agent_name}')
    agent.build()
    logger.info(f'built agent {agent}.')
    return agent
