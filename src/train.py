import gym

import algorithms


def train():
    env = gym.make(config['env']['env'])

    agent = algorithms.vanilla_dqn.VanillaDQNAgent()
    agent.build(env, config)
    agent.train()


if __name__ == '__main__':
    train()
