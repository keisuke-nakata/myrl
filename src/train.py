import algorithms


def train():
    agent = algorithms.vanilla_dqn.VanillaDQNAgent()
    agent.build(config)
    agent.train()


if __name__ == '__main__':
    train()
