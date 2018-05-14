from chainer.serializers import save_hdf5


class BaseLearner:
    def __init__(self, network):
        self.network = network

    def learn(self):
        raise NotImplementedError

    def dump_network(self, path):
        save_hdf5(filename=path, obj=self.network)


class QLearner(BaseLearner):
    def learn(self):
        pass
