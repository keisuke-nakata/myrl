import copy

import numpy as np
from chainer.serializers import save_hdf5


class BaseLearner:
    def __init__(self, network, gamma=0.99):
        self.network = network
        self.gamma = gamma

        self.n_steps = 0

    def learn(self, experiences):
        batch = self._experiences2batch(experiences)
        self._learn(batch)
        self.n_steps += 1

    def dump_parameters(self, path):
        save_hdf5(filename=path, obj=self.network)

    def _experiences2batch(self, experiences):
        raise NotImplementedError

    def _learn(self, batch):
        raise NotImplementedError


class FittedQLearner(BaseLearner):
    """Fitted Q learning, a.k.a Q learning with target network."""
    def __init__(self, *args, **kwargs):
        """
        target_network_update_freq
            corresponds to the parameter C from Algorithm 1 of the DQN paper.
            (measured in the number of parameter updates)
        """
        self.target_network_update_freq = kwargs.pop('target_network_update_freq', 10_000)
        super().__init__(*args, **kwargs)
        self.target_network = copy.deepcopy(self.network)

    def _experiences2batch(self, experiences):
        if self.n_steps % self.target_network_update_freq == 0:
            self._update_target_network()
        last_observations, actions, rewards, observations, dones = zip(*experiences)

        batch_x = np.array(observations)
        batch_target_q = self.target_network(np.array(last_observations)).data
        batch_y = np.array(rewards) + self.gamma * np.array(dones) * batch_target_q.max(axis=1)

        return (batch_x, batch_y)

    def _learn(self, batch):
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

# Give the optimizer a reference to the model so that it
# can locate the model's parameters.
optimizer.setup(model)



           prediction_train = model(image_train)

    # Calculate the loss with softmax_cross_entropy
    loss = F.softmax_cross_entropy(prediction_train, target_train)

    # Calculate the gradients in the network
    model.cleargrads()
    loss.backward()

    # Update all the trainable paremters
    optimizer.update()

    def _update_target_network(self):
        self.target_network.copyparams(self.network)
