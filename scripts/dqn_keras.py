import gym
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def sum_huber_loss(y_true, y_pred, clip_delta=1.0):
    return K.sum(huber_loss(y_true, y_pred, clip_delta), axis=-1)


def build_network(n_actions):
    state = Input(shape=(84, 84, 4), name='state_inp')
    h = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(state)
    h = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(h)
    h = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(h)
    h = Dense(512, activation='relu')(h)
    out = Dense(n_actions, activation=None)(h)
    network = Model(inputs=state, outputs=out)

    optimizer = Adam(lr=1e-4)
    network.compile(optimizer=optimizer, loss=sum_huber_loss, metrics=['mae'], loss_weights=[32])
    return network


env = gym.make('PongNoFrameskip-v4')
n_actions = env.action_space.n
network = build_network(n_actions)
target_network = build_network(n_actions)
target_network.set_weights(network.get_weights())

limit = 100_000
replay = [None] * limit
replay_head = 0
full = False

initial_epsilon = 1.0

episode_loop = True
step_loop = True

step = 0
episode = 0
while episode_loop:  # episode
    step_loop = True
    episode += 1
    episode_step = 0
    episode_raw_obs = []
    episode_obs = []
    episode_reward = 0

    # reset the env
    obs = env.reset()
    episode_raw_obs.append(obs)
    n_random_actions = random.randint(0, 30)
    for _ in range(n_random_actions):
        obs, reward, done, info = env.step(0)
        episode_reward += reward
        episode_raw_obs.append(obs)
    if len(episode_raw_obs) == 1:
        last_obs = None
    else:
        last_obs = episode_raw_obs[-2]
    episode_obs.append(phi(obs, last_obs))

    while step_loop:  # step
        step += 1
        episode_step += 1

        if step > n_warmup:
            print('warmup end')

        # get current state
        state = episode_obs[-4:]
        while len(state) < 4:
            state = [state[0].copy()] + state

        # compute q_values and action
        epsilon = get_epsilon(step)
        if step <= n_warmup or np.random.random() < epsilon:
            is_random = True
        if is_random:
            action = env.action_space.sample()
        else:
            one_batch_state = np.expand_dims(np.concatenate(state / 255.0, axis=-1), 0)
            q_values = network.predict_on_batch(one_batch_state)[0]
            action = np.argmax(q_values)

        # interact with the env
        reward = 0
        for _ in range(4):
            obs, tmp_reward, done, info = env.step(action)
            reward += tmp_reward
            episode_raw_obs.append(obs)
            episode_reward += tmp_reward
            if done:
                break
        reward = np.sign(reward)
        episode_obs.append(phi(episode_raw_obs[-1], episode_raw_obs[-2]))

        # push to the replay
        replay[replay_head] = (state, action, reward, done)
        replay_head += 1
        if replay_head >= limit:
            replay_head = 0
            full = True

        if step > n_warmup and step % 4 == 0:
            # draw batch
            if full:
                end = limit
            else:
                end = replay_head - 1
            idxs = np.random.randint(end, size=32)  # `np.random.randint` is 5x faster than `np.random.choice` or `random.choices`.
            taboo = limit - 1 if replay_head == 0 else replay_head - 1  # current head points to the *next* index
            for i in range(32):
                while idxs[i] == taboo:
                    idxs[i] = np.random.randint(end)
            # NOTE: self.replay[idx + 1][0] may contain the next episode's state.
            # However such situation is allowed since `done` is True in that case.
            # If `next_state` has the special meaning when `done` is True, then fix this implementation.
            batch = [tuple(list(replay[idx]) + [replay[(idx + 1) % limit][0]]) for idx in idxs]
            batch_state = np.array([b[0] for b in batch])
            batch_action = np.array([b[1] for b in batch])
            batch_reward = np.array([b[2] for b in batch])
            batch_done = np.array([b[3] for b in batch])
            batch_next_state = np.array([b[4] for b in batch])

            # update network
            q_values = network.predict_on_batch(batch_state)
            batch_y = q_values.copy()
            target_next_q_values = target_network.predict_on_batch(batch_next_state)
            max_target_next_q_values = np.max(target_next_q_values, axis=1)
            for i, (action, reward, done, max_q) in enumerate(zip(batch_action, batch_reward, batch_done, max_target_next_q_values)):
                batch_y[i][action] = reward
                if not done:
                    batch_y[i][action] += 0.99 * max_q
            loss = network.train_on_batch(batch_state, batch_y)

        if step > n_warmup and step % 1000 == 0:
            target_network.set_weights(network.get_weights())

        if done:
            step_loop = False
            print(f'episode {episode}, reward {episode_reward}, step {len(episode_obs)}')

        if step > n_steps:
            step_loop = False
            episode_loop = False
