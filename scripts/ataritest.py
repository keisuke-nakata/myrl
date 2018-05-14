import gym
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# v0 = gym.make('Pong-v0')
# v0.reset()
# for idx, i in enumerate([2, 5] * 100):
#     v0.step(i)
#     plt.imshow(v0.render('rgb_array'))
#     plt.savefig('v0_{}.png'.format(idx))


v4 = gym.make('Pong-v4')
v4.reset()
for idx, i in enumerate([2, 5] * 100):
    v4.step(i)
    plt.imshow(v4.render('rgb_array'))
    plt.savefig('v4_{}.png'.format(idx))
