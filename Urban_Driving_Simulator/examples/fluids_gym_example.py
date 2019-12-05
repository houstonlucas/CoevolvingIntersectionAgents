import gym
import gym_fluids
import PIL
from matplotlib import pyplot as plt

env = gym.make('fluids-1-v2')
env.reset()

action = [0, 0]
reward = 0
while True:
    obs, rew, done, info = env.step(action)
    reward += rew

    # plt.imshow(obs)
    # plt.show()
    # plt.close()

    env.render()
    print(rew, action)
    action = gym_fluids.agents.fluids_supervisor(obs, info)
