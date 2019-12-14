import gym
import gym_fluids

env = gym.make('fluids-1-v2')
env.reset()

action = [0, 0]
reward = 0
done = False
while True:
    obs, rew, done, info = env.step(action)
    reward += rew
    env.render()

    if done:
        break
    action = gym_fluids.agents.fluids_supervisor(obs, info)
print("Done")
