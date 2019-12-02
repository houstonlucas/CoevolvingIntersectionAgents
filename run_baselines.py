import gym
import gym_fluids
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


start_time = time.time()
env = gym.make("fluids-v2")

env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
print("Training took {:.2f} seconds".format(time.time()-start_time))

# Wait for inupt to view the result
input("Press enter to view agent")

obs = env.reset()

action = [0, 0]
reward = 0
for i in range(1000):
    action, _state = model.predict(obs)
    obs, rew, done, info = env.step(action)
    reward += rew
    env.render()
    print(rew, action)
