import gym
import gym_fluids
import time
import torch
import numpy as np

from evograd import expectation
from evograd.distributions import Normal

HIDDEN_SIZE = 100


class CustomModel(torch.nn.Module):
    def __init__(self, weights, input_size):
        super(CustomModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, HIDDEN_SIZE)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x):
        h1 = self.relu1(self.linear1(x))
        y = self.linear2(h1)
        return y


def evaluate_weights(env, weights):
    total_reward = 0.0
    num_runs = 10
    num_steps_per_run = 1000

    model = CustomModel(weights, 2)
    for run_i in range(num_runs):
        obs = env.reset()
        for step_i in range(num_steps_per_run):
            action = model(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / num_runs


def simulate(env, batch_weights):
    rewards = []
    for weights in batch_weights:
        rewards.append(evaluate_weights(env, weights.numpy()))
    return torch.tensor(rewards)


def train(env, mu, std, alpha):
    p = Normal(mu, std)
    num_train_runs = 100
    for t in range(num_train_runs):
        sample_weights = p.sample(pop_size)
        fitnesses = simulate(env, sample_weights)
        scaled_fitnesses = (fitnesses - fitnesses.mean())/fitnesses.std()

        mean = expectation(scaled_fitnesses, sample_weights, p=p)
        mean.backward()

        with torch.no_grad():
            mu += alpha * mu
            mu.grad.zero_()


start_time = time.time()
env = gym.make("fluids-v2")
mu = torch.rand(HIDDEN_SIZE, requires_grad=True)
std = 0.5
pop_size = 50
alpha = 0.03

train(env, mu, std, alpha)


print("Training took {:.2f} seconds".format(time.time()-start_time))

# Wait for inupt to view the result
input("Press enter to view agent")

obs = env.reset()

action = [0, 0]
reward = 0
for i in range(1000):
    # action, _state = model.predict(obs)
    obs, rew, done, info = env.step(action)
    reward += rew
    env.render()
    print(rew, action)
