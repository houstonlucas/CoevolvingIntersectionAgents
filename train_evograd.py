import gym
import gym_fluids
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from evograd import expectation
from evograd.distributions import Normal

HIDDEN_SIZE = 100
OUTPUT_CHANNELS = 4
IM_SIZE = 100


# Weight vector sized empirically
WEIGHT_SIZE = 1000414

class CustomModel(torch.nn.Module):
    def __init__(self, weights, num_channels=3):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.conv1w_size = 3*3*3*OUTPUT_CHANNELS
        self.conv1b_size = OUTPUT_CHANNELS

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear = torch.nn.Linear(OUTPUT_CHANNELS * IM_SIZE * IM_SIZE // 4, HIDDEN_SIZE)
        self.w1_size = 100*10000
        self.b1_size = 100

        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, 2)
        self.w2_size = 2*100
        self.b2_size = 2

        self.set_weights(weights)

    def forward(self, x):
        x = cv2.resize(x, (IM_SIZE, IM_SIZE))
        x = torch.Tensor(x).permute(2, 0, 1).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = x.view(1, -1)
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x

    def get_weights(self):
        cw = self.conv1.weight.view(-1)
        cb = self.conv1.bias.view(-1)
        w1 = self.linear.weight.view(-1)
        b1 = self.linear.bias.view(-1)
        w2 = self.linear2.weight.view(-1)
        b2 = self.linear2.bias.view(-1)
        return torch.cat([cw, cb, w1, b1, w2, b2])

    def set_weights(self, weights):
        start = 0
        end = self.conv1w_size
        self.conv1.weight = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv1.weight.shape))

        start = end
        end += self.conv1b_size
        self.conv1.bias = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv1.bias.shape))

        start = end
        end += self.w1_size
        self.linear.weight = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.linear.weight.shape))

        start = end
        end += self.b1_size
        self.linear.bias = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.linear.bias.shape))

        start = end
        end += self.w2_size
        self.linear2.weight = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.linear2.weight.shape))

        start = end
        end += self.b2_size
        self.linear2.bias = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.linear2.bias.shape))


def evaluate_weights(env, weights):
    total_reward = 0.0
    num_runs = 2
    num_steps_per_run = 1000

    model = CustomModel(weights)

    for run_i in range(num_runs):
        obs = env.reset()
        for step_i in range(num_steps_per_run):
            action = model(obs).data.numpy()[0]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / num_runs


def simulate(env, batch_weights):
    rewards = []
    for weights in batch_weights:
        rewards.append(evaluate_weights(env, weights.numpy()))
    print("Reward: {}".format(sum(rewards)))
    return torch.tensor(rewards)


def train(env, mu, std, alpha):
    p = Normal(mu, std)
    num_train_runs = 5
    for t in range(num_train_runs):
        sample_weights = p.sample(pop_size)
        fitnesses = simulate(env, sample_weights)
        scaled_fitnesses = (fitnesses - fitnesses.mean()) / fitnesses.std()

        mean = expectation(scaled_fitnesses, sample_weights, p=p)
        mean.backward()

        with torch.no_grad():
            mu += alpha * mu
            mu.grad.zero_()


start_time = time.time()
env = gym.make("fluids-v2")
mu = torch.rand(WEIGHT_SIZE, requires_grad=True)
std = 0.5
pop_size = 5
alpha = 0.03

train(env, mu, std, alpha)

print("Training took {:.2f} seconds".format(time.time() - start_time))

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
