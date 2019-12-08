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

HIDDEN_SIZE = 50
NUM_CHANNELS1 = 4
NUM_CHANNELS2 = 4
OUTPUT_SIZE = 1
IM_SIZE = 40

# Weight vector sized empirically
WEIGHT_SIZE = 20361


class CustomModel(torch.nn.Module):
    def __init__(self, weights, num_channels=3):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, NUM_CHANNELS1, kernel_size=3, stride=1, padding=1)
        self.conv1w_size = 3 * 3 * 3 * NUM_CHANNELS1
        self.conv1b_size = NUM_CHANNELS1

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(NUM_CHANNELS1, NUM_CHANNELS2, kernel_size=3, stride=1, padding=1)
        self.conv2w_size = NUM_CHANNELS1 * 3 * 3 * NUM_CHANNELS2
        self.conv2b_size = NUM_CHANNELS2

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        conv_to_linear_size = NUM_CHANNELS2 * (IM_SIZE * IM_SIZE // 16)

        self.linear = torch.nn.Linear(conv_to_linear_size, HIDDEN_SIZE)
        self.w1_size = HIDDEN_SIZE * conv_to_linear_size
        self.b1_size = HIDDEN_SIZE

        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.w2_size = OUTPUT_SIZE * HIDDEN_SIZE
        self.b2_size = OUTPUT_SIZE

        a = self.get_weights()
        self.set_weights(weights)

    def forward(self, x):
        x = cv2.resize(x, (IM_SIZE, IM_SIZE)) / 255.0
        x = torch.Tensor(x).permute(2, 0, 1).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(1, -1)
        x = F.tanh(self.linear(x))
        x = 2 * F.tanh(self.linear2(x))
        return x

    def get_weights(self):
        c1w = self.conv1.weight.view(-1)
        c1b = self.conv1.bias.view(-1)
        c2w = self.conv2.weight.view(-1)
        c2b = self.conv2.bias.view(-1)
        w1 = self.linear.weight.view(-1)
        b1 = self.linear.bias.view(-1)
        w2 = self.linear2.weight.view(-1)
        b2 = self.linear2.bias.view(-1)
        return torch.cat([c1w, c1b, c2w, c2b, w1, b1, w2, b2])

    def set_weights(self, weights):
        start = 0
        end = self.conv1w_size
        self.conv1.weight = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv1.weight.shape))

        start = end
        end += self.conv1b_size
        self.conv1.bias = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv1.bias.shape))

        start = end
        end += self.conv2w_size
        self.conv2.weight = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv2.weight.shape))

        start = end
        end += self.conv2b_size
        self.conv2.bias = nn.Parameter(torch.tensor(weights[start:end]).reshape(*self.conv2.bias.shape))

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
    num_runs = 3
    num_steps_per_run = 500

    model = CustomModel(weights)

    for run_i in range(num_runs):
        obs = env.reset()
        for step_i in range(num_steps_per_run):
            img = env.render(mode="rgb_array")
            action = model(img).data.numpy()[0]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / num_runs


def simulate(env, batch_weights):
    rewards = []
    for weights in batch_weights:
        rewards.append(evaluate_weights(env, weights.numpy()))
    print("Reward avg: {}".format(np.average(rewards)))
    return torch.tensor(rewards)


def train(env, mu, std, alpha):
    p = Normal(mu, std)
    num_train_runs = 500
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
env = gym.make("Pendulum-v0")
mu = torch.randn(WEIGHT_SIZE, requires_grad=True)
std = 0.05
pop_size = 20
alpha = 0.01

train(env, mu, std, alpha)

print("Training took {:.2f} seconds".format(time.time() - start_time))

# Wait for inupt to view the result
input("Press enter to view agent")

obs = env.reset()

action = [0, 0]
reward = 0
done = False
p = Normal(mu, std)
model = CustomModel(p.sample(1)[0])
while not done:
    img = env.render(mode="rgb_array")
    action = model(img).data.numpy()[0]
    action = int(np.clip(action, 0, 1)[0])
    obs, rew, done, info = env.step(action)
    reward += rew
    # env.render()

print("Finished")
