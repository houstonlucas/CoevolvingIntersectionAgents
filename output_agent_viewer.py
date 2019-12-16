import pickle
import neat
import numpy as np
import fluids
import os
import gym
import gym_fluids
from multiprocessing import Pool

env_name = 'fluids-3-v2'
only_accel = False
path = '/home/gaetano/Desktop/winners/'
winner_name = 'winner_set3.pkl'


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    with open(path + winner_name, 'rb') as f:
        winner_net = pickle.load(f)
    total_reward = 0.0
    num_steps_per_run = 1000

    env = gym.make(env_name)
    obs = env.reset()

    car_keys = env.env.fluidsim.get_control_keys()
    controlled_key = list(car_keys)[0]

    for step_i in range(num_steps_per_run):
        action = winner_net.activate(obs)
        if only_accel:
            actions = env.env.fluidsim.get_supervisor_actions(fluids.SteeringVelAction,
                                                              keys=car_keys)
            action[0] = actions[list(car_keys)[0]].steer
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
