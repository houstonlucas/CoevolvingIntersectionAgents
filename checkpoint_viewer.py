
import neat
import numpy as np

import visualize

import os
import gym
import gym_fluids
from multiprocessing import Pool


env_name = 'fluids-1-v2'
checkpoint_to_restore = 'neat-checkpoint-36'


def eval_genomes(genomes, config):
    p = Pool(10)
    inputs = [(genome_id, genome, config) for genome_id, genome in genomes]
    id_fitnesses = p.map(run_set, inputs)
    p.close()
    for genome_id, genome in genomes:
        for id, fitness in id_fitnesses:
            if genome_id == id:
                genome.fitness = fitness
                break


def run_set(triplet):
    genome_id, genome, config = triplet
    env = gym.make(env_name)
    genome.fitness = 0.0
    num_runs = 1
    num_steps_per_run = 1000
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for run_i in range(num_runs):
        genome.fitness += single_run(net, env, num_steps_per_run)
    genome.fitness /= num_runs
    return genome_id, genome.fitness


def single_run(net, env, num_steps_per_run):
    total_reward = 0.0
    obs = env.reset()
    for step_i in range(num_steps_per_run):
        action = net.activate(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint(checkpoint_to_restore)
    winner = p.run(eval_genomes, 1)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = gym.make(env_name)
    num_runs = 2
    num_steps_per_run = 1000
    total_reward = 0.0
    for run_i in range(num_runs):
        obs = env.reset()
        for step_i in range(num_steps_per_run):
            action = winner_net.activate(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break

    print("Average reward per run: {}".format(total_reward / num_runs))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
