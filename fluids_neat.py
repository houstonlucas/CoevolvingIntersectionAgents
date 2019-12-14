import fluids
import neat
import numpy as np

import visualize
import pickle

import os
import gym
import gym_fluids
from matplotlib import pyplot as plt
from multiprocessing import Pool
from MetricsRecorder import MetricsRecorder
from collections import defaultdict

env_names = ['fluids-1-v2', 'fluids-2-v2', 'fluids-3-v2', 'fluids-4-v2', 'fluids-5-v2', 'fluids-6-v2']
only_accel = False
num_sets = 40

recorder = None


def eval_genomes(genomes, config):
    p = Pool(10)
    inputs = [(genome_id, genome, config) for genome_id, genome in genomes]
    id_fitnesses = p.map(run_set, inputs)
    p.close()
    for genome_id, genome in genomes:
        for id, fitness, metrics in id_fitnesses:
            if genome_id == id:
                genome.fitness = fitness
                break
    fitnesses = [triplet[1] for triplet in id_fitnesses]
    metrics = [triplet[2] for triplet in id_fitnesses]
    recorder.record(fitnesses, metrics)


def run_set(triplet):
    genome_id, genome, config = triplet
    genome.fitness = 0.0
    metrics = defaultdict(lambda: 0.0)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for env_name in env_names:
        env = gym.make(env_name)
        num_steps_per_run = 1000
        total_reward, run_metrics = single_run(net, env, num_steps_per_run)
        for key in run_metrics:
            metrics[key] += run_metrics[key]
        genome.fitness += total_reward
    genome.fitness /= len(env_names)
    return genome_id, genome.fitness, dict(metrics)


def single_run(net, env, num_steps_per_run):
    total_reward = 0.0
    obs = env.reset()
    car_keys = env.env.fluidsim.get_control_keys()
    controlled_key = list(car_keys)[0]
    for step_i in range(num_steps_per_run):
        action = net.activate(obs)
        if only_accel:
            actions = env.env.fluidsim.get_supervisor_actions(fluids.SteeringVelAction,
                                                              keys=car_keys)
            action[0] = actions[controlled_key].steer
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    # Get the metrics to return
    car = env.env.fluidsim.state.controlled_cars[controlled_key]
    metrics = {}
    metrics["collisions"] = car.total_collisions
    metrics["infractions"] = car.total_infractions
    metrics["livelieness"] = car.total_liveliness
    metrics["jerk"] = car.total_jerk
    metrics["traj_following"] = car.total_traj_follow
    return total_reward, metrics


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    global recorder

    for set_number in range(num_sets):
        recorder = MetricsRecorder(set_number)
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run for up to 300 generations.
        winner = p.run(eval_genomes, 300)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        # Save out best net
        pickle.dump(winner_net, open("winners/winner_set{}.pkl".format(set_number), "wb+"))
        # Save out metrics
        recorder.write_out()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
