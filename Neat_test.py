import os

import gym
import gym_fluids
import time

import neat
import numpy as np

import visualize

env_name = "Pendulum-v0"


def eval_genomes(genomes, config):
    env = gym.make(env_name)
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        num_runs = 5
        num_steps_per_run = 1000
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for run_i in range(num_runs):
            obs = env.reset()
            for step_i in range(num_steps_per_run):
                action = 2.0*np.array(net.activate(obs))
                # action = 0 if action < 0 else 1
                obs, reward, done, _ = env.step(action)
                genome.fitness += reward
                if done:
                    break
        genome.fitness /= num_runs
        genome.fitness += 2000.0


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(20))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    input("Press enter to view output")

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = gym.make(env_name)
    num_runs = 5
    num_steps_per_run = 1000
    total_reward = 0.0
    for run_i in range(num_runs):
        obs = env.reset()
        for step_i in range(num_steps_per_run):
            action = 2.0 * np.array(winner_net.activate(obs))
            # action = 0 if action < 0 else 1
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break

    print(total_reward/num_runs)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
