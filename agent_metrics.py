import pickle
import neat
import numpy as np
import fluids
import os
import gym
import gym_fluids
import collections

only_accel = False
path = '/home/gaetano/Desktop/winners/'
winner_name = 'winner_set3.pkl'
num_steps_per_run = 1000

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    with open(path + winner_name, 'rb') as f:
        winner_net = pickle.load(f)

    envs = ['fluids-1-v2', 'fluids-2-v2', 'fluids-3-v2', 'fluids-4-v2', 'fluids-5-v2', 'fluids-6-v2']
    meta_metrics = []

    for env_name in envs:
        total_reward = 0.0
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
        metrics["final_reward"] = reward
        meta_metrics.append(metrics)

    counter = collections.Counter()
    for d in meta_metrics:
        counter.update(d)

    result = dict(counter)

    print("resultant dictionary : ", str(counter))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
