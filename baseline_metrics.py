import gym
import gym_fluids
import collections, functools, operator

envs = ['fluids-1-v2', 'fluids-2-v2', 'fluids-3-v2', 'fluids-4-v2', 'fluids-5-v2', 'fluids-6-v2']
meta_metrics = []

for i, env_name in enumerate(envs):
    env = gym.make(env_name)
    env.reset()

    car_keys = env.env.fluidsim.get_control_keys()
    controlled_key = list(car_keys)[0]

    action = [0, 0]
    reward = 0
    done = False
    while True:
        obs, rew, done, info = env.step(action)
        reward += rew

        if done:
            env.close()
            break
        # env.render()
        action = gym_fluids.agents.fluids_supervisor(obs, info)
    env.close()

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

print("Done")
