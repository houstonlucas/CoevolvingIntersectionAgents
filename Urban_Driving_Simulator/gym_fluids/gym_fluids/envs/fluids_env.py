import fluids
import pygame
import numpy as np
import gym
from gym import spaces

OBS_W = 400
TIME_LIMIT = 600

FLUIDS_ARGS = {"visualization_level": 3,
               "fps": 30,
               "obs_args": {"obs_dim": OBS_W},
               "obs_space": fluids.OBS_VEC,
               "background_control": fluids.BACKGROUND_CSP,
               "obeys_lights": True,
               "obeys_cars": True}

STATE_ARGS = {"layout": fluids.STATE_CITY,
              "background_cars": 10,
              "controlled_cars": 1,
              "background_peds": 0,
              }

STATE_ARGS_1 = {"layout": fluids.STATE_CITY,
                "background_cars": 0,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(4, .5, 0.0)],
                "set_path": [2],
                }

STATE_ARGS_2 = {"layout": fluids.STATE_CITY,
                "background_cars": 1,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(1, .5, 0.0), (2, .75, 0.0)],
                "set_path": [1, 0],
                }

STATE_ARGS_3 = {"layout": fluids.STATE_CITY,
                "background_cars": 1,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(2, .25, 0.0), (1, .75, 0.0)],
                "set_path": [2, 1],
                }

STATE_ARGS_4 = {"layout": fluids.STATE_CITY,
                "background_cars": 1,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(4, .5, 0.0), (7, .5, 0.0)],
                "set_path": [1, 2],
                }

STATE_ARGS_5 = {"layout": fluids.STATE_CITY,
                "background_cars": 1,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(4, .5, 0.0), (7, .5, 0.0)],
                "set_path": [1, 0],
                }

STATE_ARGS_6 = {"layout": fluids.STATE_CITY,
                "background_cars": 1,
                "controlled_cars": 1,
                "background_peds": 0,
                "set_car": [(1, .25, 0.0), (4, .5, 0.0)],
                "set_path": [0, 2],
                }


class FluidsEnv(gym.Env):
    def __init__(self):
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()

    def fluids_action_maker(self, action):
        return fluids.SteeringAccAction(action[0], action[1])

    def step(self, action):
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        actions = {car_keys[0]: self.fluids_action_maker(action)}
        reward_step, done = self.fluidsim.step(actions)
        obs = self.fluidsim.get_observations(car_keys)

        done = self.fluidsim.run_time() > TIME_LIMIT or done
        obs = obs[car_keys[0]].get_array()
        info_dict = {"supervisor_action": self.fluidsim.get_supervisor_actions(keys=car_keys,
                                                                               action_type=self.fluids_action_type)
        [car_keys[0]].get_array()}
        return obs, reward_step, done, info_dict

    def render(self, mode='human'):
        self.fluidsim.render()


class FluidsEnv1(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = True
        FLUIDS_ARGS["obeys_cars"] = True
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_1))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsEnv2(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = True
        FLUIDS_ARGS["obeys_cars"] = False
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_2))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsEnv3(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = True
        FLUIDS_ARGS["obeys_cars"] = False
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_3))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsEnv4(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = True
        FLUIDS_ARGS["obeys_cars"] = False
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_4))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsEnv5(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = True
        FLUIDS_ARGS["obeys_cars"] = False
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_5))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsEnv6(FluidsEnv):
    def __init__(self):
        FLUIDS_ARGS["obeys_lights"] = False
        FLUIDS_ARGS["obeys_cars"] = False
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)
        self.fluids_action_type = fluids.SteeringAccAction

    def reset(self):
        self.fluidsim.set_state(fluids.State(**STATE_ARGS_6))
        car_keys = list(self.fluidsim.get_control_keys())
        assert (len(car_keys) == 1)
        obs = self.fluidsim.get_observations(car_keys)
        return obs[car_keys[0]].get_array()


class FluidsVelEnv(FluidsEnv):
    def __init__(self):
        super(FluidsVelEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32)
        self.fluids_action_type = fluids.VelocityAction

    def fluids_action_maker(self, action):
        return fluids.VelocityAction(action[0])
