from gym.envs.registration import register
import gym_fluids.agents
register(
    id="fluids-v2",
    entry_point="gym_fluids.envs:FluidsEnv",
    max_episode_steps=1000,
    nondeterministic=True,
    )

register(
    id="fluids-vel-v2",
    entry_point="gym_fluids.envs:FluidsVelEnv",
    max_episode_steps=1000,
    nondeterministic=True
    )