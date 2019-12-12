from gym.envs.registration import register
import gym_fluids.agents
register(
    id="fluids-v2",
    entry_point="gym_fluids.envs:FluidsEnv",
    max_episode_steps=1000,
    nondeterministic=True,
    )

register(
    id="fluids-1-v2",
    entry_point="gym_fluids.envs:FluidsEnv1",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-2-v2",
    entry_point="gym_fluids.envs:FluidsEnv2",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-3-v2",
    entry_point="gym_fluids.envs:FluidsEnv3",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-4-v2",
    entry_point="gym_fluids.envs:FluidsEnv4",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-5-v2",
    entry_point="gym_fluids.envs:FluidsEnv5",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-6-v2",
    entry_point="gym_fluids.envs:FluidsEnv6",
    max_episode_steps=1000,
    nondeterministic=True
    )

register(
    id="fluids-vel-v2",
    entry_point="gym_fluids.envs:FluidsVelEnv",
    max_episode_steps=1000,
    nondeterministic=True
    )