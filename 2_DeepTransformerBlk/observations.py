import numpy as np

from utils import add_channel_dim

"""
observations = {agent_id: obs}

obs = [
    red_normalized_force
    red_efficiency
    blue_normalized_force
    blue_efficiency
"""


def get_observations(env):
    """ Select one of the followings """

    observations = get_observations_po_0(env)  # 6 channels, new observation scale

    return observations


def get_observations_po_0(env):
    """
    observations: {agent-id: (env.config.fov * 2 + 1,
                              env.config.fov * 2 + 1,
                              env.config.observation_channels)}
     normalized by arc-tan().  4CH
    """

    if env.config.observation_channels != 4:
        raise ValueError()

    fov = env.config.fov
    observations = {}

    for red in env.reds:
        if red.alive:
            myx = red.pos[0]
            myy = red.pos[1]

            red_normalized_force, red_efficiency = \
                compute_partial_observation_map(myx, myy, env.reds, fov)

            blue_normalized_force, blue_efficiency = \
                compute_partial_observation_map(myx, myy, env.blues, fov)

            # transform to float32 & add channel dim
            red_normalized_force = add_channel_dim(red_normalized_force)
            red_efficiency = add_channel_dim(red_efficiency)

            blue_normalized_force = add_channel_dim(blue_normalized_force)
            blue_efficiency = add_channel_dim(blue_efficiency)

            observations[red.id] = np.concatenate(
                [
                    red_normalized_force,
                    red_efficiency,
                    blue_normalized_force,
                    blue_efficiency,
                ], axis=2)  # (5,5,4)

    return observations


def compute_partial_observation_map(myx, myy, agents, fov):
    """
    (myx, myy) : myself global pos
    agents: reds or blues
    fov: field of view

    :return: normalized_force_map, efficiency_map
    """

    force_map = np.zeros((2 * fov + 1, 2 * fov + 1))
    ef_map = np.zeros((2 * fov + 1, 2 * fov + 1))

    for x in range(myx - fov, myx + fov + 1):
        for y in range(myy - fov, myy + fov + 1):
            for agent in agents:
                if agent.alive and (agent.pos[0] == x) and (agent.pos[1] == y):
                    force_map[agent.pos[0] - myx + fov, agent.pos[1] - myy + fov] += agent.force
                    ef_map[agent.pos[0] - myx + fov, agent.pos[1] - myy + fov] += agent.ef

    # normalize
    alpha = 50.
    normalized_force_map = 2.0 / np.pi * np.arctan(force_map / alpha)

    efficiency_map = np.zeros((2 * fov + 1, 2 * fov + 1))

    xf, yf = np.nonzero(force_map)
    for (x, y) in zip(xf, yf):
        efficiency_map[x, y] = ef_map[x, y] / force_map[x, y]

    return normalized_force_map, efficiency_map
