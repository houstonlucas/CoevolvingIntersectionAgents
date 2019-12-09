COLLISION_WEIGHT = 500.0
INFRACTION_WEIGHT = 500.0
LIVELINESS_WEIGHT = 1.0
JERK_WEIGHT = 1.0
TRAJ_WEIGHT = 1.0


def path_reward(state):
    # Get cars
    controlled_cars = [state.objects[k] for k in state.controlled_cars.keys()]

    # Loop over every car
    for c in controlled_cars:
        # Calc collisions
        if state.is_in_collision(c):
            collision = -1.0
            c.total_collisions += 1
        else:
            collision = 0.0

        # Calc infractions
        if state.is_in_infraction(c):
            infraction = -1.0
            c.total_infractions += 1
        else:
            infraction = 0.0

        # get liveliness
        liveliness = -1.0

        # get jerk
        jerk = -1.0 * c.jerk

        # get traj_follow
        traj_follow = c.last_to_goal

        # calc reward
        reward = COLLISION_WEIGHT * collision + INFRACTION_WEIGHT * infraction + \
                 LIVELINESS_WEIGHT * liveliness + JERK_WEIGHT * jerk + TRAJ_WEIGHT * traj_follow

        # update reward for car
        c.current_reward = reward

    # print("REWARD: Collisions-{:.3f} Infractions-{:.3f} liveliness-{:.3f} jerk-{:.3f} traj_follow-{:.3f}".format(collision, infraction, liveliness, jerk, traj_follow))
    # print("REWARD: Current reward-{:.3f}".format(reward))
    # Return final metric
    return reward
