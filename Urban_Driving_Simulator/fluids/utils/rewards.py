COLLISION_WEIGHT = 500.0
LIVELINESS_WEIGHT = 1.0
JERK_WEIGHT = 1.0
TRAJ_WEIGHT = 1.0


def path_reward(state):
    total_reward = 0

    # Get cars
    controlled_cars = [state.objects[k] for k in state.controlled_cars.keys()]

    # Loop over every car
    for c in controlled_cars:
        # Calc collisions

        if state.is_in_collision(c):
            collision = -1.0
            c.collisions += 1
        else:
            collision = 0.0

        # get liveliness
        liveliness = -1.0 * c.total_time

        # get jerk
        jerk = -1.0 * c.jerk

        # get traj_follow
        traj_follow = c.last_to_goal

        # calc reward
        reward = COLLISION_WEIGHT * collision + LIVELINESS_WEIGHT * liveliness + \
                 JERK_WEIGHT * jerk + TRAJ_WEIGHT * traj_follow

        # update reward for car
        c.current_reward = reward

        # add reward to total
        total_reward += reward

    # Return final metric
    return total_reward
