def path_reward(state):
    total_reward = 0

    # Get cars
    controlled_cars = [state.objects[k] for k in state.controlled_cars.keys()]

    # Loop over every car
    for c in controlled_cars:
        # Calc collisions
        collision = -500 if state.is_in_collision(c) else 0

        # get liveliness
        liveliness = -1.0*c.total_time

        # get jerk
        jerk = -1.0*c.jerk

        # get traj_follow
        traj_follow = c.last_to_goal

        # calc reward
        reward = collision + liveliness + jerk + traj_follow

        # update reward for car
        c.current_reward = reward

        # add reward to total
        total_reward += reward

    # Return final metric
    return total_reward
