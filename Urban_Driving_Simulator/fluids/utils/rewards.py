def path_reward(state):
    # Get cars
    controlled_cars = [state.objects[k] for k in state.controlled_cars.keys()]

    # Calculate collisions
    collisions = sum([-500 if state.is_in_collision(car) else 0 for car in controlled_cars])

    # Calculate liveliness
    liveliness = -1.0*sum([c.total_time for c in controlled_cars])

    # Calculate Jerk
    total_jerk = -1.0*sum([c.jerk for c in controlled_cars])

    # Calculate trajectory following
    traj_follow = sum([c.last_to_goal for c in controlled_cars])

    # Return final metric
    return traj_follow + collisions + liveliness + total_jerk