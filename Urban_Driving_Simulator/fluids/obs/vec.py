from six import iteritems
import numpy as np
import pygame
from fluids.assets.shape import Shape
from fluids.obs.obs import FluidsObs
from fluids.utils import rotation_array
from scipy.misc import imresize
from fluids.consts import *


class VecObservation(FluidsObs):
    """
    Grid observation type. 
    Observation is an occupancy grid over the detection region. 
    Observation has 11 dimensions: terrain, drivable regions, illegal drivable 
    regions, cars, pedestrians, traffic lights x 3, way points, point trajectory and edge trajectory.
    Array representation is (grid_size, grid_size, 11)
    """

    def __init__(self, car, obs_dim=500, shape=(500, 500)):
        from fluids.assets import ALL_OBJS, TrafficLight, Lane, Terrain, Sidewalk, \
            PedCrossing, Street, Car, Waypoint, Pedestrian
        state = car.state
        self.car = car
        self.shape = shape
        self.grid_dim = obs_dim
        self.downsample = self.shape != (obs_dim, obs_dim)
        self.grid_square = Shape(x=car.x + obs_dim / 3 * np.cos(car.angle),
                                 y=car.y - obs_dim / 3 * np.sin(car.angle),
                                 xdim=obs_dim, ydim=obs_dim, angle=car.angle,
                                 color=None, border_color=(200, 0, 0))
        gd = self.grid_dim
        a0 = self.car.angle + np.pi / 2
        a1 = self.car.angle
        # rel = (self.car.x + gd / 2 * np.cos(a0) - gd / 6 * np.cos(a1),
        #        self.car.y - gd / 2 * np.sin(a0) + gd / 6 * np.sin(a1),
        #        self.car.angle)
        rel = (self.car.x,
               self.car.y,
               self.car.angle)


        scale_factor = 500.0
        # Gather the lights and cars objects for later information gathering
        traffic_lights = []
        other_cars = []
        for k, obj in iteritems(state.objects):
            if (car.can_collide(obj) or type(obj) in {TrafficLight, Lane, Street}) and self.grid_square.intersects(obj):
                typ = type(obj)
                if typ == TrafficLight:
                    traffic_lights.append(obj)
                elif typ == Car:
                    other_cars.append(obj)


        # Information about the stop light
        light_x, light_y = 0.0, 0.0
        color_to_state = {RED:-1.0, YELLOW: 0.0, GREEN: 1.0}
        light_state = 1.0 # Default state to green so that if it doesn't exist the default behaviour is to go.
        # 0 if no light 1 if light exists
        light_exists = 0.0

        closest_dist = 99999999
        for light in traffic_lights:
            rel_obj = light.get_relative(rel)
            x, y = rel_obj.x, rel_obj.y
            dist_sq = x**2 + y**2
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                # TODO: Scale so the values aren't as large
                light_x, light_y = rel_obj.x, rel_obj.y
                light_exists = 1.0
                light_state = color_to_state[light.color]

        light_x /= scale_factor
        light_y /= scale_factor

        # Information about the closest car
        other_car_x, other_car_y = 0.0, 0.0
        other_car_vx, other_car_vy = 0.0, 0.0
        other_car_exists = 0.0

        closest_dist = 99999999
        for other_car in other_cars:
            rel_obj = other_car.get_relative(rel)
            x, y = rel_obj.x, rel_obj.y
            dist_sq = x**2 + y**2
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                # TODO: Scale so the values aren't as large
                other_car_x, other_car_y = rel_obj.x, rel_obj.y
                other_car_exists = 1.0

                relative_theta = other_car.angle - car.angle
                other_car_vx = other_car.vel * np.cos(relative_theta)
                other_car_vy = other_car.vel * -np.sin(relative_theta)
                other_car_exists

        other_car_x /= scale_factor
        other_car_y /= scale_factor

        # Information about the next waypoint
        # TODO: Scale so the values aren't as large
        if car.waypoints:
            rel_waypoint = car.waypoints[0].get_relative(rel)
            waypoint_x = rel_waypoint.x
            waypoint_y = rel_waypoint.y
        else:
            waypoint_x, waypoint_y = 50.0, 0.0

        waypoint_x /= scale_factor
        waypoint_y /= scale_factor

        self.pygame_rep = [light_x, light_y, light_state, light_exists, other_car_x, other_car_y, other_car_vx, other_car_vy, other_car_exists, waypoint_x, waypoint_y, car.vel]
        # self.pygame_rep = [pygame.transform.rotate(window, 90) for window in [terrain_window,
        #                                                                       drivable_window,
        #                                                                       undrivable_window,
        #                                                                       car_window,
        #                                                                       ped_window,
        #                                                                       light_window_red,
        #                                                                       light_window_green,
        #                                                                       light_window_yellow,
        #                                                                       direction_window,
        #                                                                       direction_pixel_window,
        #                                                                       direction_edge_window
        #                                                                       ]]

    def render(self, surface):
        self.grid_square.render(surface, border=10)
        if self.car.vis_level > 3:

            if self.car.vis_level > 4:
                for obj in self.all_collideables:
                    obj.render_debug(surface)
            for y in range(4):
                for x in range(2):
                    i = y + x * 4
                    if i < len(self.pygame_rep):
                        surface.blit(self.pygame_rep[i],
                                     (surface.get_size()[0] - self.grid_dim * (x + 1), self.grid_dim * y))
                        pygame.draw.rect(surface, (200, 0, 0),
                                         pygame.Rect((surface.get_size()[0] - self.grid_dim * (x + 1) - 5,
                                                      0 - 5 + self.grid_dim * y),
                                                     (self.grid_dim + 10, self.grid_dim + 10)), 10)

    def get_array(self):
       return self.pygame_rep

    def sp_imresize(self, arr, shape):
        return np.array([imresize(arr[:, :, i], shape) for i in range(arr.shape[2])]).T
