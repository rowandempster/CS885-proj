#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

from __future__ import division
import copy
import numpy as np
import random
import time
from collections import deque
from carla import ColorConverter as cc
import pygame

import gym
from gym import spaces
from gym.utils import seeding
import carla
import cv2

from .coordinates import train_coordinates
from .misc import _vec_decompose, delta_angle_between
from .carla_logger import *


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        self.host = params['host']
        self.logger = setup_carla_logger(
            "output_logger", experiment_name=str(params['port']))
        self.logger.error("Env running in port {}".format(params['port']))
        # parameters
        self.dt = params['dt']
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']
        self.max_time_episode = params['max_time_episode']
        self.obs_size = params['obs_size']
        self.state_size = (self.obs_size[0], self.obs_size[1] - 36)

        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.enable_target = params['enable_target']
        self.enable_ped = params['enable_ped']

        # action and observation space
        self.action_space = spaces.Box(
            np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.state_space = spaces.Box(
            low=-50.0, high=50.0, shape=(15, ), dtype=np.float32)

        # Connect to carla server and get world object
        # print('connecting to Carla server...')
        self._make_carla_client(self.host, self.port)

        # Load routes
        self.starts, self.dests = train_coordinates(self.task_mode)
        self.route_deterministic_id = 0

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')

        self.CAM_RES = 1024
        # Add camera sensor
        self.camera_img = np.zeros((self.CAM_RES, self.CAM_RES, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=30), carla.Rotation(pitch=-90))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.CAM_RES))
        self.camera_bp.set_attribute('image_size_y', str(self.CAM_RES))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # A list stores the ids for each episode
        self.actors = []

        # Future distances to get heading
        self.distances = [1., 5., 10.]
        self.target_vehicle = None
        self.ped = None
        
        self.ped_tasks = 5 + -2*np.arange(0, 6, 1)
        self.ped_task_i = None

    def set_ped_task_i(self, ped_task_i):
        self.ped_task_i = ped_task_i

    def get_ped_task_i(self):
        return self.ped_task_i

    def get_is_success(self):
        return self.isSuccess

    def get_ped_tasks(self):
        return self.ped_tasks

    def reset(self):
        self.collision_sensor = None
        self.lane_sensor = None

        # Delete sensors, vehicles and walkers
        while self.actors:
            (self.actors.pop()).destroy()

        self._load_world()

        # Spawn the ego vehicle at a random position between start and dest
        # Start and Destination
        if self.task_mode == 'Straight':
            self.route_id = 0
        elif self.task_mode == 'Curve':
            self.route_id = 1  #np.random.randint(2, 4)
        elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test' or self.task_mode == 'Left_turn':
            if self.code_mode == 'train':
                self.route_id = np.random.randint(0, 4)
            elif self.code_mode == 'test':
                self.route_id = self.route_deterministic_id
                self.route_deterministic_id = (
                    self.route_deterministic_id + 1) % 4
        elif self.task_mode == 'U_curve':
            self.route_id = 0
        self.start = self.left_turn_wpts[0].transform
        self.dest = self.left_turn_wpts[-1].transform

        if self.enable_target:
            self.target_vehicle = self._try_spawn_random_vehicle()
            self.actors.append(self.target_vehicle)
        if self.enable_ped:
            self.ped = self._try_spawn_random_ped()
            self.actors.append(self.ped)
        ego_spawned = False
        start = self.start
        init_fwd_speed = 0
        while not ego_spawned:
            # Code_mode == train, spwan randomly between start and destination
            # if self.code_mode == 'train':
                # start, init_fwd_speed = self._get_random_position_between()
                # init_fwd_speed = 5 * np.random.random()
                # print(f"init_fwd_speed: {init_fwd_speed}")
            try:
                self._try_spawn_ego_vehicle_at(start)
                ego_spawned = True
            except:
                ego_spawned = False
                # print("ego failed to spawn, re-trying")
        yaw = (start.rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
                    x=init_fwd_speed * np.cos(yaw),
                    y=init_fwd_speed * np.sin(yaw))

        # Add collision sensor
        self.collision_sensor = self.world.try_spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(
            lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)


        def get_camera_img(data):
            self.og_camera_img = data
        self.collision_hist = []
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.actors.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: get_camera_img(data))


        # Update timesteps
        self.time_step = 1
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        if self.code_mode == "train":
            self.settings.no_rendering_mode = True
        else:
            self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        # Set the initial speed to desired speed
        self.ego.set_velocity(init_speed)
        physics = self.ego.get_physics_control()
        physics.gear_switch_time *= 0.0
        physics.use_gear_autobox = False
        self.ego.apply_physics_control(physics)
        for _ in range(2):
            self.world.tick()

        # Get waypoint infomation
        ego_x, ego_y = self._get_ego_pos()
        self.current_wpt, progress = self._get_waypoint_xyz()

        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array([
            np.cos(wpt_yaw / 180 * np.pi),
            np.sin(wpt_yaw / 180 * np.pi)
        ])
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        ego_heading_vec = np.array(
            [np.cos(ego_heading),
                np.sin(ego_heading)])

        future_angles = self._get_future_wpt_angle(
            distances=self.distances)

        # Update State Info (Necessary?)
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])

        # decompose v and a to tangential and normal in ego coordinates
        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        # Reset action of last time step
        # TODO:[another kind of action]
        self.last_action = np.array([0.0, 0.0])

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]

        self.state_info['velocity_t'] = v_t
        self.state_info['acceleration_t'] = a_t

        # self.state_info['ego_heading'] = ego_heading
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt

        self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info['action_t_1'] = self.last_action
        self.state_info['angles_t'] = future_angles
        self.state_info['progress'] = progress
        self.state_info['target_vehicle_dist_y'] = 50
        self.state_info['target_vehicle_dist_x'] = 50
        self.state_info['target_vehicle_vel'] = 0
        if self.target_vehicle is not None:
            t_loc = self.target_vehicle.get_location()
            e_loc = self.ego.get_location()
            self.state_info['target_vehicle_dist_y'] = t_loc.y - e_loc.y
            self.state_info['target_vehicle_dist_x'] = e_loc.x - t_loc.x
            self.state_info['target_vehicle_vel'] = -1*self.target_vehicle.get_velocity().y
        self.state_info['ped_dist_y'] = 50
        self.state_info['ped_dist_x'] = 50
        self.state_info['ped_vel'] = 0
        if self.ped is not None:
            t_loc = self.ped.get_location()
            e_loc = self.ego.get_location()
            self.state_info['ped_dist_y'] = t_loc.y - e_loc.y
            self.state_info['ped_dist_x'] = e_loc.x - t_loc.x
            self.state_info['ped_vel'] = -1*self.ped.get_velocity().y

        # End State variable initialized
        self.isCollided = False
        self.isTimeOut = False
        self.isSuccess = False
        self.isOutOfLane = False
        self.isSpecialSpeed = False

        return self._get_obs(), copy.deepcopy(self.state_info)

    def step(self, action):

        # Assign acc/steer/brake to action signal
        # Ver. 1 input is the value of control signal
        # throttle_or_brake, steer = action[0], action[1]
        # if throttle_or_brake >= 0:
        #     throttle = throttle_or_brake
        #     brake = 0
        # else:
        #     throttle = 0
        #     brake = -throttle_or_brake

        # Ver. 2 input is the delta value of control signal
        # TODO:[another kind of action] change the action space to [-2, 2]
        current_action = np.array(action) + self.last_action
        current_action = np.clip(
            current_action, -1.0, 1.0, dtype=np.float32)
        throttle_or_brake, steer = current_action

        if throttle_or_brake >= 0:
            throttle = throttle_or_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_or_brake

        # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            gear=1,
            manual_gear_shift=True
            )
        self.ego.apply_control(act)

        for _ in range(1):
            self.world.tick()

        # Get waypoint infomation
        ego_x, ego_y = self._get_ego_pos()
        self.current_wpt, progress = self._get_waypoint_xyz()

        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array(
            [np.cos(wpt_yaw / 180 * np.pi),
                np.sin(wpt_yaw / 180 * np.pi)])
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        ego_heading_vec = np.array((np.cos(ego_heading),
                                    np.sin(ego_heading)))

        future_angles = self._get_future_wpt_angle(
            distances=self.distances)

        # Get dynamics info
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])

        # decompose v and a to tangential and normal in ego coordinates
        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]

        self.state_info['velocity_t'] = v_t
        self.state_info['acceleration_t'] = a_t

        # self.state_info['ego_heading'] = ego_heading
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt

        self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info['action_t_1'] = self.last_action
        self.state_info['angles_t'] = future_angles
        self.state_info['progress'] = progress
        self.state_info['target_vehicle_dist_y'] = 50
        self.state_info['target_vehicle_dist_x'] = 50
        self.state_info['target_vehicle_vel'] = 0
        if self.target_vehicle is not None:
            t_loc = self.target_vehicle.get_location()
            e_loc = self.ego.get_location()
            self.state_info['target_vehicle_dist_y'] = t_loc.y - e_loc.y
            self.state_info['target_vehicle_dist_x'] = e_loc.x - t_loc.x
            self.state_info['target_vehicle_vel'] = -1*self.target_vehicle.get_velocity().y
        self.state_info['ped_dist_y'] = 50
        self.state_info['ped_dist_x'] = 50
        self.state_info['ped_vel'] = 0
        if self.ped is not None:
            t_loc = self.ped.get_location()
            e_loc = self.ego.get_location()
            self.state_info['ped_dist_y'] = t_loc.y - e_loc.y
            self.state_info['ped_dist_x'] = e_loc.x - t_loc.x
            self.state_info['ped_vel'] = -1*self.ped.get_velocity().y

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = current_action

        # calculate reward
        isDone = self._terminal()
        current_reward = self._get_reward(np.array(current_action))

        return (self._get_obs(), current_reward, isDone,
                copy.deepcopy(self.state_info))


    def to_display_surface(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.
        Args:
        actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
        bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _try_spawn_random_vehicle(self):
        transform = carla.Transform()
        transform.location.x = 103.5
        transform.location.y = np.random.randint(0, 6) * -8 + 35
        transform.location.z = 0.2
        transform.rotation.yaw += -90
        vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            vehicle.set_simulate_physics(True)
            vehicle.set_velocity(carla.Vector3D(0, -self.desired_speed, 0))
            collision_sensor = self.world.try_spawn_actor(
                self.collision_bp, carla.Transform(), attach_to=vehicle)
            self.actors.append(collision_sensor)
            def get_collision_hist(event):
                impulse = event.normal_impulse
                intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                    impulse.z**2)
                self.collision_hist.append(intensity)
                if len(self.collision_hist) > self.collision_hist_l:
                    self.collision_hist.pop(0)
            collision_sensor.listen(
                lambda event: get_collision_hist(event))
            return vehicle
        raise Exception("Failed to spawn target vehicle")

    def _try_spawn_random_ped(self):
        blueprintsWalkers = self.world.get_blueprint_library().filter("vehicle.yamaha.yzf")
        walker_bp = random.choice(blueprintsWalkers)
        transform = carla.Transform()
        transform.location.x = 112
        transform.location.z = 0.5
        transform.location.y = 1.5
        transform.location.y = float(self.ped_tasks[self.ped_task_i])
        transform.rotation.yaw += -90
        ped = self.world.try_spawn_actor(walker_bp, transform)
        if ped is not None:
            ped.set_velocity(carla.Vector3D(0, -2, 0))
            ped.set_simulate_physics(True)
            collision_sensor = self.world.try_spawn_actor(
                self.collision_bp, carla.Transform(), attach_to=ped)
            self.actors.append(collision_sensor)
            def get_collision_hist(event):
                if self.ped.get_location().y < -6:
                    return
                impulse = event.normal_impulse
                intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                    impulse.z**2)
                self.collision_hist.append(intensity)
                if len(self.collision_hist) > self.collision_hist_l:
                    self.collision_hist.pop(0)
            collision_sensor.listen(
                lambda event: get_collision_hist(event))
            return ped
        raise Exception("Failed to spawn ped")

    def render(self, display, mode='human'):
        camera_surface = self.to_display_surface(self.og_camera_img)
        display.blit(camera_surface, (0, 0))

    def close(self):
        while self.actors:
            (self.actors.pop()).destroy()

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest.location.x)**2+(ego_y-dest.location.y)**2) < 2.0:
            self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.time_step, self.route_id))
            self.isSuccess = True
            return True

        # If collides
        if len(self.collision_hist) > 0:
            # print("Collision happened! Episode Done.")
            self.logger.debug(
                'Collision happened! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step >= self.max_time_episode:
            # print("Time out! Episode Done.")
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.time_step, self.route_id))
            self.isTimeOut = True
            return True

        # If out of lane
        # if len(self.lane_invasion_hist) > 0: 
        if abs(self.state_info['lateral_dist_t']) > 2.0:
            # print("lane invasion happened! Episode Done.")
            if self.state_info['lateral_dist_t'] > 0:
                self.logger.debug(
                    'Left Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            else:
                self.logger.debug(
                    'Right Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            self.isOutOfLane = True
            return True

        # If speed is special
        velocity = self.ego.get_velocity()
        v_norm = np.linalg.norm(np.array((velocity.x, velocity.y)))
        if v_norm < 0.0:
            self.logger.debug(
                'Speed too slow! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isSpecialSpeed = True
            return True
        elif v_norm > (5 * self.desired_speed):
            self.logger.debug(
                'Speed too fast! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isSpecialSpeed = True
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or actor.type_id == 'sensor.camera.rgb' or actor.type_id == 'sensor.other.collision':
                        actor.stop()
                    actor.destroy()

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_ego_pos(self):
        """Get the ego vehicle pose (x, y)."""
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        return ego_x, ego_y

    def _set_carla_transform(self, pose):
        """Get a carla tranform object given pose.

        Args:
            pose: [x, y, z, pitch, roll, yaw].

        Returns:
            transform: the carla transform object
        """
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            self.actors.append(vehicle)
            self.ego = vehicle
            return True
        return False

    def _get_obs(self):
        # [img version]
        # current_obs = self.camera_img[36:, :, :].copy()
        # return np.float32(current_obs / 255.0)

        # [vec version]
        return np.float32(self._info2normalized_state_vector())

    def _get_reward(self, action):
        """
        calculate the reward of current state
        params:
            action: np.array of shape(2,)
        """
        # Route completion reward
        c_completion = 100.0
        # Route non-completion penalty (for any reason: crash, too many steps)
        c_terminal = -50.0

        # Velocity reward constants
        c_v_eff_under_limit = 1.0
        c_v_eff_over_limit = -2.0
        # Penalty for needing another step
        r_step = -0.0
        # Penalty for non-smooth actions
        c_action_reg = -0.0
        # Penalty for yaw delta w.r.t. road heading
        c_yaw_delta = -0.0
        # Penalty for lateral deviation
        c_lat_dev = -0.1
        # Distance from goal penalty
        c_dist_from_goal = 3.5
        # Progress reward
        c_progress = 0.0

        # if self.isCollided or self.isOutOfLane or self.isTimeOut:
        if self.isCollided or self.isTimeOut:
            return c_terminal
        if self.isSuccess:
            return c_completion

        # reward for speed
        v = self.ego.get_velocity()
        speed_norm = np.linalg.norm(np.array([v.x, v.y]))
        if speed_norm > self.desired_speed:
            r_v_eff = c_v_eff_over_limit * abs(self.desired_speed - speed_norm)
        else:
            r_v_eff = c_v_eff_under_limit * speed_norm

        delta_yaw, _, _ = self._get_delta_yaw()
        r_delta_yaw = c_yaw_delta * delta_yaw

        r_action_regularized = c_action_reg * np.linalg.norm(action)**2

        lateral_dist = self.state_info['lateral_dist_t']
        r_lateral = c_lat_dev * abs(lateral_dist)

        r_dist_from_goal = c_dist_from_goal * (-1 + self.state_info['progress'])
        r_progress = c_progress * self.state_info['progress']

        r_tot = r_v_eff + r_step + r_delta_yaw + r_action_regularized + r_lateral + r_progress + r_dist_from_goal

        # print(f"r_v_eff: {r_v_eff}")
        # print(f"r_step: {r_step}")
        # print(f"r_delta_yaw: {r_delta_yaw}")
        # print(f"r_action_regularized: {r_action_regularized}")
        # print(f"r_lateral: {r_lateral}")
        # print(f"r_progress: {r_progress}")
        # print(f"r_tot: {r_tot}")
        return r_tot

    def _load_world(self):
        # Set map
        if self.task_mode == 'Straight':
            self.world = self.client.load_world('Town01')
        elif self.task_mode == 'Curve':
            # self.world = self.client.load_world('Town01')
            self.world = self.client.load_world('Town05')
        elif self.task_mode == 'Long':
            self.world = self.client.load_world('Town01')
            # self.world = self.client.load_world('Town02')
        elif self.task_mode == 'Lane':
            # self.world = self.client.load_world('Town01')
            self.world = self.client.load_world('Town05')
        elif self.task_mode == 'U_curve':
            self.world = self.client.load_world('Town03')
        elif self.task_mode == 'Lane_test':
            self.world = self.client.load_world('Town03')
        elif self.task_mode == 'Left_turn':
            self.world = self.client.load_world('Town05')
        self.map = self.world.get_map()
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        # if self.left_turn_wpts:
            # for wp in self.left_turn_wpts:
                # self.world.debug.draw_point(wp.transform.location, color=carla.Color(r=0,b=0,g=255))

    def _make_carla_client(self, host, port):
        print("connecting to Carla server...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(100.0)
        self.left_turn_wpts = None
        self._load_world()
        self.left_turn_wpts = []
        r1 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 48 and wpt.lane_id > 0]
        r1.reverse()
        self.left_turn_wpts += r1
        r2 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 744 and wpt.lane_id > 0]
        r2.reverse()
        self.left_turn_wpts += r2
        r3 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 30 and wpt.lane_id == 1]
        r3.reverse()
        self.left_turn_wpts += r3
        self.left_turn_wpts = self.left_turn_wpts[250:550]
        # self.left_turn_wpts = self.left_turn_wpts[280:550]

        # Set weather
        print(
            "Carla server port {} connected!".format(port))

    def _get_random_position_between(self):
        """
        get a random carla position on the line between start and dest
        """
        pos_frac = random.uniform(0, 0.75)
        mean_vel = (pos_frac / 0.75) * self.desired_speed
        return self.left_turn_wpts[int(pos_frac * len(self.left_turn_wpts))].transform, np.clip(random.gauss(mean_vel, self.desired_speed / 2), 0, self.desired_speed)

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt, _ = self._get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
        ego_yaw = self.ego.get_transform().rotation.yaw % 360

        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw, ego_yaw

    def _get_waypoint_xyz(self):
        """
        Get the (x,y) waypoint of current ego position
            if t != 0 and None, return the wpt of last moment
            if t == 0 and None wpt: return self.starts
        """
        waypoint, progress = self._get_waypoint(location=self.ego.get_location())
        waypoint_loc = self._get_waypoint(location=self.ego.get_location())[0].transform.location
        self.world.debug.draw_point(waypoint_loc, life_time=20)
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw)), progress
        else:
            return self.current_wpt, progress

    def _get_waypoint(self, location):
        min_wp = self.left_turn_wpts[0]
        index = 0
        for i, wp in enumerate(self.left_turn_wpts):
            if location.distance(wp.transform.location) < location.distance(min_wp.transform.location):
                min_wp = wp
                index = i
        return min_wp, float(index) / len(self.left_turn_wpts)

    def _get_future_wpt_angle(self, distances):
        """
        Get next wpts in distances
        params:
            distances: list of int/float, the dist of wpt which user wants to get
        return:
            future_angles: np.array, <current_wpt, wpt(dist_i)> correspond to the dist in distances
        """
        angles = []
        current_wpt, _ = self._get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            current_road_heading = self.current_wpt[3]
        else:
            current_road_heading = current_wpt.transform.rotation.yaw

        for d in distances:
            wpts = current_wpt.next(d)
            if len(wpts) > 1:
                wpt = [w for w in wpts if w.road_id == 744][0]
            else:
                wpt = wpts[0]
            # self.world.debug.draw_point(wpt.transform.location, life_time=1, color=carla.Color(r=0,b=255,g=0))
            wpt_heading = wpt.transform.rotation.yaw
            delta_heading = delta_angle_between(current_road_heading,
                                                wpt_heading)
            angles.append(delta_heading)

        return np.array(angles, dtype=np.float32)

    def _info2normalized_state_vector(self):
        '''
        params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
        type: np.array
        return: array of size[9,], torch.Tensor (v_x, v_y, a_x, a_y
                                                 delta_yaw, dyaw, d_lateral, action_last,
                                                 future_angles)
        '''
        velocity_t = self.state_info['velocity_t'] / (self.desired_speed * 1.5)
        accel_t = self.state_info['acceleration_t'] / 40
        delta_yaw_t = np.array(self.state_info['delta_yaw_t']).reshape(
            (1, )) / 180
        dyaw_dt_t = np.array(self.state_info['dyaw_dt_t']).reshape((1, )) / 30.0
        lateral_dist_t = self.state_info['lateral_dist_t'].reshape(
            (1, )) / 5      
        action_last = self.state_info['action_t_1'] / 3

        future_angles = self.state_info['angles_t'] / 90
        target_dist_y = np.array(self.state_info['target_vehicle_dist_y']).reshape((1, )) / 40
        target_dist_x = np.array(self.state_info['target_vehicle_dist_x']).reshape((1, )) / 30
        target_vel = np.array(self.state_info['target_vehicle_vel']).reshape((1, )) / 7
        ped_dist_y = np.array(self.state_info['ped_dist_y']).reshape((1, )) / 40
        ped_dist_x = np.array(self.state_info['ped_dist_x']).reshape((1, )) / 30
        ped_vel = np.array(self.state_info['ped_vel']).reshape((1, )) / 7

        
        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            action_last, future_angles, target_dist_y, target_dist_x, target_vel
        ],
                                  axis=0)
        info_vec = info_vec.squeeze()

        return info_vec