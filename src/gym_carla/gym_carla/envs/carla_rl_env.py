#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

from __future__ import division
import copy
from logging import disable
from os import strerror
import numpy as np
import random
import time
import math
from collections import deque
import pygame

import gym
from gym import spaces
from gym.utils import seeding
from pygame import display
import carla
import cv2
import roslibpy


from .coordinates import train_coordinates
from .misc import _vec_decompose, delta_angle_between
from .carla_logger import *

## importing the carla eye view rendering
from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.render import RoutePlanner
from gym_carla.envs.misc import *


from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

# ==============================================================================
# -- Socket Connection -------------------------------------------------------------
# ==============================================================================


## socket details
client = roslibpy.Ros(host='localhost', port=9090)
client.run()
listener1 = roslibpy.Topic(client, '/carla_output', 'icra22pkg/carlaInterface')
listener2 = roslibpy.Topic(client, '/carla_control', 'icra22pkg/carla_control')
print("Connected to ROS... Data")
listener1.advertise()

# ==============================================================================
# -- Ros control -------------------------------------------------------------
# ==============================================================================


class ros_control(object):
    def __init__(self):
        self._world = None
        print("Ego initalized")
        self._pathIndex = 1
        self._control = carla.VehicleControl()
        self._control.steer = 0
        self._control.reverse = 0
        self._control.throttle = 0
        self._control.brake = 1
        self._control.hand_brake = 0
        self._st_max_lim = -math.pi/3 #1.22173 #math.pi/3
        self._zEgo = 0.1

    def applyControl(self):
        self._world.player.apply_control(self._control)

    ### Message callback functions for sending carla_output data and receiving carla_control data
    #listener 2 callback function
    def receive_message(self,msg):
        self._control.steer = float(msg['steer'])
        self._control.throttle = float(msg['throttle'])
        self._control.brake = float(msg['brake'])
        #self._control.reverse = float(msg['reverse'])
        #self._control.handbrake = float(msg['hand_brake'])

    #listener 1 callback function
    def send_message(self,xRefArr,yRefArr,yawRefArr,vRefArr):
        print("Sending message")
        dict = {'xEgo':self._xEgo, 'yEgo':self._yEgo, 
                'yawEgo':self._yawEgo, 
                'vEgo':self._vEgo, 'stEgo':self._stEgo,
                'xRef':xRefArr[self._pathIndex],
                'yRef':yRefArr[self._pathIndex],
                'yawRef':yawRefArr[self._pathIndex],
                'vRef':vRefArr}
        listener1.publish(roslibpy.Message(dict))        
        print("Message sent")


    def getFeedback(self,transform,velocity,control,xRefArr,yRefArr,yawRefArr,vRefArr):
        t = transform #self.ego.get_transform()
        v = velocity #self.ego.get_velocity()
        c = control #self.ego.get_control()
        self._zEgo = t.location.z
        self._xEgo = t.location.x
        self._yEgo = -t.location.y
        self._yawEgo = -t.rotation.yaw * math.pi/180
        self._vEgo = math.sqrt(v.x**2 + v.y**2)
        self._stEgo = c.steer * self._st_max_lim  
        print("No issues until")
        self.send_message(xRefArr,yRefArr,yawRefArr,vRefArr) #roslibpy publisher
        print("No issues.. message sent")

    # def vehicleControll_Callback(self, Msg):
    #     listener2.subscribe(self.receive_message) # receive the carla_control data from m
    #     self._control.steer = Msg.steer / self._st_max_lim
    #     self._control.reverse = 0 #Msg.reverse
    #     self._control.throttle = Msg.throttle
    #     self._control.brake = Msg.brake
    #     self._control.hand_brake = 0 #Msg.hand_brake
    #     # self.applyControl()

    def localizeAlongPath(self,xRefArr,yRefArr,xEgo,yEgo):
        pathLength = len(xRefArr)
        distance2Ego = []
        for n in range(1, pathLength - 5):
            diffX = xRefArr[n] - xEgo
            diffY = yRefArr[n] - yEgo
            distance2Ego.append(diffX * diffX + diffY * diffY)
        self._pathIndex = distance2Ego.index(min(distance2Ego)) + 5
        # print(distance2Ego[self._pathIndex])
        # print(self._pathIndex)

    def read_csv(self): 
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'routes/left_turn_route.csv')
        self._xRefArr = []
        self._yRefArr = []
        self._yawRefArr = []
        self._vRefArr = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    # print("x: " + str(row[0]) + ", y: " + str(row[1]) + ", y: " + str(row[2]) + ", v: " + str(row[3]))
                    line_count += 1
                    self._xRefArr.append(float(row[0]))
                    self._yRefArr.append(float(row[1]))
                    self._yawRefArr.append(float(row[2]))
                    self._vRefArr.append(float(row[3]))
        global xStart
        global yStart
        global zStart
        global yawStart
        xStart = self._xRefArr[0]
        yStart = self._yRefArr[0]
        zStart = 0.2
        yawStart = self._yawRefArr[0]
        print("Processed: " +  str(line_count))

# ==============================================================================
# -- GYM Environment ------------------------------------------------------------
# ==============================================================================

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, params):
        self.logger = setup_carla_logger(
            "output_logger", experiment_name=str(params['port']))
        self.logger.info("Env running in port {}".format(params['port']))
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
        self.display_size = 256 # screen size of the bird-eye renderer
        self.obs_range = 32 #observation range (in meters)
        self.d_behind = 12 #distance behind the ego vehicle (in meters)
        self.pixor = False #use pixor or not
        self.pixor_size = 64 #pixor model
        self.discrete = False #discrete or continuous action space
        self.obs_size = int(self.obs_range/0.125) #observation size o.125 refers to the bin size of the lidar sensor



        ##### Reference values to be passed to the roslibpy publisher
        self._xRefArr = []
        self._yRefArr = []
        self._yawRefArr = []
        self._vRefArr = self.desired_speed

        # action and observation space
        self.action_space = spaces.Box(
            np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.state_space = spaces.Box(
            low=-50.0, high=50.0, shape=(12, ), dtype=np.float32)

        # action and observation spaces
        observation_space_dict = {
        'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }
        if self.pixor:
            observation_space_dict.update({
            'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
            'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
            'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
            })
        self.observation_space = spaces.Dict(observation_space_dict)




        # Connect to carla server and get world object
        # print('connecting to Carla server...')
        self._make_carla_client('87e4be046369', self.port)

        
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

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
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

        #initialize the renderer
        if self.pixor:
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

    
    def reset(self):

        while True:
            try:
                self.collision_sensor = None
                self.lane_sensor = None
                # Delete sensors, vehicles and walkers
                while self.actors:
                    (self.actors.pop()).destroy()
                
                self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])


                # Disable sync mode
                self._set_synchronous_mode(False)

                # Spawn the ego vehicle at a random position between start and dest
                # Start and Destination
                if self.task_mode == 'Straight':
                    self.route_id = 0
                elif self.task_mode == 'Curve':
                    self.route_id = 1  #np.random.randint(2, 4)
                elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test' or self.task_mode == 'Left_turn':
                    if self.code_mode == 'train':
                        self.route_id = (self.route_deterministic_id + 1) % 4
                    elif self.code_mode == 'test':
                        self.route_id = self.route_deterministic_id
                        self.route_deterministic_id = (
                            self.route_deterministic_id + 1) % 4
                elif self.task_mode == 'U_curve':
                    self.route_id = 0

                print("route id:", self.route_id)
                # Load routes
                self.start = self.starts[self.route_id]
                self.dest = self.dests[self.route_id]

                # The tuple (x,y) for the current waypoint
                self.current_wpt = np.array((self.start[0], self.start[1],
                                             self.start[5]))
                
                # Appending the values of x,y,yaw of starts to the RefArr
                self._xRefArr.append(float(self.current_wpt[0]))
                self._yRefArr.append(float(self.current_wpt[1]))
                self._yawRefArr.append(float(self.current_wpt[2]))
                print("Current waypoint:",self.current_wpt)
               
                global xStart
                global yStart
                global zStart
                global yawStart
                xStart = self._xRefArr[0]
                yStart = self._yRefArr[0]
                zStart = 0.2
                yawStart = self._yawRefArr[0]    

                ego_spawn_times = 0
                while True:
                    if ego_spawn_times > self.max_ego_spawn_times:
                        self.reset()
                    transform = self._set_carla_transform(self.start)
                    # Code_mode == train, spwan randomly between start and destination
                    if self.code_mode == 'train':
                        transform = self._get_random_position_between(
                            start=self.start,
                            dest=self.dest,
                            transform=transform)
                    if self._try_spawn_ego_vehicle_at(transform):
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)


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

                self.collision_hist = []


                # Add lidar sensor
                self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
                self.lidar_sensor.listen(lambda data: get_lidar_data(data))
                def get_lidar_data(data):
                    self.lidar_data = data

                # Add camera sensor
                self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
                self.camera_sensor.listen(lambda data: get_camera_img(data))
                def get_camera_img(data):
                    array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
                    array = np.reshape(array, (data.height, data.width, 4))
                    array = array[:, :, :3]
                    array = array[:, :, ::-1]
                    self.camera_img = array


                # Update timesteps
                self.time_step = 1
                self.reset_step += 1

                # Enable sync mode
                self.settings.synchronous_mode = True
                self.world.apply_settings(self.settings)

                self.routeplanner = RoutePlanner(self.ego,self.max_wpt)
                self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()


                # Set the initial speed to desired speed
                yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
                init_speed = carla.Vector3D(
                    x=self.desired_speed * np.cos(yaw),
                    y=self.desired_speed * np.sin(yaw))
                self.ego.set_velocity(init_speed)
                self.world.tick()
                self.world.tick()

                # Get waypoint infomation
                ego_x, ego_y = self._get_ego_pos()
                self.current_wpt = self._get_waypoint_xyz()

                delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
                road_heading = np.array([
                    np.cos(wpt_yaw / 180 * np.pi),
                    np.sin(wpt_yaw / 180 * np.pi)
                ])
                ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
                ego_heading_vec = np.array(
                    [np.cos(ego_heading),
                     np.sin(ego_heading)])

                self.birdeye_render.set_hero(self.ego,self.ego.id)

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

                # End State variable initialized
                self.isCollided = False
                self.isTimeOut = False
                self.isSuccess = False
                self.isOutOfLane = False
                self.isSpecialSpeed = False

                return self._get_obs(), copy.deepcopy(self.state_info)

            except:
                self.logger.error("Env reset() error")
                time.sleep(2)
                self._make_carla_client('87e4be046369', self.port)


    def step(self,action):
        try:
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

            RC = ros_control()
            current_action = np.array(action) + self.last_action
            current_action = np.clip(
                current_action, -1.0, 1.0, dtype=np.float32)
            throttle_or_brake, steer = current_action

        # # Apply control signal
            if throttle_or_brake >= 0:
                throttle = throttle_or_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_or_brake

            listener2.subscribe(RC.receive_message)
            th = RC._control.throttle
            st = RC._control.steer
            br = RC._control.brake
            print("throttle",throttle)
            print("steer",steer)
            print("brake",brake)


            act = carla.VehicleControl(
                throttle=float(th),
                steer=float(-st),
                brake=float(br))
            self.ego.apply_control(act)

            print("Current_action",current_action,":",th,st)
          
            for _ in range(4):
                self.world.tick()

            # Get waypoint infomation
            ego_x, ego_y = self._get_ego_pos()
            self.current_wpt = self._get_waypoint_xyz()

            delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
            road_heading = np.array(
                [np.cos(wpt_yaw / 180 * np.pi),
                 np.sin(wpt_yaw / 180 * np.pi)])
            ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
            ego_heading_vec = np.array((np.cos(ego_heading),
                                        np.sin(ego_heading)))

            future_angles = self._get_future_wpt_angle(
                distances=self.distances)

            print("No issues until here - step 1")
            

            # Get dynamics info
            transform = self.ego.get_transform()
            velocity = self.ego.get_velocity()
            accel = self.ego.get_acceleration()
            dyaw_dt = self.ego.get_angular_velocity().z
            v_t_absolute = np.array([velocity.x, velocity.y])
            a_t_absolute = np.array([accel.x, accel.y]) 
            ct = self.ego.get_control()          

            RC.getFeedback(transform,velocity,ct,self._xRefArr,self._yRefArr,self._yawRefArr,self._vRefArr)
            print("No issues until here - step 2 and now onto localizing")            
            # Localize ego vehicle
            RC.localizeAlongPath(self._xRefArr,self._yRefArr,ego_x,ego_y)

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

            # Update timesteps
            self.time_step += 1
            self.total_step += 1
            self.last_action = current_action
            print("Timestep count:",self.time_step)

            # calculate reward
            isDone = self._terminal()
            current_reward = self._get_reward(np.array(current_action))

            return (self._get_obs(), current_reward, isDone,
                    copy.deepcopy(self.state_info))

        except:
            self.logger.error("Env step() error")
            time.sleep(2)
            return (self._get_obs(), 0.0, True, copy.deepcopy(self.state_info))


    def render(self, mode='human'):
        pass

    def close(self):
        while self.actors:
            (self.actors.pop()).destroy()

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 2.0:
            print("Get destination! Episode Done.")
            self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.time_step, self.route_id))
            self.isSuccess = True
            return True

        # If collides
        if len(self.collision_hist) > 0:
            print("Collision happened! Episode Done.")
            self.logger.debug(
                'Collision happened! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step >= self.max_time_episode:
            print("Time out! Episode Done.")
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.time_step, self.route_id))
            self.isTimeOut = True
            # return True

        # If out of lane
        # if len(self.lane_invasion_hist) > 0:
        if abs(self.state_info['lateral_dist_t']) > 1.5:
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
        if v_norm < 4:
            self.logger.debug(
                'Speed too slow! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isSpecialSpeed = True
            return True
        elif v_norm > (1.5 * self.desired_speed):
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

    def _init_renderer(self):
        """Initialize the birdeye view renderer """
        pygame.init()
        self.display = pygame.display.set_mode(
        (self.display_size * 3, self.display_size),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
        'screen_size': [self.display_size, self.display_size],
        'pixels_per_meter': pixels_per_meter,
        'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

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
        """Get observation.
        """
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
        if self.display_route:
            roadmap_render_types.append('waypoints')
        self.birdeye_render.render(self.display, roadmap_render_types)
        roadmap = pygame.surfarray.array3d(self.display)
        roadmap = roadmap[0:self.display_size, :, :]
        roadmap = display_to_rgb(roadmap, self.obs_size)
        # Add ego vehicle
        for i in range(self.obs_size):
            for j in range(self.obs_size):
                if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
                    roadmap[i, j, :] = birdeye[i, j, :]

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))

        ## Lidar image generation
        point_cloud = []
        # Get point cloud data
        for location in self.lidar_data:
            point_cloud.append([location.x, location.y, -location.z])
        point_cloud = np.array(point_cloud)
        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
        x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
        z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
        lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
        # Add the waypoints to lidar image
        if self.display_route:
            wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
        else:
            wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        # Display lidar image
        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        self.display.blit(lidar_surface, (self.display_size, 0))

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 2, 0))

        # Display on pygame
        pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, 
        np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

        if self.pixor:
            ## Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

        # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
        # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
        for actor in self.world.get_actors().filter('vehicle.*'):
            x, y, yaw, l, w = get_info(actor)
            x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
            if actor.id != self.ego.id:
                if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
                    x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                    local_info=(x_local, y_local, yaw_local, l, w),
                    d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
                    cos_t = np.cos(yaw_pixel)
                    sin_t = np.sin(yaw_pixel)
                    logw = np.log(w_pixel)
                    logl = np.log(l_pixel)
                    pixels = get_pixels_inside_vehicle(
                    pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
                    pixel_grid=self.pixel_grid)
                    for pixel in pixels:
                        vh_clas[pixel[0], pixel[1]] = 1
                        dx = x_pixel - pixel[0]
                        dy = y_pixel - pixel[1]
                        vh_regr[pixel[0], pixel[1], :] = np.array(
                            [cos_t, sin_t, dx, dy, logw, logl])

        # Flip the image matrix so that the origin is at the left-bottom
        vh_clas = np.flip(vh_clas, axis=0)
        vh_regr = np.flip(vh_regr, axis=0)

        # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
        pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

        obs = {
        'camera':camera.astype(np.uint8),
        'lidar':lidar.astype(np.uint8),
        'birdeye':birdeye.astype(np.uint8),
        'state': state,
        }

        if self.pixor:
            obs.update({
                'roadmap':roadmap.astype(np.uint8),
                'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr':vh_regr.astype(np.float32),
                'pixor_state': pixor_state,
            })        
        return np.float32(self._info2normalized_state_vector())

    def _get_reward(self, action):
        """
        calculate the reward of current state
        params:
            action: np.array of shape(2,)
        """
        # end state
        # reward for done: collision/out/SpecislSPeed & Success
        r_step = 10.0
        if self.isCollided or self.isOutOfLane or self.isSpecialSpeed:
            r_done = -50.0
            return r_done
        if self.isSuccess:
            r_done = 30.0
            return r_done

        # reward for speed
        v = self.ego.get_velocity()
        ego_velocity = np.array([v.x, v.y])
        speed_norm = np.linalg.norm(ego_velocity)
        delta_speed = (speed_norm - self.desired_speed) + 0.5 * speed_norm
        r_speed = -delta_speed**2 / 5.0
        # print("r_speed:", speed_norm)

        # reward for steer
        delta_yaw, _, _ = self._get_delta_yaw()
        r_steer = -10.0 * (delta_yaw * np.pi / 180)**2
        # print("r_steer:", delta_yaw, '------>', r_steer)

        # reward for action smoothness
        r_action_regularized = -5 * np.linalg.norm(action)**2
        # print("r_action:", action, '------>', r_action_regularized)

        # reward for lateral distance to the center of road
        lateral_dist = self.state_info['lateral_dist_t']
        r_lateral = -10.0 * lateral_dist**2
        # print("r_lateral:", lateral_dist, '-------->', r_lateral)

        return r_speed + r_steer + r_action_regularized + r_lateral + r_step

    def _make_carla_client(self, host, port):
        """Make a carla client """
        clock = pygame.time.Clock()
        while True:
            try:
                self.logger.info("connecting to Carla server...")
                self.client = carla.Client(host, port)
                self.client.set_timeout(10.0)                

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
                elif self.task_mode == 'Left_turn':
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'Lane_test':
                    self.world = self.client.load_world('Town03')
                self.map = self.world.get_map()

                # Set weather
                self.world.set_weather(carla.WeatherParameters.ClearNoon)

                self.logger.info(
                    "Carla server port {} connected!".format(port))
                break
            except Exception:
                self.logger.error(
                    'Fail to connect to carla-server...sleeping for 2')
                time.sleep(2)

    def _get_random_position_between(self, start, dest, transform):
        """
        get a random carla position on the line between start and dest
        """
        if self.task_mode == 'Straight':
            # s_x, s_y, s_z = start[0], start[1], start[2]
            # d_x, d_y, d_z = dest[0], dest[1], dest[2]

            # ratio = np.random.rand()
            # new_x = (d_x - s_x) * ratio + s_x
            # new_y = (d_y - s_y) * ratio + s_y
            # new_z = (d_z - s_z) * ratio + s_z

            # transform.location = carla.Location(x=new_x, y=new_y, z=new_z)
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 30)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Curve':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 45)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Long' or self.task_mode == 'Lane':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            ratio = float(np.random.rand() * 60)

            transform = self.map.get_waypoint(
                location=start_location).next(ratio)[0].transform
            transform.location.z = start[2]

        elif self.task_mode == 'Left_turn':
            start_location = carla.Location(x=start[0], y=start[1], z=0.22)
            #ratio = float(np.random.rand() * 60)
            transform = self.map.get_waypoint(
                location=start_location).transform
            transform.location.z = start[2]

        return transform

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
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
        waypoint = self.map.get_waypoint(location=self.ego.get_location())
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw))
        else:
            return self.current_wpt

    def _get_future_wpt_angle(self, distances):
        """
        Get next wpts in distances
        params:
            distances: list of int/float, the dist of wpt which user wants to get
        return:
            future_angles: np.array, <current_wpt, wpt(dist_i)> correspond to the dist in distances
        """
        angles = []
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            current_road_heading = self.current_wpt[3]
        else:
            current_road_heading = current_wpt.transform.rotation.yaw

        for d in distances:
            wpt_heading = current_wpt.next(d)[0].transform.rotation.yaw
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
        velocity_t = self.state_info['velocity_t']
        accel_t = self.state_info['acceleration_t']

        delta_yaw_t = np.array(self.state_info['delta_yaw_t']).reshape(
            (1, )) / 2.0
        dyaw_dt_t = np.array(self.state_info['dyaw_dt_t']).reshape((1, )) / 5.0

        lateral_dist_t = self.state_info['lateral_dist_t'].reshape(
            (1, )) * 10.0
        action_last = self.state_info['action_t_1'] * 10.0

        future_angles = self.state_info['angles_t'] / 2.0

        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            action_last, future_angles
        ],
                                  axis=0)
        info_vec = info_vec.squeeze()

        return info_vec


