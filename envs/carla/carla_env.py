# import pygame
# import pygame.freetype
import weakref
import logging
import time, random
import collections
import numpy as np
import math
import cv2
import re
import sys
import os
from collections import deque, defaultdict
from tensorflow.keras.models import load_model
import tensorflow as tf
# import matplotlib.pyplot as plt
# append sys PATH for CARLA simulator 
sys.path.append('/home/ana/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
sys.path.append('home/ana/CARLA_0.9.13/PythonAPI/carla/')

import carla
from carla import ColorConverter as cc
from agents.navigation.basic_agent import BasicAgent
from carla import VehicleLightState as vls

import tensorflow as tf
layers = tf.keras.layers


# HEIGHT = 240
# WIDTH = 320
HEIGHT = 120
WIDTH = 160

class InterSection(object):
    SHOW_CAM = True
    im_width = WIDTH
    im_height = HEIGHT
    image_seg = None
    CAMERA_POS_Z = 50
    CAMERA_POS_X = 0

    def __init__(
        self,
        vehicle_type = 'single', 
        frame=10, 
        port=2200, 
        seed=0):  # port=2000

        self.vehicle_type = vehicle_type
        self.frame = frame 
        self.ego_vehicle = None
        self.obs_list = []   # surrounding vehicles
        self.collision_sensor = None
        self.port = port
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town10HD_Opt') 
        self.world.unload_map_layer(carla.MapLayer.Buildings) 
        self.map = self.world.get_map()      
        self._weather_index = 8
        settings = self.world.get_settings()
        # settings.no_rendering_mode = True
        settings.no_rendering_mode = not self.SHOW_CAM
        self.world.apply_settings(settings)        
        self.seed = seed
        # self.reset()   
        self.cnn_model = load_model('/home/ana/Documents/Architecture_Transformers/CNN_image_model.h5', compile=False)  
        # if self.SHOW_CAM:
        #     self.spectator = self.world.get_spectator() 
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')

        self.image_for_CNN = None

##############################################################################################
    def _safe_try_spawn(
            self, 
            blueprint, 
            spawn_points, 
            max_tries=30):
        
        '''
        Try multiple spawn points using 
        `try_spawn_actor`, returning as 
        soon as you succeed.    
        '''

        random.shuffle(spawn_points)
        for sp in spawn_points[:max_tries]:
            actor = self.world.try_spawn_actor(blueprint, sp)
            if actor is not None:
                return actor
            

    def _spawn_with_small_offsets(
            self, 
            blueprint, 
            base_transform, 
            trials=20, 
            span=2.0):
        
        '''
        It sweeps small displacements 
        at the same base point to avoid 
        collisions at spawn.
        '''

        offsets = np.linspace(-span, span, trials)
        for dx in offsets:
            sp = carla.Transform(
                        carla.Location(
                        x=base_transform.location.x + dx,
                        y=base_transform.location.y,
                        z=base_transform.location.z),
                        base_transform.rotation)
                        
            actor = self.world.try_spawn_actor(blueprint, sp)
            if actor is not None:
                return actor
            

    def apply_cnn(self, im):
        """
        Applies inference to the image. 
        Generates an embedding.
        """

        img = np.float32(im) / 255.0
        img = cv2.resize(img, (160, 120))      # -> (120,160,3)
        img = tf.convert_to_tensor(np.expand_dims(img, axis=0))  # -> (1,120,160,3)

        cnn_applied = self.cnn_model(img, training=False)  # -> (1,280)
        cnn_applied = np.squeeze(cnn_applied)  # -> (280,)
        # print("CNN embedding shape:", cnn_applied.shape)

        return cnn_applied
    
    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # -> (240,320,3)
        self.image_seg = i
    

    def reset(self):

        ''''
        It starts a new episode and generates 
        the first state.
        '''

        self.actor_list = []
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # settings.synchronous_mode = False
        settings.fixed_delta_seconds = 1 / self.frame 
        self.world.apply_settings(settings)
        self.ego_location_history = deque(maxlen=10)
        self.dist_travelled = 0
        self.ii = None
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor        
        self.traffic_manager = self.client.get_trafficmanager(self.port+50)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_random_device_seed(self.seed)
        self.seed = (self.seed + 1) % 500
        self.traffic_manager.set_synchronous_mode(True)
        # self.traffic_manager.set_synchronous_mode(False)       
        synchronous_master = False        
        list_actor = self.world.get_actors()

        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Green) 
                actor_.set_green_time(2000.0)
                
        ## spawn the ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter('vehicle.mercedes.coupe_2020')[0]
        bp_ego.set_attribute('color', '0, 0, 0')
        bp_ego.set_attribute('role_name', 'hero')

        spawn_point_ego = self.world.get_map().get_spawn_points()[0]
        spawn_point_ego.location.x = 0
        spawn_point_ego.location.y = -64.5
        spawn_point_ego.location.z  = 0.1
        spawn_point_ego.rotation.yaw = 180

        if self.ego_vehicle is not None:
            self.destroy()
            
        ego = self._spawn_with_small_offsets(bp_ego, spawn_point_ego)  
        if ego is None:
            sp_list = self.world.get_map().get_spawn_points()
            ego = self._safe_try_spawn(bp_ego, sp_list)

        if ego is None:
            ego = self.world.try_spawn_actor(bp_ego, spawn_point_ego)

        self.ego_vehicle = ego  

        self.ego_current_speed_ratio = -100
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, self.ego_current_speed_ratio)
        
        self.target_speed = 48
        
        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        self.world.tick()
        
        self.agent = BasicAgent(self.ego_vehicle, target_speed = self.target_speed)

        self.speed_limit_flag = 0
        l = self.ego_vehicle.bounding_box.extent.x*0.9
        w = self.ego_vehicle.bounding_box.extent.y
        self.fix_theta = np.arctan(w/l) * 180 / np.pi
        self.fix_length = np.sqrt(l**2+w**2)
        self.displacement_waypoint = self.map.get_waypoint(spawn_point_ego.location)
        self.waypoint_ego = self.map.get_waypoint(spawn_point_ego.location)

        self.count = 0
        self.subcount1 = 0
        self.subcount2 = 0
        self.interval = 2
        self.command_interval = round(self.frame / self.interval) # execute 2 command per second
        self.list_action = []

        ## Spawn surronding vehicles
        self.obs_list = []
        self.obs_velo_list = []
        self.obs_agent_list = []

        # randomly choose spawned points
        lat_list = [-71, -70,
                    -97, -102, 
                    -107, -103,
                    
                    -45.5,-45.5,
                    -23.5, -12]
        long_list = [-61.5, -57.5,
                     -42.5, -40.5, 
                     -2.5, -2,
                     
                     -24.5, -35,
                     -68.2, -68.2]
        yaw_list = [0.0, 0.0,
                    -61.5, -63.7, 
                    -90, -90,

                    -90,-90,
                    180,180]
        spawn_points = []
        
        random_vehicle_indexes = np.random.choice(len(lat_list), len(lat_list), replace=False)
        random_vehicle_indexes = sorted(random_vehicle_indexes)
        for i in random_vehicle_indexes:
            trans = carla.Transform()
            trans.location.x, trans.location.y, trans.location.z = lat_list[i], long_list[i], 0.1
            trans.rotation.yaw = yaw_list[i]
            spawn_points.append(trans)
        
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if (int(x.get_attribute('number_of_wheels')) == 4 and x.id != 'vehicle.volkswagen.t2' and x.id != 'vehicle.bmw.isetta')
                      and x.id != 'vehicle.carlamotors.*' and x.id=='vehicle.tesla.model3']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # randomly spwan actors in the random chosen spawned points
        batch = []
        for n, transform in enumerate(spawn_points):
            bp_sv = random.choice(blueprints)
            if bp_sv.has_attribute('color'):
                color = random.choice(bp_sv.get_attribute('color').recommended_values)
                bp_sv.set_attribute('color', color)
            if bp_sv.has_attribute('driver_id'):
                driver_id = random.choice(bp_sv.get_attribute('driver_id').recommended_values)
                # bp_sv.set_attself.vehicle_typeribute('driver_id', driver_id)
                bp_sv.set_attribute('driver_id', driver_id)
            bp_sv.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            light_state = vls.RightBlinker | vls.LeftBlinker | vls.Brake
                
            batch.append(SpawnActor(bp_sv, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_list.append(response.actor_id)
        
            
        ## Spawn walkers randomly
        self.walker_list = []
        self.walker_id = []
        walker_spawn_points = []
        self.surrounding_number_walker = 2
        x = [-60, -62]
        y = [-47.5, -48.5]
        
        for i in range(self.surrounding_number_walker):
            spawn_point = carla.Transform()
            spawn_point.location.x = x[i]
            spawn_point.location.y = y[i]
            spawn_point.location.z = 1
            spawn_point.rotation.yaw = 0
            walker_spawn_points.append(spawn_point)
        
        walker_batch = []
        self.walker_speed = []
        percentagePedestriansRunning = 0
        blueprints_walkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(blueprints_walkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    self.walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    self.walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                # print("Walker has no speed")
                self.walker_speed.append(0.0)
            walker_batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(walker_batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walker_list.append({"id": results[i].actor_id})
        
        self.walker_direction = []
        for i in range(len(self.walker_list)):
            td = carla.Vector3D()
            td.x = 1
            self.walker_direction.append(td)
            
        for i in range(len(self.walker_list)):
            self.walker_id.append(self.walker_list[i]["id"])
            
        self.walkers = self.world.get_actors(self.walker_id)    # 2 pedestrians
    
    
        ######### VERY IMPORTANT, METHOD FOUND BY JINGDA!!!!!!! ########
        self.obs_actors = self.world.get_actors(self.obs_list)  # 10 vehicles tesla model3

        iii = 0
        for v in self.obs_actors:
            self.traffic_manager.auto_lane_change(v,True)
            self.traffic_manager.vehicle_percentage_speed_difference(v, np.random.randint(-50,-20))
            self.traffic_manager.distance_to_leading_vehicle(v, np.random.randint(8,12))
            iii += 1
        
        self.speed_limit_obs_flags = np.zeros(iii)
        
        ## configurate and spawn the collision sensor
        # clear the collision history list
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        # spawn the collision sensor actor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
                bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        # obtain the collision signal and append to the history list
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: InterSection._on_collision(weak_self, event))
        
        ## reset the step counter
        self.count = 0
        self.count_yaw = 0
        self.reset_traj_dataset()
        # interpolated waypoints for this scenario map, otherwise perform unrealistic lane-change
        script_dir = os.path.dirname(__file__)
        self.wp = np.load(os.path.join(script_dir, 'map/wp.npy'))
        self.wp2 = np.load(os.path.join(script_dir, 'map/wp2.npy'))
        # self.wp2 = np.load(script_dir+'/map/wp2.npy')
        state = self.get_observation_scene()

        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"40")

        # camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))    
        cam_bird_eye_view = carla.Transform(carla.Location(x=self.CAMERA_POS_X, y=0.0, z=self.CAMERA_POS_Z),  # altura grande = bird view
                                            carla.Rotation(pitch=-90.0)               # olhar PARA BAIXO
)    
        self.sensor_seg = self.world.spawn_actor(self.sem_cam, cam_bird_eye_view, attach_to=self.ego_vehicle)
        self.actor_list.append(self.sensor_seg)
        self.sensor_seg.listen(lambda data: self.process_img(data))

        if self.image_seg is not None:
            self.image_for_CNN = self.apply_cnn(self.image_seg[:, :])  # shape = (280,)
    
        else:
            self.image_for_CNN = np.zeros(280, dtype=np.float32)

        # neighbor_trajs, ego_info, neighbor_wps = state
        # state = (neighbor_trajs, ego_info, neighbor_wps, self.image_for_CNN)

        # state_main = state                     # (6,10,5)
        # vision = self.image_for_CNN 

        return state, {"vision": self.image_for_CNN}
    
    

    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag
    
    ## search waypoints
    def get_position(self, waypoint):
        loc = waypoint.transform.location
        return [(loc.x, loc.y)]
    
    def depth_first_search(self, curr_waypoint, depth=0, max_depth=49):  
        if depth > max_depth:
            return [self.get_position(curr_waypoint)] 
        else:
            trasversed_lanes = []
            child_lanes = curr_waypoint.next(0.5)   # look for next waypoint 0.5m away (lista de waypoints)
            if len(child_lanes) > 0:
                for child in child_lanes:
                    trajs = self.depth_first_search(child, depth+1, max_depth)   # recursive search
                    trasversed_lanes.extend(trajs)
            if len(trasversed_lanes) == 0:
                return [self.get_position(curr_waypoint)]
            
            res = []
            for lane in trasversed_lanes:
                res.append(self.get_position(curr_waypoint) + lane)
            return res        # return a list of lists, each list is a lane
    
    def filter_and_pad(self, all_results, vehicle_location, k=3, length=50):
        lane_position = {}
        for i, result in enumerate(all_results):
            lane_position[i] = np.min(np.linalg.norm(np.array(result)-np.array(vehicle_location)[np.newaxis,:]) )
        sort_lanes = sorted(lane_position.items(), key=lambda x:x[1])[:k]
        
        new_result = np.zeros((k, length, 2))
        for i, lane in enumerate(sort_lanes):
            select_lane = np.array(all_results[lane[0]])[:length]
            new_result[i] = np.pad(select_lane, pad_width=[[0, length-select_lane.shape[0]], [0, 0]])
        return new_result 
    
    def fitler_goal_waypoints(self,results, goal, preview_dis):
        goal = np.array(goal)[np.newaxis,:]
        min_dist = []
        for result in results:
            m_dist = np.min(np.linalg.norm(np.array(result)-goal,axis=-1))
            min_dist.append(m_dist)
        arg = np.argmin(np.array(min_dist))
        traj = np.array(results[arg])
        return results[arg][preview_dis]
    
    def filter_initial_waypoints(self,result, ego_location, preview_dis):
        ego_location = np.array(ego_location)[np.newaxis,:]
        m_dist = np.argmin(np.linalg.norm(np.array(result)-ego_location,axis=-1))
        self.ego_wp = self.filter_and_pad([result], ego_location)
        return result[m_dist + preview_dis]
    

    def filter_planned_ego_waypoints(self, vehicle, preview_dis):
        location = vehicle.get_location()
        ego_location = [location.x, location.y]
        ego_location = np.array(ego_location)[np.newaxis,:]

        dist_1 = np.linalg.norm(self.wp-ego_location,axis=-1)
        dist_2 = np.linalg.norm(self.wp2-ego_location,axis=-1)
        min_d1, arg_md1 = np.min(dist_1), np.argmin(dist_1)
        min_d2, arg_md2 = np.min(dist_2), np.argmin(dist_2)
        if min_d1 < min_d2:
            r = self.wp[arg_md1 + preview_dis]
            rr = self.wp2[arg_md2 + preview_dis + 2]
            lr = None
        else:
            r = self.wp2[arg_md2 + preview_dis]
            lr = self.wp[arg_md1 + preview_dis + 2]
            rr = None
        
        return lr, r, rr

    
    def filter_ego_waypoints(self,vehicle, preview_dis):
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        vehicle_location = [location.x, location.y]
        left_results, right_results = None, None
        
        goal = [-52.5, -32]      # the goal point for this scenario
        results = self.depth_first_search(waypoint,max_depth=200)  # all possible lane segments from the current waypoint

        # plt.scatter(goal[0],goal[1],s=50, color='r')
        if vehicle_location[1]<-50:
            # r = self.fitler_goal_waypoints(results, goal, preview_dis)
            r = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)  
        else:
            r = self.fitler_goal_waypoints(results, goal, preview_dis)
        # plt.scatter(r[0],r[1],s=20, color='b')
        lr, rr = None, None
        
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None):
            
            if vehicle_location[1]<-50:
                lr = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)
            else:
                left_results = self.depth_first_search(waypoint.get_left_lane(),max_depth=200)
                lr = self.fitler_goal_waypoints(left_results, goal, preview_dis)
            # plt.scatter(lr[0],lr[1],s=20, color='b')
      
        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None):
            if vehicle_location[1]<-50:
                rr = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)
            else:
                right_results = self.depth_first_search(waypoint.get_right_lane(),max_depth=200)
                rr = self.fitler_goal_waypoints(right_results, goal, preview_dis)

        return lr, r, rr

    
    def get_all_waypoints(self, vehicle,judge=False): 
        location = vehicle.get_location()       # current location of the ego vehicle   
        waypoint = self.map.get_waypoint(location)   # get the nearest waypoint according to the current location
        vehicle_location = [location.x, location.y]
        
        left_results, right_results = None, None
        results = self.depth_first_search(waypoint)   # all possible lane segments from the current waypoint
        if judge:    # to judge whether the vehicle is off-route
            self.judge_off_route(location.x, location.y, results)   
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None): 
            left_results = self.depth_first_search(waypoint.get_left_lane())
            results.extend(left_results)
        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None): 
            right_results = self.depth_first_search(waypoint.get_right_lane())
            results.extend(right_results)
        
        goal = [-52.5, -32]  # the goal point for this scenario
        new_results = self.filter_and_pad(results, goal)
        return new_results  # return a (k, 50, 2) array, k is the number of lane segments
    
    def get_walker_waypoint(self, walker):
        x_walker, y_walker = walker.get_location().x, walker.get_location().y
        x = np.arange(0, 25.0, 0.5) + x_walker
        y = [y_walker]*50
        traj = np.stack([x, y],axis=1)
        return traj
    
    def select_top_actors(self, actors, walkers, vehicle_location, k=5):
        lane_position = {}
        for i, act in enumerate(actors):
            act_position = act.get_location()
            pos = [act_position.x, act_position.y]
            lane_position[i] = [np.linalg.norm(pos-np.array(vehicle_location)), 0] 
        for i, act in enumerate(walkers):
            act_position = act.get_location()
            pos = [act_position.x, act_position.y]
            lane_position[i] = [np.linalg.norm(pos-np.array(vehicle_location)), 1] 
        sort_lanes = sorted(lane_position.items(), key=lambda x:x[1][0])[:k]
        return sort_lanes  # return a list of tuples, each tuple is (actor_id, [distance, type])
    
    def reset_traj_dataset(self):
        self.traj_dataset = defaultdict()
        self.traj_dataset['ego'] = dict()              # defaultdict(None, {'ego': {}})
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_'+str(obs_id)] = dict() # defaultdict(None, {'ego': {}, 'v_0': {}, 'v_1': {}, 'v_2': {}, 'v_3': {}, 'v_4': {}, 'v_5': {}, 'v_6': {}, 'v_7': {}, 'v_8': {}, 'v_9': {}})
        for obs_id in range(len(self.walkers)):
            self.traj_dataset['w_'+str(obs_id)] = dict() # defaultdict(None, {'ego': {}, 'v_0': {}, 'v_1': {}, 'v_2': {}, 'v_3': {}, 'v_4': {}, 'v_5': {}, 'v_6': {}, 'v_7': {}, 'v_8': {}, 'v_9': {}, 'w_0': {}, 'w_1': {}})
    
    def angle_norm(self, yaw):
        theta = yaw - 90
        return (theta*np.pi/180 + np.pi) % (2*np.pi) - np.pi

    
    def get_actor_state(self, actor, types):
        return [actor.get_location().x, actor.get_location().y,
                     self.angle_norm(actor.get_transform().rotation.yaw),
                     actor.get_velocity().x, actor.get_velocity().y]     # 0 vehicle, 1 obs vehicle, 2 walker

    def record_one_step(self):
        self.traj_dataset['ego'][self.count] = self.get_actor_state(self.ego_vehicle, 0)   # [x, y, yaw, vx, vy] em cada step(ego)
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_'+str(obs_id)][self.count] = self.get_actor_state(self.obs_actors[obs_id],1)  # [x, y, yaw, vx, vy] em cada step(veiculos vizinhos)
        for obs_id in range(len(self.walkers)):
            self.traj_dataset['w_'+str(obs_id)][self.count] = self.get_actor_state(self.walkers[obs_id], 2)   # [x, y, yaw, vx, vy] em cada step(pedestres)

    
    def query_single_trajs(self,name):
        self_trajs = np.zeros((10, 5))
        queryed_trajs = self.traj_dataset[name]  # dict of the trajs of a single actor
        for i in range(10):
            queryed_time = self.count - i
            if queryed_time in queryed_trajs:
                self_trajs[-i, :] = np.array(queryed_trajs[queryed_time])
        return self_trajs
    
    def get_observation_scene(self):
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        self.record_one_step()   # record the current step state for all actors
        self.ego_location_history.append([x_ego, y_ego])
        if len(self.ego_location_history)==1:
            step_dist = 0
        else:
            step_dist = np.sqrt((self.ego_location_history[-2][0]-x_ego)**2 +(self.ego_location_history[-2][1]-y_ego)**2 )   # distance from last step to current step
        self.dist_travelled += step_dist   # total distance from the start point to current step
        
        ego_waypoint = self.get_all_waypoints(self.ego_vehicle,judge=True) # all possible lane segments from the current waypoint of ego vehicle
        ego_traj = self.query_single_trajs('ego')  # the past 10 steps trajectory of ego vehicle
        
        select_actor_ids = self.select_top_actors(self.obs_actors , self.walkers, [x_ego, y_ego])   # select 5 nearest actors (vehicles and pedestrians) # list of tuples (actor_id, [distance, type]) # type: 0 vehicle, 1 walker
        neighbor_waypoints = np.zeros((6, 3, 50, 2)) # (6, k, 50, 2), k is the number of lane segments # 6 = 1 ego + 5 neighbors  # 50 waypoints for each lane segment # 2 (x, y)
        ego_waypoint = self.filter_and_pad([self.wp,self.wp2], [x_ego, y_ego])   # (k, 50, 2) # k is the number of lane segments # 50 waypoints for each lane segment # 2 (x, y)
        neighbor_waypoints[0] = ego_waypoint # (k, 50, 2)
        neighbor_trajs = np.zeros((6, 10, 5))
        neighbor_trajs[0] = ego_traj # (10, 5) # 10 steps # 5 [x, y, yaw, vx, vy] do veículo ego
        for i, actor_id in enumerate(select_actor_ids):
            actor_type = actor_id[1][1]   # 0 vehicle, 1 walker
            index = actor_id[0]  # actor index in the corresponding list
            if actor_type==0:
                actor = self.obs_actors[index]   # ex: Actor(id=57, type=vehicle.tesla.model3, Actor(id=56, type=vehicle.tesla.model3), Actor(id=55, type=vehicle.tesla.model3), Actor(id=54, type=vehicle.tesla.model3)
                neighbor_waypoints[i+1] = self.get_all_waypoints(actor) # (3, 50, 2) = (3 lanes, 50 waypoins de cada lane, posição de cada waypoint) dos veículos vizinhos
                neighbor_trajs[i+1] = self.query_single_trajs('v_'+str(index)) # (10, 5) # 10 steps # 5 [x, y, yaw, vx, vy] dos quatro veículos vizinhos
            else:
                actor = self.walkers[index] # ex: Actor(id=58, type=walker.pedestrian.0026)
                neighbor_waypoints[i+1] = self.get_walker_waypoint(actor) # (3, 50, 2) = (3 lane, 50 waypoins de cada lane, posição de cada waypoint) do pedestre vizinho
                neighbor_trajs[i+1] = self.query_single_trajs('w_'+str(index))  # (10, 5) # 10 steps # 5 [x, y, yaw, vx, vy] do pedestre vizinho
        
        neighbor_waypoints = neighbor_waypoints.reshape(18, 50, 2)
        return (neighbor_trajs, ego_traj[-1], neighbor_waypoints[:,::5])  # shape de neighbor_waypoints (18, 10, 2) # 18 = 6*3 lanes # 10 waypoints for each lane segment # 2 (x, y)
    
    def action_adapter(self, model_action): 
        speed = model_action[0] # output (-1, 1)
        speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10) m/s
        
        speed = np.clip(speed, 0, 10)
        model_action[1] = np.clip(model_action[1], -1, 1)


        # discretization
        if model_action[1] < -1/3:
            lane = -1
        elif model_action[1] > 1/3:
            lane = 1
        else:
            lane = 0

        return (speed * 3.6, lane)
    
    def step(self, action):
        trans = self.ego_vehicle.get_transform()
        # if self.SHOW_CAM:
        #     self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),
        #                                                 carla.Rotation(yaw = -180, pitch=-90)))
        ## configurate the control command for the ego vehicle (if necessary)
        vx_ego = self.ego_vehicle.get_velocity().x
        vy_ego = self.ego_vehicle.get_velocity().y
        velocity_ego = (vx_ego**2 + vy_ego**2
                        + (self.ego_vehicle.get_velocity().z)**2)**(1/2)
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        acceleration_ego = ((self.ego_vehicle.get_acceleration().x)**2 + (self.ego_vehicle.get_acceleration().y)**2
                        + (self.ego_vehicle.get_acceleration().z)**2)**(1/2)

        self.y_ego = y_ego
        self.x_ego = x_ego
        self.acceleration_ego = acceleration_ego
        self.vx_ego = vx_ego
        self.vy_ego = vy_ego
        self.velocity_ego = velocity_ego
    
        self.world.tick()
        
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location()) 
        target_speed, lat_action = self.action_adapter(action)

        self.target_speed = target_speed

        self.agent.set_target_speed(self.target_speed)

        preview_dis = round(np.clip(velocity_ego*2, 1, 15))
        wp_list = self.filter_planned_ego_waypoints(self.ego_vehicle, preview_dis)

        lr, r, rr = wp_list
        
        ## ego vehicle's lateral plan
        if x_ego > -30:
            lat_action = 0
        try:
            preview_dis = round(np.clip(velocity_ego*2, 1, 15)) 
            if lat_action == -1:
                if (waypoint.lane_change & carla.LaneChange.Left != 0) and (lr is not None):
                    target_location = waypoint.get_left_lane().next(preview_dis)[0].transform.location
           
                    target_location.x  = lr[0]
                    target_location.y  = lr[1]
                    self.agent.set_destination(target_location)
            
                else:
                    target_location = waypoint.next(preview_dis)[0].transform.location
                    target_location.x  = r[0]
                    target_location.y  = r[1]
                    self.agent.set_destination(target_location)
            elif lat_action == 1:
                if (waypoint.lane_change & carla.LaneChange.Right != 0)and (rr is not None):
                    target_location = waypoint.get_right_lane().next(preview_dis)[0].transform.location
                    target_location.x  = rr[0]
                    target_location.y  = rr[1]
                    self.agent.set_destination(target_location)
                else:
                    target_location = waypoint.next(preview_dis)[0].transform.location
                    target_location.x  = r[0]
                    target_location.y  = r[1]
                    self.agent.set_destination(target_location)
            else:
                target_location = waypoint.next(preview_dis)[0].transform.location
                target_location.x  = r[0]
                target_location.y  = r[1]
                self.agent.set_destination(target_location)
        except:
            pass
        
        ## set target speeds of obs vehicles
        v_index=0
        for v in self.obs_actors:
            if v.get_speed_limit() > 80 and self.speed_limit_obs_flags[v_index]==0:
                self.traffic_manager.vehicle_percentage_speed_difference(v,np.random.randint(47,67))
                self.speed_limit_obs_flags[int(v_index)] = 1
            v_index += 1

        ## walkers control
        last_walker_location = np.zeros((self.surrounding_number_walker))
        for i in range(len(self.walker_list)):
            control_walker = self.walkers[i].get_control()
            control_walker.speed = float(self.walker_speed[i])
            control_walker.direction = self.walker_direction[i]
            if abs(self.walkers[i].get_location().x - last_walker_location[i]) < 0.0005: 
                control_walker.jump = True
            else:
                control_walker.jump = False
            self.walkers[i].apply_control(control_walker)
            last_walker_location[i] = self.walkers[i].get_location().x
        
        self.control = self.agent.run_step()
    
        ## achieve the control to the ego vehicle
        self.ego_vehicle.apply_control(self.control)
        
        ## obtain the state transition and other variables after taking the action (control command)
        next_state = self.get_observation_scene() 

        # if self.image_for_CNN is not None:
        if self.image_seg is not None:
            self.image_for_CNN = self.apply_cnn(self.image_seg[:, :])  # shape = (280,)
    
        else:
            self.image_for_CNN = np.zeros(280, dtype=np.float32)

        # next_fusion = tf.concat([next_state[1], self.image_for_CNN], axis=0)
        # next_neighbor_trajs, next_ego_info, next_neighbor_wps = next_state
        # next_state = (next_neighbor_trajs, next_ego_info, next_neighbor_wps, self.image_for_CNN)

        # if self.SHOW_CAM and self.image_seg is not None:
        #     cv2.imshow('Bird Eye View Image', self.image_seg)
        #     cv2.waitKey(1)

        ## detect if the step is the terminated step, by considering: collision and episode fininsh
        self.collision = self.get_collision_history()[1]
        self.finish = (y_ego > -32) and (x_ego > -54 and x_ego <-50.5)
        self.max_time = self.count > 300
        
        success = 1 if self.finish else 0
        coll = -1 if self.collision else 0
        
        if self.finish or self.collision or self.off_route or self.max_time:
            done = True
        else:
            done = False
        
        reward = success + coll 

        # info = (self.finish, self.collision, self.off_route, self.max_time)
        info = {
                    "finish": self.finish,
                    "collision": self.collision,
                    "off_route": self.off_route,
                    "max_time": self.max_time,
                    "vision" : self.image_for_CNN,
                }
        self.count += 1

        if done:
            self.destroy()

        return next_state, reward, done, info
    
    
    def judge_off_route(self, x, y, waypoint):
        min_list = []
        for wp in waypoint:
            dist = np.array(waypoint) - np.array([x, y])[np.newaxis,...]
            dist = np.linalg.norm(dist, axis=-1)
            min_list.append(np.min(dist))
        
        self.off_route = False if np.min(min_list) < 2 else True  
        if x < -55 or y < -70:
            self.off_route = True
        if (y > -25) and not (x > -54 and x <-50.5):
            self.off_route = True
        if (y<-66) and x<-30:
            self.off_route = True

    # def destroy(self):

    #     self.collision_sensor.stop()
    #     actors = [
    #         self.ego_vehicle,
    #         self.collision_sensor,
    #         ]

    #     self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        
    #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.obs_list])
        
    #     self.collision_sensor = None
    #     self.ego_vehicle = None
#################################### destroy melhorado ####################################################
    def destroy(self):
        # para sensores
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
            
            if self.sensor_seg is not None:
                self.sensor_seg.stop()
        except:
            pass

        cmd = []

        # ego + sensor de colisão
        if getattr(self, "ego_vehicle", None) is not None:
            cmd.append(carla.command.DestroyActor(self.ego_vehicle))
        if getattr(self, "collision_sensor", None) is not None:
            cmd.append(carla.command.DestroyActor(self.collision_sensor))

        # veículos NPC criados (obs_list contém ids)
        if getattr(self, "obs_list", None):
            for aid in self.obs_list:
                cmd.append(carla.command.DestroyActor(aid))
            self.obs_list = []

        # walkers (usando listas walker_id e walkers, se existirem)
        if getattr(self, "walker_id", None):
            for wid in self.walker_id:
                cmd.append(carla.command.DestroyActor(wid))
            self.walker_id = []
        if getattr(self, "walkers", None):
            for w in self.walkers:
                try:
                    cmd.append(carla.command.DestroyActor(w.id))
                except:
                    pass
            self.walkers = []

        if cmd:
            try:
                self.client.apply_batch_sync(cmd, True)
            except:
                # fallback best-effort
                for c in cmd:
                    try:
                        self.client.apply_batch([c])
                    except:
                        pass

        self.collision_sensor = None
        self.ego_vehicle = None
        self.sensor_seg = None