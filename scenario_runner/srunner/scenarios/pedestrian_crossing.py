#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import random
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                    KeepVelocity,
                                                                    Idle,
                                                                    ActorTransformSetter,
                                                                    )

from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                                DriveDistance,
                                                                                )

from srunner.scenarios.basic_scenario import BasicScenario


class PedestrianCrossing(BasicScenario):
    """
    Grupo de pedestres atravessando a rua.
    Versão adaptada para uso com Gym + CARLA + ScenarioRunner antigo.
    """

    def __init__(self, 
                 world, 
                 ego_vehicles, 
                 config, 
                 debug_mode=False, 
                 criteria_enable=True, 
                 timeout=60
                 ):
        
        self.other_actors = []

        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)

        self._rng = random.Random()

        self._ego_end_distance = 40.0
        # self._min_trigger_dist = 12.0
        self.timeout = timeout

        self._walker_data = []
        route = config.route 

        points = []
        for point, _ in route:
            points.append(point)
        
        dist_route = 0.0
        for i in range(1, len(points)):
            p1 = points[i-1].location
            p2 = points[i].location
            dist_route += p1.distance(p2) 

        delta_dist = self._rng.uniform(0.2 * dist_route, 0.6 * dist_route)
        random_pedestrians = random.randint(1, 6) 

        for i in range(random_pedestrians):
            walker = {}
            spacing = self._rng.uniform(0.5, 1.0)  # random space between pedestrians
            offset_x = 0.0
            for _ in range(i):
                offset_x += spacing  # sum of accumulated spacings
            
            walker["x"] = delta_dist + offset_x
            walker["y"] = self._rng.uniform(0.0, 1.5) # slight variation in the Y (lateral)
            walker["z"] = 1.2
            walker["yaw"] = 270.0

            self._walker_data.append(walker)

        max_ego_speed = 30.0 / 3.6   # 8.33 m/s
        estimated_ego_speed = 0.6 * max_ego_speed

        for walker_data in self._walker_data:
            walker_data["idle_time"] = self._rng.uniform(0.0, 1.5)  # time the pedestrian will remain still
            walker_data["speed"] = self._rng.uniform(1.3, 2.0)      # pedestrian movement speed
            reaction_time = self._rng.uniform(2.0, 5.0)       # Reaction time between 2 and 5 seconds
            # walker_data["trigger_dist"] = min(self._min_trigger_dist, estimated_ego_speed * reaction_time) # Calculates pedestrian activation distance based on v = 0.6 * vmax
            walker_data["trigger_dist"] = estimated_ego_speed * reaction_time # Calculates pedestrian activation distance based on v = 0.6 * vmax


        super().__init__("PedestrianCrossing",
                        ego_vehicles,
                        config,
                        world,
                        debug_mode,
                        criteria_enable=criteria_enable,
                        )
        

    def _get_walker_transform(self, wp, displacement):
        disp_x = displacement["x"]
        disp_y = displacement["y"]
        disp_z = displacement["z"]
        disp_yaw = displacement["yaw"]

        # Displace it to the crosswalk. Move forwards towards the crosswalk 
        start_vec = wp.transform.get_forward_vector()
        start_right_vec = wp.transform.get_right_vector()

        spawn_location = wp.transform.location + carla.Location(disp_x * start_vec.x + disp_y * start_right_vec.x,
                                                                disp_x * start_vec.y + disp_y * start_right_vec.y,
                                                                disp_x * start_vec.z + disp_y * start_right_vec.z + disp_z,
                                                                )

        # spawn_rotation = carla.Rotation(pitch=wp.transform.rotation.pitch,
        #                                 yaw=wp.transform.rotation.yaw + disp_yaw,
        #                                 roll=wp.transform.rotation.roll,
        #                                 )

        spawn_rotation = wp.transform.rotation
        spawn_rotation.yaw += disp_yaw

        return carla.Transform(spawn_location, spawn_rotation)
    
    
    def _adjust_to_nearest_sidewalk(self, spawn_transform, sid_wps):
        '''
        Sets the pedestrian spawn position to the nearest sidewalk waypoint.

        Parameters:
        - spawn_transform: desired carla.Transform for the pedestrian (can be outside the sidewalk)
        - sid_wps: list of carla.Waypoints that are on Sidewalk, Shoulder or Parking

        Returns:
        - carla.Transform set to sidewalk
        '''

        if not sid_wps:
            raise ValueError('The sidewalk waypoint list (sid_wps) is empty.')

        closest_wp = min(sid_wps, key=lambda wp: wp.transform.location.distance(spawn_transform.location))

        location = closest_wp.transform.location + carla.Location(z=1.2)
        road_yaw = closest_wp.transform.rotation.yaw
        adjusted_yaw = (road_yaw - 90.0) % 360.0
        rotation = carla.Rotation(pitch=0.0, yaw=adjusted_yaw, roll=0.0)

        return carla.Transform(location, rotation)
    
    
    def _initialize_actors(self, config):
        # Get the start point of the initial pedestrian
        self._collision_wp = self._reference_waypoint
        # Get the crosswalk start point
        start_wp = self._collision_wp
        while start_wp.lane_type != carla.LaneType.Sidewalk and \
                start_wp.lane_type != carla.LaneType.Shoulder and \
                start_wp.lane_type != carla.LaneType.Parking:  # Sidewalk, Shoulder, and Parking
            
            wp = start_wp.get_right_lane()
            if wp is None:
                raise ValueError("Couldn't find a waypoint to start the flow")
            start_wp = wp

        
        init_wp = start_wp
        roadid = init_wp.road_id
        sid_wps = [init_wp]
        for i in range(200):
            next_wps = init_wp.next(0.5)
            if not next_wps:
                break
            next_wp = next_wps[0]
            if next_wp.road_id != roadid: 
                break
            init_wp = next_wp
            sid_wps.append(init_wp)

        
        # Spawn the walkers
        for i, walker_data in enumerate(self._walker_data):
            spawn_transform = self._get_walker_transform(start_wp, walker_data)   # Absolute position of the pedestrian

            spawn_transform = self._adjust_to_nearest_sidewalk(spawn_transform, sid_wps)

            walker = CarlaDataProvider.request_new_actor('walker.*', spawn_transform)
            
            if walker is None:
                for walker in self.other_actors:   # self.other_actors é uma variavel de instancia da classe pai
                    walker.destroy()
                raise ValueError("Failed to spawn an adversary")

            walker.set_location(spawn_transform.location + carla.Location(z=-200))
            walker = self._replace_walker(walker)

            self.other_actors.append(walker)

            collision_dist = spawn_transform.location.distance(self._collision_wp.transform.location)

            # Distance and duration to cross the whole road + a bit more (supposing symetry in both directions)
            move_dist = 2.3 * collision_dist

            walker_data['transform'] = spawn_transform
            walker_data['distance'] = move_dist
            walker_data['duration'] = move_dist / walker_data['speed']

    
    def _create_behavior(self):
        sequence = py_trees.composites.Sequence(name="PedestrianCrossing")

        if not self.other_actors or not self._walker_data:
            sequence.add_child(
                DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition")
            )
            return sequence

        for walker_actor, walker_data in zip(self.other_actors, self._walker_data):
            sequence.add_child(ActorTransformSetter(walker_actor, walker_data["transform"], True))

        main_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
            name="WalkerMovement",
        )

        for i, (walker_actor, walker_data) in enumerate(zip(self.other_actors, self._walker_data)):
            trigger = InTriggerDistanceToLocation(
                self.ego_vehicles[0],
                walker_data["transform"].location,
                walker_data["trigger_dist"],
            )

            walker_sequence = py_trees.composites.Sequence(name=f"WalkerCrossing_{i}")
            walker_sequence.add_child(trigger)
            walker_sequence.add_child(Idle(walker_data["idle_time"]))
            walker_sequence.add_child(
                KeepVelocity(
                    walker_actor,
                    walker_data["speed"],
                    duration=walker_data["duration"],
                    distance=walker_data["distance"],
                    name=f"WalkerKeepVelocity_{i}",
                )
            )
            walker_sequence.add_child(
                ActorDestroy(walker_actor, name=f"DestroyWalker_{i}")
            )

            main_behavior.add_child(walker_sequence)

        main_behavior.add_child(
            DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition")
        )

        sequence.add_child(main_behavior)
        return sequence
    

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]
    

    def __del__(self):
        """
        Remove all actors upon deletion
        """   
        self.remove_all_actors()

    # TODO: Pedestrian have an issue with large maps were setting them to dormant breaks them,
    # so all functions below are meant to patch it until the fix is done
    def _replace_walker(self, walker):
        """As the adversary is probably, replace it with another one"""
        type_id = walker.type_id
        walker.destroy()
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("Couldn't spawn the walker substitute")
        walker.set_simulate_physics(False)
        walker.set_location(spawn_transform.location + carla.Location(z=-50))
        return walker
    

    # def _setup_scenario_trigger(self, config):
    #     """Normal scenario trigger but in parallel, a behavior that ensures the pedestrian stays active"""
    #     trigger_tree = super()._setup_scenario_trigger(config)

    #     if not self.route_mode:
    #         return trigger_tree

    #     parallel = py_trees.composites.Parallel(
    #         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="ScenarioTrigger")

    #     for i, walker in enumerate(reversed(self.other_actors)):
    #         parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], walker, 100))

    #     parallel.add_child(trigger_tree)
    #     return parallel
