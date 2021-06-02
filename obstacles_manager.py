import numpy as np
import math
from math import pi, sqrt, cos, sin, inf, degrees
from obstacle import Obstacle

import main

LEAD_VEHILCE_LOOKAHEAD_OFFSET_FOR_RELEASE = 5
# CAR_LONG_SIDE = 2
#VEHICLE_LABEL = 10
#PEDESTRIAN_LABEL = 4

class ObstaclesManager:

    def __init__(self, lead_vehicle_lookahead_base, vehicle_obstacle_lookahead_base, pedestrian_obstacle_lookahead, behavioral_planner):
        self._lead_vehicle_lookahead_base = lead_vehicle_lookahead_base
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base

        self._vehicle_obstacle_lookahead_base =  vehicle_obstacle_lookahead_base
        self._vehicle_obstacle_lookahead =  self._vehicle_obstacle_lookahead_base
        
        self._pedestrian_obstacle_lookahead = pedestrian_obstacle_lookahead
        
        self._bp = behavioral_planner

        self._obstacles = {}

    def get_om_state(self, measurement_data, ego_pose, semantic_img=None):        
        #self._set_current_frame(semantic_img)
        self._set_measurement_data_frame(measurement_data)
        self._set_ego_location(ego_pose)

        obstacles = []
        #if self.semantic_img is not None:
        all_vehicles_on_sight, lead_vehicle = self._update_vehicles()
        self._update_pedestrian()
        # obstacles = obstacles_vehicles + obstacles_pedestrian
        obstacles, future_obstacles = self._get_obstacles()
        
        return all_vehicles_on_sight, obstacles, future_obstacles, lead_vehicle

    def _set_measurement_data_frame(self, measurement_data):
        self.measurement_data = measurement_data

    def _set_ego_location(self, ego_pose):
        self._ego_pose = ego_pose
       
    def compute_lookahead(self, ego_speed):
        # safe brake space
        speed_km_h = (ego_speed * 3600)/1000
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base + (speed_km_h/10)*3

        self._vehicle_obstacle_lookahead = self._vehicle_obstacle_lookahead_base + (speed_km_h/10)*3

        # print(F"New speed for vehicle lead {ego_speed}")
        # print(F"New look ahead for vehicle lead {self._lead_vehicle_lookahead}")

    def _update_vehicles(self):
        all_vehicles_on_sight = []
        lead_vehicle_dist = inf
        lead_vehicle = None
        
        # print("Compute distance and orientation")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('vehicle'):
                location = agent.vehicle.transform.location
                rotation = agent.vehicle.transform.rotation
                dimension = agent.vehicle.bounding_box.extent
                distance, orientation = self._compute_distance_orientation_from_vehicle(location, rotation)
                # the vehicle is inside the obstacle range
                if distance < self._vehicle_obstacle_lookahead:
                    # print(f"VEHICLE ID: {agent.id} - VEHICLE SPEED: {agent.vehicle.forward_speed}")
                    # compute the bb with respect to the world frame
                    box_pts = main.obstacle_to_world(location, dimension, rotation)
                    all_vehicles_on_sight.append(box_pts)
                    # check for vehicle on the same lane
                    if self._check_for_vehicle_on_same_lane(orientation):
                        # check if the vehicle on the same lane is a lead vehicle 
                        if self._check_for_lead_vehicle(location):
                            if distance < lead_vehicle_dist:
                                lead_vehicle_dist = distance
                                lead_vehicle = agent.vehicle
                    else:
                        # the vehicle is not in the same lane.
                        # It is added as obstacle
                        #print("Vehicle at distance:" + str(distance))
                        self._add_obstacle(agent.vehicle, agent.id)
                        '''
                        future_boxes_pts = self.predict_future_location(agent.vehicle, box_pts)
                        for box in future_boxes_pts:
                            all_future_vehicles.append(box)
                        obstacles_vehicles.append(box_pts)
                        '''
                else:   
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        pass
        
        return all_vehicles_on_sight, lead_vehicle

    def _update_pedestrian(self):
        obstacles_pedestrian = []
        #if self._check_instance_pedestrian():
        #print("Update pedestrian")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('pedestrian'):
                location = agent.pedestrian.transform.location
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                if dist < self._pedestrian_obstacle_lookahead:
                    self._add_obstacle(agent.pedestrian, agent.id)
                else:
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        pass


    def _compute_distance_orientation_from_vehicle(self, car_location, car_rotation):
        # compute the distance between the player agent and the obstacle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)

        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        car_heading_vector = [cos(math.radians(car_rotation.yaw)), sin(math.radians(car_rotation.yaw))]

        dot_product = np.dot(car_heading_vector, ego_heading_vector)
        
        return car_distance, dot_product

    def _check_for_vehicle_on_same_lane(self, orientation):
        return orientation > 1/sqrt(2)

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def _check_for_lead_vehicle(self, car_location):
        # compute the distance from the vehicle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)
        
        # compute the versor of the vector distance
        lead_car_delta_vector = np.divide(car_delta_vector, car_distance)
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        
        # compute the angle between the car heading versor and distance versor
        dot_product = np.dot(lead_car_delta_vector, ego_heading_vector)
        #print(F"CAR DISTANCE {car_distance} Orientation: {dot_product}")
        distance_from_lead = self._lead_vehicle_lookahead if not self._bp._follow_lead_vehicle else  self._lead_vehicle_lookahead + LEAD_VEHILCE_LOOKAHEAD_OFFSET_FOR_RELEASE

        return (car_distance < distance_from_lead) and (dot_product > 1/sqrt(2))
    

    def _add_obstacle(self, obstacle, id):
        if id in self._obstacles:
            self._obstacles[id].update_state(obstacle)
        else:
            self._obstacles[id] = Obstacle(obstacle)

    def _get_obstacles(self):
        obstacles = []
        future_obstacles = []
        for id, obs in self._obstacles.items():
            obstacles.append(obs.get_current_location())
            for loc in obs.get_future_locations():
                future_obstacles.append(loc) 
        return obstacles, future_obstacles
         

                