import numpy as np
import math
from math import pi, sqrt, cos, sin, inf, degrees
from obstacle import Obstacle

import main 

# Usefull for avoiding lead vehicle hooking and unhooking
LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE = 5

# Constant
VEHICLE = 0
PEDESTRIAN = 1

#CAR_LONG_SIDE = 2
#VEHICLE_LABEL = 10
#PEDESTRIAN_LABEL = 4

class ObstaclesManager:


    def __init__(self, lead_vehicle_lookahead_base, vehicle_obstacle_lookahead_base, pedestrian_obstacle_lookahead, behavioral_planner):
        """
        The constructor takes in input some distance measurements and the behavioural planner instance.
        An empty dictionary is created for obstacles.   
        
        args:
            lead_vehicle_lookahead_base: base distance within which to find lead vehicles
            vehicle_obstacle_lookahead_base: base distance within which to find vehicles considered as obstacles
            pedestrian_obstacle_lookahead: distance within which to find pedestrian as obstacles
            behavioral_planner: useful to check if _follow_lead_vehicle in the bp is activated

        variables to set:
            self._obstacles: an empty dictionary for store obstacles, both pedestrians and vehicles
        """      
        self._lead_vehicle_lookahead_base = lead_vehicle_lookahead_base
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base

        self._vehicle_obstacle_lookahead_base =  vehicle_obstacle_lookahead_base
        self._vehicle_obstacle_lookahead =  self._vehicle_obstacle_lookahead_base
        
        self._pedestrian_obstacle_lookahead = pedestrian_obstacle_lookahead
        
        self._bp = behavioral_planner

        self._obstacles = {}

    def get_om_state(self, measurement_data, ego_pose):
        """
        Set measurement data and ego pose, updates the information of pedestrians and vehicles in the obstacles
        dictionary; finally return obstacles as bounding box and their projections in the future.
        
        args:
            measurement_data: information read from the server about all agents in the simulation and so 
                also information for pedestrians and vehicles
            ego_pose: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
        
        returns:
            all_vehicles_on_sight: list containing all vehicle within the _vehicle_obstacle_lookahead distance  
            obstacles: list containing all vehicles and pedestrians that fall within the distances provided for them,
                but only vehicles that aren't on the same lane of the ego vehicle
            future_obstacles: list containing the future projection for all obstacles mentioned above
            lead_vehicle: if exists, it contains the lead vehicle agent (which will be eliminated from the list of 
                obstacles mentioned above).
        """        
        self._set_measurement_data_frame(measurement_data)
        self._set_ego_location(ego_pose)

        obstacles = []
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
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3
        self._vehicle_obstacle_lookahead = self._vehicle_obstacle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3

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
                        try:
                            self._obstacles.pop(agent.id)
                        except KeyError:
                            pass
                        # print(f"SAME LANE - distance: {distance} - orientation: {math.acos(orientation)}")
                        # check if the vehicle on the same lane is a lead vehicle 
                        if self._check_for_lead_vehicle(location):
                            if distance < lead_vehicle_dist:
                                lead_vehicle_dist = distance
                                lead_vehicle = agent.vehicle
                    else:
                        self._add_obstacle(agent.vehicle, agent.id, VEHICLE)
                else:   
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        pass
        
        return all_vehicles_on_sight, lead_vehicle

    def _update_pedestrian(self):
        #if self._check_instance_pedestrian():
        #print("Update pedestrian")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('pedestrian'):
                location = agent.pedestrian.transform.location
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                if dist < self._pedestrian_obstacle_lookahead:
                    self._add_obstacle(agent.pedestrian, agent.id, PEDESTRIAN)
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
        # print(f"ego_heading_vector: {ego_heading_vector} - car_heading_vector: {car_heading_vector}")

        dot_product = np.dot(car_heading_vector, ego_heading_vector)
        # print(f"dot_product: {dot_product:.2f}")
        
        return car_distance, dot_product

    def _check_for_vehicle_on_same_lane(self, orientation):
        return orientation > cos(math.radians(45))

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
        distance_from_lead = self._lead_vehicle_lookahead if not self._bp._follow_lead_vehicle else  self._lead_vehicle_lookahead + LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE

        return (car_distance < distance_from_lead) and (dot_product > 1/sqrt(2))
    

    def _add_obstacle(self, obstacle, id, agent_type):
        if id in self._obstacles:
            self._obstacles[id].update_state(obstacle)
        else:
            self._obstacles[id] = Obstacle(obstacle, agent_type)

    def _get_obstacles(self):
        obstacles = []
        future_obstacles = []
        for id, obs in self._obstacles.items():
            obstacles.append(obs.get_current_location())
            for loc in obs.get_future_locations():
                future_obstacles.append(loc)
        return obstacles, future_obstacles
         

                