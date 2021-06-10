import numpy as np
import math
from math import pi, sqrt, cos, sin, inf, degrees
from obstacle import Obstacle

import main 

# Usefull for avoiding lead vehicle hooking and unhooking
LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE = 5

# Constant indicating the type of agent
VEHICLE = 0
PEDESTRIAN = 1

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
        
        self._pedestrian_obstacle_lookahead_base = pedestrian_obstacle_lookahead
        self._pedestrian_obstacle_lookahead = self._pedestrian_obstacle_lookahead_base
        
        # The instance of behavioural planner passed
        self._bp = behavioral_planner

        # Dictionary containing the obstacles
        self._obstacles = {}


    def get_om_state(self, measurement_data, ego_pose):
        """
        Set measurement data and ego pose, updates the information of pedestrians and vehicles in the 
        obstacles dictionary; finally return obstacles as bounding box and their projections in the future.
        
        args:
            measurement_data: information read from the server about all agents in the simulation and so 
                also information for pedestrians and vehicles
            ego_pose: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
        
        returns:
            on_same_lane_vehicles: list containing all vehicle within the _vehicle_obstacle_lookahead distance
                and on the same lane of the ego vehicle.  
            obstacles: list containing all vehicles and pedestrians that fall within the distances provided for them,
                but only vehicles that aren't on the same lane of the ego vehicle
            future_obstacles: list containing the future projection for all obstacles mentioned above
            lead_vehicle: if exists, it contains the lead vehicle agent (which will be eliminated from the list of 
                obstacles mentioned above).
        """        
        # Set the current measurement from server and the currente ego pose. 
        self._set_measurement_data_frame(measurement_data)
        self._set_ego_location(ego_pose)

        obstacles = []

        # We search among non-layer agents for vehicles that can be identified 
        # as obstacles, as vehicles on the same lane or as lead vehicles 
        on_same_lane_vehicles, lead_vehicle = self._update_vehicles()

        # We search among non-layer agents for pedestrians that can be identified
        # as obstacles
        self._update_pedestrian()

        # We obtain separately the bounding boxes for the current obstacles and 
        # the future predictions, in the world frame
        obstacles, future_obstacles = self._get_obstacles()
        
        return obstacles, future_obstacles, on_same_lane_vehicles, lead_vehicle


    def _set_measurement_data_frame(self, measurement_data):
        """
        Set measurement data for the current frame.
        
        args:
            measurement_data: information read from the server about all agents in the simulation 
                and so also information for pedestrians and vehicles
        """
        self.measurement_data = measurement_data


    def _set_ego_location(self, ego_pose):
        """
        Set ego pose for the current frame.
        
        args:
            ego_pose: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
        """   
        self._ego_pose = ego_pose


    def compute_lookahead(self, ego_speed):
        """
        Using the current speed of the ego vehicle, converts it from m/s in km/h format and determines 
        the _lead_vehicle_lookahead and _vehicle_obstacle_lookahead adding to the base the x extension of 
        the carand a dynamic value depending on braking distance heuristic (computed starting from speed).
        
        args:
            ego_speed: the current speed of the ego vehicle (m/s)
        
        variables to set:
            self._lead_vehicle_lookahead: distance within which to find lead vehicles
            self._vehicle_obstacle_lookahead: distance within which to find vehicles as obstacle
        """ 

        # Converting the speed from m/s to km/h
        speed_km_h = (ego_speed * 3600)/1000

        # Redefinition of the lookahead distances starting from those bases and adding a part that depends on the dimension 
        # along x of the ego vehicle, and another part that depends on a braking distance heuristic.
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3
        self._vehicle_obstacle_lookahead = self._vehicle_obstacle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3


    def _update_vehicles(self):
        """
        Iterates on the list of non_player_agents finding vehicles and check if a vehicle can be
        considered as an obstacle, if it falls in the range of the _vehicle_obstacle_lookahead distance;
        furthermore, if a vehicle has the same orientation of the ego vehicle, it is added to the list of 
        the vehicles on the same lane. Finally, the nearest vehicle on the same lane in front of the ego 
        vehicle is considered as the lead vehicle.
        
        returns:
            on_same_lane_vehicles: list containing all vehicle within the _vehicle_obstacle_lookahead distance
                and on the same lane of the ego vehicle.  
            lead_vehicle: if exists, it contains the lead vehicle agent (which will be eliminated from the list 
                of obstacles mentioned above). 
        """   
        # An empty list for vehicles on the same lane is initialized
        on_same_lane_vehicles = []

        # Let's say there is no vehicle lead and, therefore, the distance is infinite
        lead_vehicle_dist = inf
        lead_vehicle = None
        
        for agent in self.measurement_data.non_player_agents:
            # Only if the agent is a vehicle
            if agent.HasField('vehicle'):
                # Obtain some useful measurement about the vehicle
                location = agent.vehicle.transform.location
                rotation = agent.vehicle.transform.rotation
                dimension = agent.vehicle.bounding_box.extent

                # Compute the distance and the orientation between the player agent and the vehicle
                distance, orientation = self._compute_distance_orientation_from_vehicle(location, rotation)

                # The vehicle is inside the vehicle obstacle range, determine dinamically from current ego speed
                if distance < self._vehicle_obstacle_lookahead:

                    # Compute the bb with respect to the world frame
                    box_pts = main.obstacle_to_world(location, dimension, rotation)
                    
                    # Check for vehicle on the same lane
                    if self._check_for_vehicle_on_same_lane(orientation):
                        
                        # Adding its bounding box in the worrld frame inside the list 
                        on_same_lane_vehicles.append(box_pts)

                        try:
                            # If a vehicle is in the obstacle range and is on the same lane, removes it from the dictionary of all obstacles
                            self._obstacles.pop(agent.id)
                        except KeyError:
                            # If the obstacle is not present in the dictionary, an exception is raised trying 
                            # to eliminate it. We intercept it, and pass
                            pass

                        # Check if the vehicle on the same lane is a lead vehicle, checking if the distance between ego and vehicle
                        # is within the distance_from_lead range and if the cosine of the angle between the versors of the two agents is
                        # smaller than 45°
                        if self._check_for_lead_vehicle(location):
                            if distance < lead_vehicle_dist:
                                lead_vehicle_dist = distance
                                lead_vehicle = agent.vehicle
                    
                    # If it is not a vehicle on the same lane
                    else:
                        # Adding or update the agent inside the obstacle dictionary using its id as key 
                        self._add_obstacle(agent.vehicle, agent.id, VEHICLE)
                
                # If the vehicle is outside the obstacle range,, try to eliminate it from the dictionary of all obstacles
                else:
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        # If the obstacle is not present in the dictionary, an exception is raised trying 
                        # to eliminate it. We intercept it, and pass
                        pass
        
        return on_same_lane_vehicles, lead_vehicle


    def _update_pedestrian(self):
        """
        Iterates on the list of non_player_agents finding pedestrians and check if a pedestrian can be
        considered as an obstacle, if it falls in the range of the _pedestrian_obstacle_lookahead distance.
        """   

        for agent in self.measurement_data.non_player_agents:
            # Only if the agent is a pedestrian
            if agent.HasField('pedestrian'):

                # Obtain the location of the pedestrian in the world frame
                location = agent.pedestrian.transform.location

                # Compute the Euclidean distace between the ego and the pedestrian
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                
                # The pedestrian is inside the pedestrian obstacle range
                if dist < self._pedestrian_obstacle_lookahead:

                    # Adding or update the agent inside the obstacle dictionary using its id as key
                    self._add_obstacle(agent.pedestrian, agent.id, PEDESTRIAN)
                
                # If the pedestrian is outside the obstacle range, try to eliminate it from the dictionary of all obstacles
                else:
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        # If the obstacle is not present in the dictionary, an exception is raised trying 
                        # to eliminate it. We intercept it, and pass
                        pass


    def _compute_distance_orientation_from_vehicle(self, car_location, car_rotation):
        """
        Compute the distance and the cosine of the angle between the player agent and the obstacle
        
        args:
            car_location: the position (x,y,z) of the vehicle respect to world frame
            car_rotation: the orientation (yaw, pitch, roll) of the vehicle respect to the world frame
        
        returns:
            car_distance: distance between the player agent and the obstacle
            dot_product: cosine of the angle between the player agent and the obstacle
        """
        # Compute the distance between the player agent and the obstacle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)

        # Compute the versor of ego based on its yaw angle
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]

        # Compute the versor of vehicle based on its yaw angle
        car_heading_vector = [cos(math.radians(car_rotation.yaw)), sin(math.radians(car_rotation.yaw))]

        # Compute the dot produt for calculate the value of cosine of the angle between the versors
        dot_product = np.dot(car_heading_vector, ego_heading_vector)
        
        return car_distance, dot_product


    def _check_for_vehicle_on_same_lane(self, orientation):
        """
        Check if the cosine of the angle between the player agent and the obstacle is greater than the 
        cosine of 45°.
        
        args:
            orientation: cosine of the angle between the player agent and the obstacle
        
        returns:
            True: if orientation is greater than the cosine of 45°; else False. 
        """
        return orientation > cos(math.radians(45))


    def _check_for_lead_vehicle(self, car_location):
        """
        Compute the distance from the vehicle, compute the versor of the vector distance, compute the 
        angle between the car heading versor and distance versor and return True if is a really lead 
        vehicle. This for to see if we need to modify our velocity profile to accomodate the lead vehicle.

        args:
            car_location: the position (x,y,z) of the vehicle respect to world frame
        
        returns:
            True: if is a lead vehicle.
        """
        # Compute the distance between the player agent and the obstacle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)
        
        # Compute the versor of the vector distance
        lead_car_delta_vector = np.divide(car_delta_vector, car_distance)

        # Compute the versor of ego based on its yaw angle
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        
        # Compute the cosine of the angle between the car heading versor and distance versor
        dot_product = np.dot(lead_car_delta_vector, ego_heading_vector)

        # The distansce used tu check if a vehicle is a lead vehicle, is set to the threshold 
        # lead_vehicle_lookahead if the follow_lead_vehicle flag of behavioral planner is False, 
        # otherwise it is set to lead_vehicle_lookahead plus an offset, to prevent the lead 
        # vehicle hooking and unhooking.
        distance_from_lead = self._lead_vehicle_lookahead if not self._bp._follow_lead_vehicle else  self._lead_vehicle_lookahead + LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE

        # Perform the two check, if the distance is smaller than distance_from_lead threshold, and 
        # if the cosine of the angle between the car heading versor and distance versor is greather
        # than the cosine of 45°
        return (car_distance < distance_from_lead) and (dot_product > 1/sqrt(2))
    

    def _add_obstacle(self, obstacle, id, agent_type):
        """
        Create a new instance of Obstacle if it is not already present inside the obstacles dictionary;
        otherwise updates the informations of instance already existing.

        args:
            obstacle: the instance of the non_player_agents that must be entered or updated inside the obstacles dictionary.
            id: unique identifier of the non_player_agents
            agent_type: VEHICLE (0) or PEDESTRIAN (1)
        """

        # If the id is already used as key inside the dictionary, its state is only updated
        if id in self._obstacles:
            self._obstacles[id].update_state(obstacle)

        # If If the obstacle is not present, it is inserted inside the dictionary
        else:
            self._obstacles[id] = Obstacle(obstacle, agent_type)


    def _get_obstacles(self):
        """
        Iterates on obstacles' dictionary adding their bounding box location inside a list; 
        furthermore, for each obstacle add the projections of bounding box inside an another list.

        return:
            obstacles: a list containing the obstacles
            future_obstacles: a list containg the future_obstacles (projections of bounding box)
        """
        # Creates two lists to fill with the bounding boxes of obstacles, actual and future
        obstacles = []
        future_obstacles = []

        for id, obs in self._obstacles.items():
            # Adding the bounding box of the selected obstacle inside the list
            obstacles.append(obs.get_current_location())

            # For the seleted obstacle, we obtain and add the future bounding boxes inside the list
            for loc in obs.get_future_locations():
                future_obstacles.append(loc)

        return obstacles, future_obstacles
         

                