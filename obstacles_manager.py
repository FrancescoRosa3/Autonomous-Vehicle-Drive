import numpy as np
import math
from math import pi, sqrt, cos, sin, inf, degrees

import obstacle
from obstacle import Obstacle
import main 

# Useful for avoiding lead vehicle hooking and unhooking
LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE = 5

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

        # Initializa obstacles list
        obstacles = []

        # Look for obstacle vehicles, vehicles on the same lane as the ego vehicle and
        # the lead vehicles among non-player agents  
        on_same_lane_vehicles, lead_vehicle = self._update_vehicles()

        # Look for obstacle pedestrains among non-player agents  
        self._update_pedestrian()

        # Get the obstacles bounding boxes and the obstacle projections bounding boxes
        # in the world frame
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

        # Update of the lookahead distances by taking into account the ego vehicle extension on the x axis,
        # and another part that depends on a braking distance heuristic related with the ego vehicle current speed.
        self._lead_vehicle_lookahead = self._lead_vehicle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3
        self._vehicle_obstacle_lookahead = self._vehicle_obstacle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3
        self._pedestrian_obstacle_lookahead = self._pedestrian_obstacle_lookahead_base + main.CAR_RADII_X_EXTENT + (speed_km_h/10)*3


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
        # Initialize the list of the vehicles on the same lane
        on_same_lane_vehicles = []

        # Initialize the lead vechicle and it's distance
        lead_vehicle_dist = inf
        lead_vehicle = None
        
        # Iterate over all the non player agents
        for agent in self.measurement_data.non_player_agents:
            # Check only for vehicle's agents
            if agent.HasField('vehicle'):
                # Obtain some useful measurement about the vehicle
                location = agent.vehicle.transform.location
                rotation = agent.vehicle.transform.rotation
                dimension = agent.vehicle.bounding_box.extent

                # Compute the distance and the orientation between the player agent and the vehicle
                distance, orientation = self._compute_distance_orientation_from_vehicle(location, rotation)

                # Check if the considered vehicle is inside the vehicle obstacle range, dinamically
                # modified by taking into account the current ego vehicle speed.
                if distance < self._vehicle_obstacle_lookahead:

                    # Compute the bounding box with respect to the world frame
                    box_pts = main.obstacle_to_world(location, dimension, rotation)
                    
                    # If the vehicle is on the same lane of the ego vehicle, it is not considered an obstacle
                    if self._check_for_vehicle_on_same_lane(orientation):
                        
                        # Adding the vehicle bounding box to the list of vehicles on the same lane of the ego vehicle. 
                        on_same_lane_vehicles.append(box_pts)

                        # Try to remove the vehicle from the obstacle dictionary. This is needed to consider the vehicles
                        # that where previously obstacles and then entered the ego vehicle lane.
                        try:
                            self._obstacles.pop(agent.id)
                        except KeyError:
                            pass

                        # Check if the vehicle on the same lane is the lead vehicle.
                        # This is done by checking if it is in front of the vehicle and if it is the nearest one.
                        if self._check_for_lead_vehicle(location):
                            if distance < lead_vehicle_dist:
                                lead_vehicle_dist = distance
                                lead_vehicle = agent.vehicle
                    
                    # If it is not a vehicle on the same lane of the ego vehicle add it to the obstacle dictionary.
                    else:
                        self._add_obstacle(agent.vehicle, agent.id, obstacle.VEHICLE)
                
                # If the vehicle is outside the obstacle range, try to eliminate it from the dictionary of all obstacles.
                # This is needed to take into account previous obstacles which went out of range.
                else:
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        pass
        
        return on_same_lane_vehicles, lead_vehicle


    def _update_pedestrian(self):
        """
        Iterates on the list of non_player_agents finding pedestrians and check if a pedestrian can be
        considered as an obstacle, if it falls in the range of the _pedestrian_obstacle_lookahead distance.
        """   

        # Iterate over all the non player agents
        for agent in self.measurement_data.non_player_agents:
            # Check only for pedestrian's agents
            if agent.HasField('pedestrian'):

                # Obtain the location of the pedestrian in the world frame
                location = agent.pedestrian.transform.location

                # Compute the Euclidean distace between the ego vehicle and the pedestrian
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                
                # If the pedestrian is within the lookahead range add it to obstacle dictionary.
                if dist < self._pedestrian_obstacle_lookahead:
                    
                    # Add or update the agent inside the obstacle dictionary using its id as key
                    self._add_obstacle(agent.pedestrian, agent.id, obstacle.PEDESTRIAN)
                
                # If the pedestrian is outside the obstacle range, try to eliminate it from the dictionary of all obstacles
                else:
                    try:
                        self._obstacles.pop(agent.id)
                    except KeyError:
                        pass


    def _compute_distance_orientation_from_vehicle(self, car_location, car_rotation):
        """
        Compute the distance and the cosine of the angle between the player agent direction versor
        and the obstacle direction versor.
        
        args:
            car_location: the position (x,y,z) of the vehicle respect to world frame
            car_rotation: the orientation (yaw, pitch, roll) of the vehicle respect to the world frame
        
        returns:
            car_distance: distance between the player agent and the obstacle.
            dot_product: cosine of the angle between the player agent direction
                versor and the obstacle direction versor.
        """
        # Compute the distance between the player agent and the obstacle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)

        # Compute the direction versor of the ego vehicle based on its yaw angle
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]

        # Compute the direction versor of the other vehicle based on its yaw angle
        car_heading_vector = [cos(math.radians(car_rotation.yaw)), sin(math.radians(car_rotation.yaw))]

        # Compute the dot product of the two versors to obtain the cosine of the angle between the versors
        dot_product = np.dot(car_heading_vector, ego_heading_vector)
        
        return car_distance, dot_product


    def _check_for_vehicle_on_same_lane(self, orientation):
        """
        Check if the cosine of the angle between the player agent direction versor and the obstacle direction versor 
        is greater than the cosine of 45°.
        
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
        # Compute the distance between the ego vehicle and the obstacle
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)
        
        # Compute the versor of the distance vector
        lead_car_delta_vector = np.divide(car_delta_vector, car_distance)

        # Compute the direction versor of the ego vehicle based on its yaw angle
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        
        # Obtain the cosine of the angle between the car heading versor and the versor of the
        # distance vector from the obstacle
        dot_product = np.dot(lead_car_delta_vector, ego_heading_vector)

        # If the ego vehicle is already following a lead vehicle add a buffer to the lookahead
        # range to prevent oscillations and, therefore lead vehicle hooking and unhooking.
        distance_from_lead = self._lead_vehicle_lookahead if not self._bp._follow_lead_vehicle else  self._lead_vehicle_lookahead + LEAD_VEHICLE_LOOKAHEAD_OFFSET_FOR_RELEASE

        # The vehicle is considered a lead vehicle if the distance is within the lead vehicle 
        # lookahead range and the cosine of the angle between the ego vehicle heading versor and
        # the distance versor is grater then the cosine of 45 degrees.
        return (car_distance < distance_from_lead) and (dot_product > cos(math.radians(45)))
    

    def _add_obstacle(self, obstacle, id, agent_type):
        """
        Create a new instance of Obstacle if it is not already present inside the obstacles dictionary;
        otherwise updates the information of the already existing instance.

        args:
            obstacle: the instance of the non_player_agents that must be entered or updated inside the obstacles dictionary.
            id: unique identifier of the non_player_agents
            agent_type: VEHICLE (0) or PEDESTRIAN (1)
        """

        # If the id is already used as key inside the dictionary, its state is only updated.
        if id in self._obstacles:
            self._obstacles[id].update_state(obstacle)

        # If the obstacle is not present in dthe dictionary, add it.
        else:
            self._obstacles[id] = Obstacle(obstacle, agent_type)


    def _get_obstacles(self):
        """
        Iterates on obstacles' dictionary adding their bounding box location inside a list; 
        furthermore, for each obstacle add the bounding box of the projections inside another list.

        return:
            obstacles: a list containing the obstacles
            future_obstacles: a list containg the future_obstacles (projections of bounding box)
        """
        
        # Initialize the bounding boxes lists for obstacles and projections
        obstacles = []
        future_obstacles = []

        for id, obs in self._obstacles.items():
            # Add the bounding box of the selected obstacle inside the list
            obstacles.append(obs.get_current_location())

            # Given the seleted obstacle, add each of its future projection inside the relative lists
            for loc in obs.get_future_locations():
                future_obstacles.append(loc)

        return obstacles, future_obstacles
         

                