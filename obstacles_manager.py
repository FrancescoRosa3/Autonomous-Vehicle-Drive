import numpy as np
from math import pi, sqrt, cos, sin, inf

import main 

CAR_LONG_SIDE = 2
#VEHICLE_LABEL = 10
#PEDESTRIAN_LABEL = 4

class ObstaclesManager:

    def __init__(self, lead_vehicle_lookahead, vehicle_obstacle_lookahead, pedestrian_obstacle_lookahead):
        self._lead_vehicle_lookahead = lead_vehicle_lookahead
        self._vehicle_obstacle_lookahead =  vehicle_obstacle_lookahead
        self._pedestrian_obstacle_lookahead = pedestrian_obstacle_lookahead

    def get_om_state(self, measurement_data, ego_pose, semantic_img=None):        
        #self._set_current_frame(semantic_img)
        self._set_measurement_data_frame(measurement_data)
        self._set_ego_location(ego_pose)

        obstacles = []
        #if self.semantic_img is not None:
        vehicles, lead_vehicle = self._update_vehicle()
        pedestrian = self._update_pedestrian()
        obstacles = vehicles + pedestrian
            
        return obstacles, lead_vehicle

    """
    def _set_current_frame(self, semantic_img):
        self.semantic_img = semantic_img
    """

    def _set_measurement_data_frame(self, measurement_data):
        self.measurement_data = measurement_data


    def _set_ego_location(self, ego_pose):
        self._ego_pose = ego_pose
        

    """
    def _check_instance_vehicle(self):
        return any(VEHICLE_LABEL in px for px in self.semantic_img)
            
    """     

    """
    def _check_instance_pedestrian(self):
        return any(PEDESTRIAN_LABEL in px for px in self.semantic_img)

    """


    def _update_vehicle(self):
        obstacles = []
        #if self._check_instance_vehicle():
        print("Update vehicle")
        
        lead_vehicle_dist = inf
        lead_vehicle = None
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('vehicle'):
                location = agent.vehicle.transform.location
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2)
                if dist < self._vehicle_obstacle_lookahead - CAR_LONG_SIDE:
                    if self._check_for_lead_vehicle(location):
                        if dist < lead_vehicle_dist:
                            lead_vehicle = agent.vehicle 
                    print("Vehicle at distance:" + str(dist))
                    rotation = agent.vehicle.transform.rotation
                    dimension = agent.vehicle.bounding_box.extent
                    box_pts = main.obstacle_to_world(location,  dimension, rotation)
                    obstacles.append(box_pts)

        print("\n")
        return obstacles, lead_vehicle

    def _update_pedestrian(self):
        obstacles = []
        #if self._check_instance_pedestrian():
        print("Update pedestrian")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('pedestrian'):
                location = agent.pedestrian.transform.location
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                if dist < self._pedestrian_obstacle_lookahead:
                    print("Pedestrian at distance:" + str(dist))
                    rotation = agent.pedestrian.transform.rotation
                    dimension = agent.pedestrian.bounding_box.extent
                    box_pts = main.obstacle_to_world(location,  dimension, rotation)
                    obstacles.append(box_pts)
        print("\n")
        return obstacles

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def _check_for_lead_vehicle(self, lead_cars_pose):
        
        lead_car_delta_vector = [lead_cars_pose[0] - self._ego_pose[0], lead_cars_pose[1] - self._ego_pose[1]]
        lead_car_distance = np.linalg.norm(lead_car_delta_vector)
        
        if lead_car_distance > self._lead_vehicle_lookahead:
            return False

        lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                            lead_car_distance)
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        
        # Check to see if the relative angle between the lead vehicle and the ego
        # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
        if np.dot(lead_car_delta_vector, ego_heading_vector) < (1 / sqrt(2)):
            return False

        return True