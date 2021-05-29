import numpy as np
from math import pi, sqrt, cos, sin, inf, degrees

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
        #print("Update vehicle")
        
        lead_vehicle_dist = inf
        lead_vehicle = None
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('vehicle'):
                location = agent.vehicle.transform.location
                distance, orientation = self._compute_distance_orientation_from_vehicle(location)
                # the vehicle is inside the obstacle range
                if distance < self._vehicle_obstacle_lookahead - CAR_LONG_SIDE:
                    # check for vehicle on the same lane
                    if self._check_for_vehicle_on_same_lane_ahead(orientation):
                        # check if the vehicle on the same lane is a lead vehicle
                        if self._check_for_lead_vehicle(distance):
                            if distance < lead_vehicle_dist:
                                lead_vehicle_dist = distance
                                lead_vehicle = agent.vehicle 
                    else:
                        # the vehicle is not in the same lane.
                        # It is added as obstacle
                        #print("Vehicle at distance:" + str(distance))
                        rotation = agent.vehicle.transform.rotation
                        dimension = agent.vehicle.bounding_box.extent
                        box_pts = main.obstacle_to_world(location,  dimension, rotation)
                        obstacles.append(box_pts)

        return obstacles, lead_vehicle

    def _update_pedestrian(self):
        obstacles = []
        #if self._check_instance_pedestrian():
        #print("Update pedestrian")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('pedestrian'):
                location = agent.pedestrian.transform.location
                dist = sqrt((self._ego_pose[0] - location.x)**2 + (self._ego_pose[1] - location.y)**2) 
                if dist < self._pedestrian_obstacle_lookahead:
                    #print("Pedestrian at distance:" + str(dist))
                    rotation = agent.pedestrian.transform.rotation
                    dimension = agent.pedestrian.bounding_box.extent
                    box_pts = main.obstacle_to_world(location,  dimension, rotation)
                    obstacles.append(box_pts)

        return obstacles

    def _compute_distance_orientation_from_vehicle(self, car_location):
        car_delta_vector = [car_location.x - self._ego_pose[0], car_location.y - self._ego_pose[1]]
        car_distance = np.linalg.norm(car_delta_vector)

        car_delta_vector = np.divide(car_delta_vector, 
                                            car_distance)
        ego_heading_vector = [cos(self._ego_pose[2]), sin(self._ego_pose[2])]
        
        # Check to see if the relative angle between the lead vehicle and the ego
        # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
        orientation = np.dot(car_delta_vector, ego_heading_vector)
        #print(F"Orientation {degrees(orientation)}")
        return car_distance, orientation

    def _check_for_vehicle_on_same_lane_ahead(self, orientation):
        return orientation > (1/sqrt(2))

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def _check_for_lead_vehicle(self, position):
        return position < self._lead_vehicle_lookahead