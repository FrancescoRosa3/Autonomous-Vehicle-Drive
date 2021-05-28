import numpy as np
from math import pi, sqrt

CAR_LONG_SIDE = 2
DISTANCE_VEHICLE = 15
DISTANCE_PEDESTRIAN = 10
#VEHICLE_LABEL = 10
#PEDESTRIAN_LABEL = 4

class ObstaclesManager:

    def __init__(self):
        pass


    def get_om_state(self, semantic_img=None, measurement_data=None):        
        #self._set_current_frame(semantic_img)
        self._set_measurement_data_frame(measurement_data)
        self._set_ego_location()

        obstacles = []
        #if self.semantic_img is not None:
        vehicles = self._update_vehicle()
        pedestrian = self._update_pedestrian()
        obstacles = vehicles + pedestrian
            
        return obstacles

    """
    def _set_current_frame(self, semantic_img):
        self.semantic_img = semantic_img
    """

    def _set_measurement_data_frame(self, measurement_data):
        self.measurement_data = measurement_data


    def _set_ego_location(self):
        self.x = self.measurement_data.player_measurements.transform.location.x
        self.y = self.measurement_data.player_measurements.transform.location.y
        self.z = self.measurement_data.player_measurements.transform.location.z
        

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
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('vehicle'):
                location = agent.vehicle.transform.location
                dist = sqrt((self.x - location.x)**2 + (self.y - location.y)**2 + (self.z - location.z)**2)
                if dist < DISTANCE_VEHICLE - CAR_LONG_SIDE * 2:
                    print("Vehicle at distance:" + str(dist))
                    rotation = agent.vehicle.transform.rotation
                    dimension = agent.vehicle.bounding_box.extent
                    box_pts = self._obstacle_to_world(location,  dimension, rotation)
                    obstacles.append(box_pts)

        print("\n")
        return obstacles


    def _update_pedestrian(self):
        obstacles = []
        #if self._check_instance_pedestrian():
        print("Update pedestrian")
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('pedestrian'):
                location = agent.pedestrian.transform.location
                dist = sqrt((self.x - location.x)**2 + (self.y - location.y)**2 + (self.z - location.z)**2) 
                if dist < DISTANCE_PEDESTRIAN:
                    print("Pedestrian at distance:" + str(dist))
                    rotation = agent.pedestrian.transform.rotation
                    dimension = agent.pedestrian.bounding_box.extent
                    box_pts = self._obstacle_to_world(location,  dimension, rotation)
                    obstacles.append(box_pts)

        print("\n")
        return obstacles


    # Transform the obstacle with its boundary point in the global frame
    def _obstacle_to_world(self, location, dimensions, orientation):
        box_pts = []

        x = location.x
        y = location.y
        z = location.z

        yaw = orientation.yaw * pi / 180

        xrad = dimensions.x
        yrad = dimensions.y
        zrad = dimensions.z

        # Border points in the obstacle frame
        cpos = np.array([
                [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
                [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
        
        # Rotation of the obstacle
        rotyaw = np.array([
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])
        
        # Location of the obstacle in the world frame
        cpos_shift = np.array([
                [x, x, x, x, x, x, x, x],
                [y, y, y, y, y, y, y, y]])
        
        cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

        for j in range(cpos.shape[1]):
            box_pts.append([cpos[0,j], cpos[1,j]])
        
        return box_pts