import numpy as np

CAR_LENGHT = 2

class ObstaclesManager:

    def __init__(self):
        #self.obstacles = []
        pass


    def get_om_state(self, image=None, depth_img = None, semantic_img = None, measurement_data, sensor_data):        
        self._set_current_frame(image, depth_img, semantic_img)
        self._set_measurement_data_frame(measurement_data, sensor_data)
        return self._update_obstacles()

    def _set_current_frame(self, image, depth_img, semantic_img):
        self.curr_img = image
        self.curr_depth_img = depth_img
        self.curr_semantic_img = semantic_img

    def _set_measurement_data_frame(measurement_data, sensor_data):
        self.measurement_data = measurement_data
        self.sensor_data = sensor_data

    def _update_obstacles(self):
        
        x   = self.measurement_data.player_measurements.transform.location.x
        y   = self.measurement_data.player_measurements.transform.location.y
        z   = self.measurement_data.player_measurements.transform.location.z
        
        obstacles = []
        
        for agent in self.measurement_data.non_player_agents:
            if agent.HasField('vehicle'):
                location = agent.vehicle.transform.location
                dist = sqrt((x - location.x)**2 + (y - location.y)**2 + (z - location.z)**2) 
                if dist < 15 - CAR_LENGHT * 2:
                    
                    orientation = agent.vehicle.transform.orientation
                    dimension = agent.vehicle.bounding_box.extent

                    box_pts = obstacle_to_world(location,  dimension, orientation)
                    obstacles.append(box_pts)

            #if agent.HasField('pedestrian'):
            #    location = agent.pedestrian.transform.location

        return obstacles

    # Transform the obstacle with its boundary point in the global frame
    def _obstacle_to_world(location, dimensions, orientation):
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