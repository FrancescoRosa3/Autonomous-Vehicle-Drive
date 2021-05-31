import main
import numpy as np
from math import cos, sin, pi

class Obstacle:

    def __init__(self, obstacle):
        self._obstacle = obstacle
        self._prev_state = None
        self._future_locations = []
        self._predict_future_location()

    def get_current_location(self):
        return self._curr_obs_box_pts 
    
    def get_future_locations(self):
        return self._future_locations

    def update_state(self, obstacle):
        self._prev_state = self._obstacle
        self._obstacle = obstacle
        self._predict_future_location()

    def _predict_future_location(self):

        location = self._obstacle.transform.location
        rotation = self._obstacle.transform.rotation
        dimension = self._obstacle.bounding_box.extent
        self._curr_obs_box_pts = main.obstacle_to_world(location, dimension, rotation)

        obstacle_speed = self._obstacle.forward_speed
        future_frames_to_check = 3
        # frames_update_frequency = 0.033
        frames_update_frequency = 1

        obstacle_yaw_angle = self._obstacle.transform.rotation.yaw * pi / 180
        
        # Compute space shift to get future Location in the world frame
        cpos_shift_arr = []
        v_x = obstacle_speed * cos(obstacle_yaw_angle)
        v_y = obstacle_speed * sin(obstacle_yaw_angle)
        #print(f"object yaw angle: {obstacle_yaw_angle:.2f} - speed: {obstacle_speed:.2f} - v_x: {v_x:.2f} - v_y: {v_y:.2f}")
        
        for i in range(1, future_frames_to_check+1):
            shift_x = v_x * (i * frames_update_frequency)
            shift_y = v_y * (i * frames_update_frequency)
            print(f"shift_x: {shift_x:.2f} - shift_y: {shift_y:.2f}")
            cpos_shift = np.array([
                [shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x],
                [shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y]])
            cpos_shift_arr.append(cpos_shift)

        self._future_locations = []
        for cpos_shift in cpos_shift_arr:
            future_box_pts = []
            cpos = self._box_pts_to_cpos(self._curr_obs_box_pts)
            cpos = np.add(cpos, cpos_shift)
            if self._prev_state != None:
                # Rotation of the obstacle
                prev_yaw_angle = self._prev_state.transform.rotation.yaw * pi / 180
                yaw_difference = obstacle_yaw_angle - prev_yaw_angle
                rotyaw = np.array([
                        [np.cos(yaw_difference), np.sin(yaw_difference)],
                        [-np.sin(yaw_difference), np.cos(yaw_difference)]])
                
                cpos = np.matmul(rotyaw, cpos)
            for j in range(cpos.shape[1]):
                future_box_pts.append([cpos[0,j], cpos[1,j]])
            self._future_locations.append(future_box_pts)

    def _box_pts_to_cpos(self, box_pts):
    
        cpos = [[], []]
        for elem in box_pts:
            cpos[0].append(elem[0])
            cpos[1].append(elem[1])
            
        cpos = np.array(cpos)

        return cpos