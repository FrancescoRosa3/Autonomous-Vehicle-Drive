import main
import numpy as np
import math
from math import cos, sin, pi

UPDATE_FREQUENCY = 3

class Obstacle:

    def __init__(self, obstacle):
        self._obstacle = obstacle
        self._prev_state = None
        self._turn_state = None
        self._future_locations = []
        self._predict_future_location()
        self._update_frequency = UPDATE_FREQUENCY
    
    def get_current_location(self):
        return self._curr_obs_box_pts 
    
    def get_future_locations(self):
        return self._future_locations

    def update_state(self, obstacle):
        if self._update_frequency == 0:
            self._prev_state = self._obstacle
            self._update_frequency = UPDATE_FREQUENCY
        else:
            self._update_frequency -=1
            
        self._obstacle = obstacle
        self._predict_future_location()

    import math

    def rotate(self, origin, point, angle):
        """
        Rotate a point counter clockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        # rotation around the origin point in the frame right handed
        dx = px - ox
        dy = py - oy
        qx = math.cos(angle) * (dx) - math.sin(angle) * (dy) + ox
        qy = math.sin(angle) * (dx) + math.cos(angle) * (dy) + oy
        return qx,  qy


    def _predict_future_location(self):

        location = self._obstacle.transform.location
        rotation = self._obstacle.transform.rotation
        dimension = self._obstacle.bounding_box.extent
        self._curr_obs_box_pts = main.obstacle_to_world(location, dimension, rotation)

        obstacle_speed = self._obstacle.forward_speed
        future_frames_to_check = 45
        frames_update_frequency = 0.033
        # frames_update_frequency = 1

        obstacle_yaw_angle = self._obstacle.transform.rotation.yaw * pi / 180
        
        # Compute space shift to get future Location in the world frame
        v_x = obstacle_speed * cos(obstacle_yaw_angle)
        v_y = obstacle_speed * sin(obstacle_yaw_angle)
        #print(f"object yaw angle: {obstacle_yaw_angle:.2f} - speed: {obstacle_speed:.2f} - v_x: {v_x:.2f} - v_y: {v_y:.2f}")
        
        shift_x = v_x * frames_update_frequency
        shift_y = v_y * frames_update_frequency
        #print(f"shift_x: {shift_x:.2f} - shift_y: {shift_y:.2f}")
        cpos_shift = np.array([
            [shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x],
            [shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y]])
        
        self._future_locations = []
        cpos = self._box_pts_to_cpos(self._curr_obs_box_pts)
        step = 15
        for k in range(1, future_frames_to_check+1, step):
            future_box_pts = []
            temp_cpos_shift = cpos_shift * step 
            cpos_trans = np.add(cpos, temp_cpos_shift)
            
            if self._prev_state != None:
                # Rotation of the obstacle
                prev_yaw_angle = self._prev_state.transform.rotation.yaw * pi / 180
                yaw_diff = (obstacle_yaw_angle - prev_yaw_angle)
                #print(f"Prev yaw {prev_yaw_angle} Current yaw {obstacle_yaw_angle}")
                for i in range(0, cpos.shape[1]):
                    x_rot, y_rot = self.rotate( [-cpos[0][i],  cpos[1][i]], [-cpos_trans[0][i], cpos_trans[1][i]], -yaw_diff)
                    cpos[0][i] = -x_rot  
                    cpos[1][i] = y_rot

            for j in range(cpos.shape[1]):
                future_box_pts.append([cpos[0,j], cpos[1,j]])
            self._future_locations.append(future_box_pts)

        #if self._prev_state != None:
            #print(f"prev speed: {self._prev_state.forward_speed} - curr speed: {obstacle_speed}")

    def _box_pts_to_cpos(self, box_pts):
    
        cpos = [[], []]
        for elem in box_pts:
            cpos[0].append(elem[0])
            cpos[1].append(elem[1])
            
        cpos = np.array(cpos)
        
        return cpos