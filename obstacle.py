import main
import numpy as np
import math
from math import cos, sin, pi

# constant for indicating the type of agent
VEHICLE = 0

# number of previous state  to store
HISTORY_SIZE = 20

# number of frames in the future for which to make prediction 
PEDESTRIANS_FRAMES_TO_CHECK = 75
VEHICLES_FRAMES_TO_CHECK = 75



class Obstacle:

    def __init__(self, obstacle, agent_type):
        """
        The constructor takes in input the agent (vehcle, pedestrian), the agent type 
        (constante : VEHICLE->0, PEDESTRIAN->1) and instantiates the list of previuous 
        states used as a circular queue, of fixed length, the list of future location for 
        bounding boxes, and some other useful variables for detrmining future predictions. 
        Finally, the prediction for the future bounding boxes is started. 
        
        args:
            obstacle: the non player agent (vehicle, pedestrian)
            agent_type: constant indicating the type of non player agent (VEHICLE->0, PEDESTRIAN->1) 
            
        variables to set:
            self._obstacle: the non player agent (vehicle, pedestrian)
            self._agent_type: constant indicating the type of non player agent (VEHICLE->0, PEDESTRIAN->1)
            self._prev_state: instance of the list of previous states, created with None values and with a 
                size of HYSTORY_SIZE. This is used as a circular queue
            self._head: index indicating the head of queue
            self._tail: index indicating the tail of queue
            self._future_locations: the lis of future location of bounding boxes for the obsacle
        """  
        self._obstacle = obstacle
        self._agent_type = agent_type
        self._prev_state = [None] * HISTORY_SIZE
        self._head = 0
        self._tail = 0
        self._future_locations = []
        self._predict_future_location()
    

    def get_current_location(self):
        """
        Get the current bounding box for the obstacle
                
        returns:
            self._curr_obs_box_pts: the current bounding box  
        """        
        return self._curr_obs_box_pts 
    

    def get_future_locations(self):
        """
        Get the the predicted bounding box for the obstacle
                
        returns:
            self._future_locations: list of predicted bounding box  
        """  
        return self._future_locations


    def update_state(self, obstacle):
        """
        Updates the queue filling the tail index of the list with the non player agent stored 
        in the previuos call of the method, updates this index and, eventually, the head index. 
        Finally, the prediction for the future bounding boxes is started.

        args:
            obstacle: the non player agent (vehicle, pedestrian)

        """

        # fill the tail index of queue with a non playe agent
        self._prev_state[self._tail] = self._obstacle
        # update the tail index
        self._tail = (self._tail + 1) % HISTORY_SIZE

        # update the head index
        if(self._prev_state[HISTORY_SIZE - 1] != None):
            self._head = (self._tail)%HISTORY_SIZE

        # store the current non pleayer agent (vehicle, pedestrian)
        self._obstacle = obstacle

        # starting prediction
        self._predict_future_location()


    def _rotate(self, origin, point, angle):
        """
        Rotate a point counter clockwise by a given angle around a given origin.
        The angle should be given in radians.

        args:
            origin: x and y coordinates of the point around which to rotate
            point: x anf y coordinates of the point that must be rotate
            angle: the angle indicating the rotation to be made (radians)
        
        returns:
            qx: the x coordinate of the point after the rotation around the origin 
            qy: the y coordinate of the point after the rotation around the origin
        """
        ox, oy = origin
        px, py = point

        # rotation around the origin point in the frame right handed
        dx = px - ox
        dy = py - oy
        qx = math.cos(angle) * (dx) - math.sin(angle) * (dy) + ox
        qy = math.sin(angle) * (dx) + math.cos(angle) * (dy) + oy
        return qx,  qy


    def _compute_rotation(self, obstacle_yaw_angle):
        """
        Compute the rotation for the future prediction of the obstacle. The rotation is given froma a
        difference of angles between the current obstacle yaw angle and the oldest or last one. The 
        largest difference in absolute value is returned. An angular offset is considered. 

        args:
            obstacle_yaw_angle: the yaw angle of the current non playe agent. 
            
        
        returns:
            yaw_diff_head: is returned if larger than yaw_diff_tail plus an angular offset 
            yaw_diff_tail: is returned if yaw_diff_head is smaller than yaw_diff_tail plus an offset

        """
        # compute the angle difference between the current state and the oldest one
        prev_yaw_angle = self._prev_state[self._head].transform.rotation.yaw * pi / 180
        yaw_diff_head = round((obstacle_yaw_angle - prev_yaw_angle), 2)

        # compute the yaw difference with respect to the latest frame
        yaw_tail = self._prev_state[self._tail-1].transform.rotation.yaw * pi / 180
        yaw_diff_tail = round((obstacle_yaw_angle - yaw_tail),2)

        if abs(yaw_diff_head) > abs(yaw_diff_tail) + math.radians(10):
            return yaw_diff_head
        else:
            return yaw_diff_tail
        

    def _predict_future_location(self):
        """
        Compute the rotation for the future prediction of the obstacle. The rotation is given froma a
        difference of angles between the current obstacle yaw angle and the oldest or latest one. The 
        largest difference in absolute value is returned. An angular offset is considered. 

        args:
            obstacle_yaw_angle: the yaw angle of the current non playe agent. 
            
        
        returns:
            yaw_diff_head: is returned if larger than yaw_diff_tail plus an angular offset 
            yaw_diff_tail: is returned if yaw_diff_head is smaller than yaw_diff_tail plus an offset

        """
        # obtain some useful information from the current non player agent
        location = self._obstacle.transform.location
        rotation = self._obstacle.transform.rotation
        dimension = self._obstacle.bounding_box.extent

        # obtain the bounding box projection in the world frame of the non player agent starting from its bounding box dimensions
        self._curr_obs_box_pts = main.obstacle_to_world(location, dimension, rotation)
        
        # obtain the forward speed of the obstacle 
        obstacle_speed = self._obstacle.forward_speed
        
        # select the number of  future frames to check, determined by the type of agent 
        future_frames_to_check = VEHICLES_FRAMES_TO_CHECK if self._agent_type == VEHICLE else PEDESTRIANS_FRAMES_TO_CHECK
        
        # time used to make prediction between frames 
        frames_update_frequency = 0.033

        obstacle_yaw_angle = self._obstacle.transform.rotation.yaw * pi / 180
        
        # Compute space shift to get future location in the world frame starting from speed
        v_x = obstacle_speed * cos(obstacle_yaw_angle)
        v_y = obstacle_speed * sin(obstacle_yaw_angle)
        shift_x = v_x * frames_update_frequency
        shift_y = v_y * frames_update_frequency

        # the matrix of shift along x axes and y axes in the world frame
        cpos_shift = np.array([
            [shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x, shift_x],
            [shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y, shift_y]])
        
        self._future_locations = []
        cpos = self._box_pts_to_cpos(self._curr_obs_box_pts)

        # predictions are made after step frames
        step = 15

        for k in range(1, future_frames_to_check+1, step):
            future_box_pts = []

            # shift for step frames
            temp_cpos_shift = cpos_shift * step 
            cpos_trans = np.add(cpos, temp_cpos_shift)
            
            # only for vehicle compute the rotation 
            if self._prev_state[self._head] != None and self._agent_type == VEHICLE:

                # difference between the current non player agent yaw angle and the oldest or latest one
                yaw_diff = self._compute_rotation(obstacle_yaw_angle)
                
                for i in range(0, cpos.shape[1]):
                    # realize a rotation around the current bounding box point (origin) of the translated one (poin to rotate) of -yaw angle
                    x_rot, y_rot = self._rotate( [-cpos[0][i],  cpos[1][i]], [-cpos_trans[0][i], cpos_trans[1][i]], -yaw_diff)
                    cpos[0][i] = -x_rot  
                    cpos[1][i] = y_rot
            
            # only for pedestrian
            else:
                cpos = cpos_trans

            # append the bounding points in the list
            for j in range(cpos.shape[1]):
                future_box_pts.append([cpos[0,j], cpos[1,j]])
            
            # update the list of predicted bounding boxex
            self._future_locations.append(future_box_pts)


    def _box_pts_to_cpos(self, box_pts):
        """
        Adding the bounding box in the world frame of a non player agent into a matrix

        args:
            box_pts: bounding box points of non player agent in the world frame 
        
        returns:
            cpos: matric containig the x and y coordinates of points of the non player agent in the world frame
        """

        cpos = [[], []]
        for elem in box_pts:
            cpos[0].append(elem[0])
            cpos[1].append(elem[1])
            
        cpos = np.array(cpos)
        
        return cpos