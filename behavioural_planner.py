#!/usr/bin/env python3
from os import close
import numpy as np
import math
from numpy import ma
from numpy.core.defchararray import index

from numpy.core.numeric import ones

from traffic_lights_manager import GO, STOP, UNKNOWN
import main

# State machine states
FOLLOW_LANE = 0
STOP_AT_TRAFFIC_LIGHT = 1
STOP_AT_OBSTACLE = 2
APPROACHING_RED_TRAFFIC_LIGHT = 3

# Stop speed threshold
STOP_THRESHOLD = 0.05
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10

# Stop traffic light threshold
STOP_TRAFFIC_LIGHT = 5
SLOW_DOWN_TRAFFIC_LIGHT = 15
TRAFFIC_LIGHT_SECURE_DISTANCE = 2

class BehaviouralPlanner:
    def __init__(self, lookahead):
        self._lookahead                     = lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._lookahead_collision_index     = 0

        # New parameters
        ### traffic light state
        self._traffic_light_state         = UNKNOWN
        ### traffic light distance
        self._traffic_light_distance        = None
        self._traffic_light_vehicle_frame = []
        ### closest index
        self._closest_index = 0


    def set_lookahead(self, lookahead):
        """
        Set waypoints lookahead.
        
        args:
            lookahead: waypoints lookahead.
        """
        self._lookahead = lookahead

    def get_state(self):
        return self._state

    def set_traffic_light_state(self, state):
        """
        Set traffic light state.
        
        args:
            state: traffic light state.
        """
        self._traffic_light_state = state

    ### [ADDITION]
    def set_obstacle_on_lane(self, collision_check_array):
        """
        Notify to the Behavioraul Planner if there are any obstacle on lane.
        
        args:
            collision_check_array: Array containing a Boolean which indicate
                if a path collide with an obstacle or not.
        """
        
        # If the collision check array length is greater than 0, procede.
        if len(collision_check_array)>0:
            # Check if the Local Planner generated ad least one free path.
            # If so, set the _obstacle_on_lane variable to False.
            # Otherwise conclude that no paths are free and set the
            # _obstacle_on_lane variable to true.
            for path_result in collision_check_array:
                if path_result:
                    self._obstacle_on_lane = False
                    return
                self._obstacle_on_lane = True
        else:
            self._obstacle_on_lane = False

    def set_traffic_light_distance(self, distance):
        """
        Set traffic light distance.
        
        args:
            distance: distance between the ego-vehicle and the traffic light
                in the Vehicle Frame along the x axis.
        """
        self._traffic_light_distance = distance

    def set_traffic_light_vehicle_frame(self, traffic_light_vehicle_frame):
        """
        Set traffic light points position with respect to Vehicle Frame.
        
        args:
            traffic_light_vehicle_frame: points position with respect to Vehicle Frame.
        """
        self._traffic_light_vehicle_frame = traffic_light_vehicle_frame

    def set_follow_lead_vehicle(self, following_lead_vehicle):
        """
        Set following_lead_vehicle variable to indicate if a lead vehicle exists.
        
        args:
            following_lead_vehicle: Boolean which represent the existance of a lead vehicle.
        """
        self._follow_lead_vehicle = following_lead_vehicle

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    STOP_AT_TRAFFIC_LIGHT  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        
        # Compute the conversion from speed in m/s to km/h.
        speed_km_h = (ego_state[3]*3600)/1000

        # Compute the heuristic for the breaking distance.
        secure_distance_brake = (speed_km_h/10)*3
        
        
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any obstacle or red traffic light.
        # In the first case, enforce the car to stop immediately.
        # In the second case, check the distance to the traffic light
        # and slow down if it is between the first and the secondo threshold,
        # or enforce the car to stop if it is under the second thresold.
        if self._state == FOLLOW_LANE:
            print("FSM STATE: FOLLOW_LANE")

            self._update_goal_index(waypoints, ego_state)
             
            if self._obstacle_on_lane:
                self._goal_state[2] = 0
                self._state = STOP_AT_OBSTACLE
            else:
                if self._traffic_light_state == STOP and self._traffic_light_distance != None:
                    if self._traffic_light_distance < (STOP_TRAFFIC_LIGHT + secure_distance_brake):
                        self._goal_state[2] = 0
                        self._state = STOP_AT_TRAFFIC_LIGHT
                    elif self._traffic_light_distance < (SLOW_DOWN_TRAFFIC_LIGHT + secure_distance_brake) :
                        self._goal_state[2] = main.HALF_CRUISE_SPEED
                        self._state = APPROACHING_RED_TRAFFIC_LIGHT

        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Procede at half cruise speed.
        # If an obstacle is detected, enforce the car to stop.
        # If the traffic light state is red but above the first threshold
        # keep the same speed. If the distance drops below the threshold
        # ensure that the goal state enforce the car to be stopped before
        # the traffic light line.
        elif self._state == APPROACHING_RED_TRAFFIC_LIGHT:
            print("FSM STATE: APPROACHING_RED_TRAFFIC_LIGHT")
            
            self._update_goal_index_with_traffic_light(waypoints, ego_state)
            self._goal_state[2] = main.HALF_CRUISE_SPEED

            if self._obstacle_on_lane:
                self._goal_state[2] = 0
                self._state = STOP_AT_OBSTACLE
            else:
                if self._traffic_light_state == STOP:
                    if self._traffic_light_distance != None and self._traffic_light_distance < (STOP_TRAFFIC_LIGHT+secure_distance_brake):
                            self._goal_state[2] = 0
                            self._state = STOP_AT_TRAFFIC_LIGHT
                elif self._traffic_light_state == GO:
                    self._state = FOLLOW_LANE
                                
            
        # In this state, the car is stopped at traffic light.
        # Transit to the the "FOLLOW_LANE" state if the traffic light becomes green
        # and there are no obstacles on lane.
        # If an obstacle happens to be on the lane, transit to "STOP_AT_OBSTACLE" state
        # enforcing the car to stay stopped.
        elif self._state == STOP_AT_TRAFFIC_LIGHT:
            print("FSM STATE: STOP_AT_TRAFFIC_LIGHT")
            if closed_loop_speed > STOP_THRESHOLD:
                self._update_goal_index_with_traffic_light(waypoints, ego_state)
            
            self._goal_state[2] = 0
            if self._obstacle_on_lane:
                self._state = STOP_AT_OBSTACLE
            elif self._traffic_light_state == GO:
                self._update_goal_index(waypoints, ego_state)
                self._state = FOLLOW_LANE


        # In this state, the car is stopped before an obstacle.
        # Transit to the the "FOLLOW_LANE" state if there are no obstacles on lane
        # and the traffic light is not red or if there are no obstacles and the
        # traffic light is red but the distance is above the greater threshold.
        # If there are no obstacles on lane but the traffic light state is red and the
        # distance is between the two threshold, set the speed to HALF CRUISE SPEED and
        # transit to "APPROACHING_RED_TRAFFIC_LIGHT". Otherwise, if the traffic light
        # state is red and the distance is below the shorter threshold, enforce the vehicle
        # to stop and transit to "STOP_AT_TRAFFIC_LIGHT".
        elif self._state == STOP_AT_OBSTACLE:
            print("FSM STATE: STOP_AT_OBSTACLE")
            self._goal_state[2] = 0
            if not self._obstacle_on_lane:
                if self._traffic_light_state == STOP and self._traffic_light_distance != None:
                    if self._traffic_light_distance < (STOP_TRAFFIC_LIGHT + secure_distance_brake):
                       self._state = STOP_AT_TRAFFIC_LIGHT
                    elif self._traffic_light_distance < (SLOW_DOWN_TRAFFIC_LIGHT + secure_distance_brake):
                        self._goal_state[2] = main.HALF_CRUISE_SPEED
                        self._state = APPROACHING_RED_TRAFFIC_LIGHT
                    else:
                        self._update_goal_index(waypoints, ego_state)
                        self._state = FOLLOW_LANE
                else:
                    self._update_goal_index(waypoints, ego_state)
                    self._state = FOLLOW_LANE
           
        else:
            raise ValueError('Invalid state value.')
        
    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            
            # [MODIFIED]
            # Update the waypoint index to be returned before breaking the while cycle,
            # in order to truly return the first waypoint after the lookahead.
            wp_index += 1
            if arc_length > self._lookahead: break

        return wp_index % len(waypoints)

    def _update_goal_index(self, waypoints, ego_state):
        """Update goal index in such a way as to obtain the first waypoint after the
        lookahead distance

        args:
            waypoints: current waypoints to track. (global frame)
            ego_state: ego state vector for the vehicle. (global frame)
        """
        
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1
        self._closest_index = closest_index
        self._goal_index = goal_index
        self._goal_state = list(waypoints[goal_index])

    def _update_goal_index_with_traffic_light(self, waypoints, ego_state):
        """Update goal index in such a way as to obtain the waypoint closest to the traffic light

        args:
            waypoints: current waypoints to track. (global frame)
            ego_state: ego state vector for the vehicle. (global frame)
        """

        # Update goal index without taking into account the traffic light to both obtain the closest waypoint
        # to the ego vehicle and also in case the distance between the ego vehicle and the traffic light
        # would not be available.
        self._update_goal_index(waypoints, ego_state)   

        # if the distance between the ego vehicle and the traffic light is available start the
        # procedure to compute the waypoint nearest to the latter.
        if self._traffic_light_distance != None:

            # Initialize the waypoint 
            closest_wp_to_traffic_light = self._closest_index
            # find the closest waypoint to the traffic light
            waypoint_index = closest_wp_to_traffic_light
            
            # Initialize the distance between the waypoint and the traffic light in the world frame
            distance_wp_traffic_light = math.inf
            
            # iterate over all the waypoints after the closest one to find the nearest to the traffic light
            while waypoint_index < len(waypoints)-1:
                temp_distance_wp_traffic_light = 0
                
                # compute the average distance between the waypoint and the traffic light in the world frame
                for traffic_light_wp in self._traffic_light_vehicle_frame:
                    traffic_light_wp_world_frame = convert_wp_in_world_frame(ego_state, traffic_light_wp)
                    temp_distance_wp_traffic_light += np.sqrt((waypoints[waypoint_index][0] - traffic_light_wp_world_frame[0])**2 + (waypoints[waypoint_index][1] -traffic_light_wp_world_frame[1])**2)
                temp_distance_wp_traffic_light = temp_distance_wp_traffic_light/len(self._traffic_light_vehicle_frame)
                
                # if the avarage distance is shorter then the previous nearest waypoint, update the waypoint
                # and the shortest distance
                if temp_distance_wp_traffic_light < distance_wp_traffic_light:
                    distance_wp_traffic_light = temp_distance_wp_traffic_light
                    closest_wp_to_traffic_light =  waypoint_index
                waypoint_index += 1

            # update the goal index and the goal state based on the obtained waypoint
            self._goal_index = closest_wp_to_traffic_light
            self._goal_state = list(waypoints[closest_wp_to_traffic_light])


        
# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """

    # Initilize the closest index info
    closest_len = float('Inf')
    closest_index = 0

    # Compute the distance between the ego_vehicle position and each waypoint.
    # Convert the waypoint position to Vehicle Frame and check if its x component
    # is poitive to verify if it is in front of the ego-vehicle.
    # If so, check if it is the nearest one.
    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2

        # Convert the waypoint position to Vehicle Frame
        p_wp_vehicle = convert_wp_in_vehicle_frame(ego_state, waypoints[i])
        if(p_wp_vehicle[0] > 0):    
            if temp < closest_len:
                closest_len = temp
                closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index


def convert_wp_in_vehicle_frame(ego_state, waypoint):
    """Conversion of a waypoint position from World Frame to Vehicle Frame.

        args:
            ego_state: ego state vector for the vehicle in the World Frame.
            waypoint: waypoint position in the World Frame.
    """
    p_wp_world = np.array([waypoint[0], waypoint[1]]).T
    o_world_vehicle = np.array([ego_state[0], ego_state[1]]).T

    R_world_vehicle =  np.array([
                                [np.cos(ego_state[2]), -np.sin(ego_state[2])],
                                [np.sin(ego_state[2]), np.cos(ego_state[2])]])
    
    # p{1} = - R{0_to_1}^T * o{0_to_1} + R{0_to_1}^T * p{1}
    # p{vehicle_frame} = - R{world_to_vehicle}^T * o{world_to_vehicle} + R{world_to_vehicle}^T * p{vehicle}
    p_wp_vehicle = -np.matmul(R_world_vehicle.T, o_world_vehicle) + np.matmul(R_world_vehicle.T, p_wp_world)
    return p_wp_vehicle

def convert_wp_in_world_frame(ego_state, waypoint):
    """Conversion of a waypoint position from Vehicle Frame to World Frame.

        args:
            ego_state: ego state vector for the vehicle in the World Frame.
            waypoint: waypoint position in the Vehicle Frame.
    """
    p_wp_vehicle = np.array([waypoint[0], waypoint[1]]).T
    o_world_vehicle = np.array([ego_state[0], ego_state[1]]).T

    R_world_vehicle =  np.array([
                                [np.cos(ego_state[2]), -np.sin(ego_state[2])],
                                [np.sin(ego_state[2]), np.cos(ego_state[2])]])
    
    # p{0} = o{0_to_1} + R{0_to_1}*p{1}
    # p{world} = o{world_to_vehicle} + R{world_to_vehicle}*p{vehicle}
    p_wp_world = o_world_vehicle + np.matmul(R_world_vehicle, p_wp_vehicle)
    return p_wp_world
