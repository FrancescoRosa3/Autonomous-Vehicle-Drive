#!/usr/bin/env python3
from os import close
import numpy as np
import math

from main import CRUISE_SPEED, HALF_CRUISE_SPEED
from traffic_lights_manager import GO, STOP, UNKNOWN

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

class BehaviouralPlanner:
    def __init__(self, lookahead):
        self._lookahead                     = lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._lookahead_collision_index     = 0
    

        ## New parameters
        ### traffic light state
        self._traffic_light_state         = UNKNOWN
        ### traffic light distance
        self._traffic_light_distance        = None
    
    def set_traffic_light_state(self, state):
        self._traffic_light_state = state

    def set_obstacle_on_lane(self, collision_check_array):
        if len(collision_check_array)>0:
            for path_result in collision_check_array:
                if path_result:
                    self._obstacle_on_lane = False
                    return
            self._obstacle_on_lane = True
        else:
            self._obstacle_on_lane = False
    
    def set_traffic_light_distance(self, distance):
        self._traffic_light_distance = distance

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_follow_lead_vehicle(self, following_lead_vehicle):
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
        speed_km_h = (ego_state[2]*3600)/1000
        # secure_distance_brake = (speed_km_h/10)*3
        secure_distance_brake = 0
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any onstacle or red traffic light.
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
                        self._goal_state[2] = HALF_CRUISE_SPEED
                        self._state = APPROACHING_RED_TRAFFIC_LIGHT

        ## New state
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Procede at half cruise speed.
        # If an obstacle is detected, enforce the car to stop immediately.
        # If the traffic light state is red but above the first threshold
        # keep the same speed. If the distance drops below the threshold
        # ensure that the goal state enforce the car to be stopped before
        # the traffic light line.
        elif self._state == APPROACHING_RED_TRAFFIC_LIGHT:
            print("FSM STATE: APPROACHING_RED_TRAFFIC_LIGHT")

            self._update_goal_index(waypoints, ego_state)
            self._goal_state[2] = HALF_CRUISE_SPEED

            if self._obstacle_on_lane:
                self._goal_state[2] = 0
                self._state = STOP_AT_OBSTACLE
            else:
                if self._traffic_light_state == STOP:
                    if self._traffic_light_distance != None and self._traffic_light_distance < (STOP_TRAFFIC_LIGHT+secure_distance_brake):
                            self._goal_state[2] = 0
                            self._state = STOP_AT_TRAFFIC_LIGHT
                elif self._traffic_light_state == GO:
                    self._update_goal_index(waypoints, ego_state)
                    self._state = FOLLOW_LANE
                    
            
        # In this state, the car is stopped at traffic light.
        # Transit to the the "follow lane" state if the traffic light becomes green
        # and there are no obstacle on lane.
        # If an obstacle happens to be on the lane, transit to "stop ato obstacle" state
        # enforcing the car to stay stopped.
        elif self._state == STOP_AT_TRAFFIC_LIGHT:
            print("FSM STATE: STOP_AT_TRAFFIC_LIGHT")
            self._goal_state[2] = 0
            if self._obstacle_on_lane:
                self._state = STOP_AT_OBSTACLE
            elif self._traffic_light_state == GO:
                self._update_goal_index(waypoints, ego_state)
                self._state = FOLLOW_LANE


        # In this state, the car is stopped at traffic light.
        # Transit to the the "follow lane" state if the traffic light becomes green
        # and there are no obstacle on lane.
        # If an obstacle happens to be on the lane, transit to "stop ato obstacle" state
        # enforcing the car to stay stopped.
        elif self._state == STOP_AT_OBSTACLE:
            print("FSM STATE: STOP_AT_OBSTACLE")
            self._goal_state[2] = 0
            if not self._obstacle_on_lane:
                if self._traffic_light_state == STOP and self._traffic_light_distance != None:
                    if self._traffic_light_distance < (STOP_TRAFFIC_LIGHT + secure_distance_brake):
                       self._state = STOP_AT_TRAFFIC_LIGHT
                    elif self._traffic_light_distance < (SLOW_DOWN_TRAFFIC_LIGHT + secure_distance_brake) :
                        self._goal_state[2] = HALF_CRUISE_SPEED
                        self._state = APPROACHING_RED_TRAFFIC_LIGHT
                else:
                    self._update_goal_index(waypoints, ego_state)
                    self._state = FOLLOW_LANE

        else:
            raise ValueError('Invalid state value.')

        print(F"Goal state out {self._goal_state}")
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

        wp_lookahead = 1


        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if self._check_for_turn(ego_state, waypoints[wp_index]):
            #print("waypoint on turn")
            return wp_index + wp_lookahead

        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        #print("Check for new waypoint")
        #print(F"Ego state X:{ego_state[0]} Y:{ego_state[1]}")
        while wp_index < len(waypoints) - 1:
            #print(F"Waypoints X:{waypoints[wp_index][0]} Y:{waypoints[wp_index][1]}")
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            
            # check for turn
            if self._check_for_turn(ego_state, waypoints[wp_index]):
                #print("waypoint on turn")
                wp_index += wp_lookahead
                break
            
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    def _check_for_turn(self, ego_state, closest_waypoint):
        dx = ego_state[0] - closest_waypoint[0]
        dy = ego_state[1] - closest_waypoint[1]
        
        offset = 1

        #print(F"Dx X:{dx} Dy:{dy}")
        if abs(dx) < offset or abs(dy) < offset:
            return False
        else:
            return True

    def _update_goal_index(self, waypoints, ego_state):
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1:
            goal_index += 1
        self._goal_index = goal_index
        self._goal_state = waypoints[goal_index]
        
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
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
