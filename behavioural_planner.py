#!/usr/bin/env python3
from os import close
import numpy as np
import math

from main import CRUISE_SPEED, HALF_CRUISE_SPEED
from traffic_lights_manager import GO, STOP, UNKNOWN

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2

# New implemented states 
FOLLOW_LANE_HALF_SPEED = 3

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
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0

        ## New parameters
        ### traffic light state
        self._traffic_light_state         = UNKNOWN
        ### traffic light distance
        self._traffic_light_distance        = None
    
    def set_traffic_light_state(self, state):
        self._traffic_light_state = state

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
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
        if self._state == FOLLOW_LANE:
            print("FSM STATE: FOLLOW_LANE")
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            
            if self._traffic_light_state == STOP and self._traffic_light_distance != None:
                if self._traffic_light_distance < STOP_TRAFFIC_LIGHT:
                    self._goal_state[2] = 0
                    self._state = DECELERATE_TO_STOP
                elif self._traffic_light_distance < SLOW_DOWN_TRAFFIC_LIGHT:
                    self._goal_state[2] = HALF_CRUISE_SPEED
                    self._state = FOLLOW_LANE_HALF_SPEED

        ## New state
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Procede at half cruise speed.
        # If the traffic light state is red but above a certain threshold
        # keep the same speed. If the distance drops below the threshold
        # ensure that the goal state enforces the car to be stopped before
        # the traffic light line.
        elif self._state == FOLLOW_LANE_HALF_SPEED:
            print("FSM STATE: FOLLOW_LANE_HALF_SPEED")
            # print(abs(closed_loop_speed))

            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            self._goal_index = goal_index

            # If the traffic light is still red, check the distance.
            # If it is below the given threshold, enforces the car to be stopped
            # before the traffic light line.
            # If the traffic line is not red anymore return to lane following.
            if self._traffic_light_state == STOP:
                if self._traffic_light_distance != None and self._traffic_light_distance < STOP_TRAFFIC_LIGHT:
                        self._goal_state[2] = 0
                        self._state = DECELERATE_TO_STOP
            elif self._traffic_light_state == GO:
                self._goal_state = waypoints[goal_index]
                self._state = FOLLOW_LANE
            
        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            print("FSM STATE: DECELERATE_TO_STOP")
            # print(abs(closed_loop_speed), STOP_THRESHOLD)
            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                self._stop_count = 0

        # In this state, check to see if the traffic light is not red anymore.
        # If so, we can now leave the traffic light and transition to the next state.
        elif self._state == STAY_STOPPED:
            print("FSM STATE: STAY_STOPPED")
            # If the traffic light is no longer red, we can now
            # transition back to our lane following state.
            if self._traffic_light_state == GO:
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
        print("Check for new waypoint")
        print(F"Ego state X:{ego_state[0]} Y:{ego_state[1]}")
        while wp_index < len(waypoints) - 1:
            print(F"Waypoints X:{waypoints[wp_index][0]} Y:{waypoints[wp_index][1]}")
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            # check for turn
            if self._check_for_turn(ego_state, waypoints[wp_index]):
                print("waypoint on turn")
                break
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    def _check_for_turn(self, ego_state, closest_waypoint):
        dx = ego_state[0] - closest_waypoint[0]
        dy = ego_state[1] - closest_waypoint[1]
        print(F"Dx X:{dx} Dy:{dy}")
        if abs(dx) < 1 or abs(dy) < 1:
            return False
        else:
            return True

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
