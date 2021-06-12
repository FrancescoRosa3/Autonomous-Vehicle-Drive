#!/usr/bin/env python3
import numpy as np
import scipy.spatial
import math
from math import sin, cos, pi, sqrt
from main import CAR_RADII_X_EXTENT
from behavioural_planner import STOP_AT_OBSTACLE

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight

    ### [ADDITION] [MODIFIED]
    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free,
    # and an array indicating the distance between the vehicle and obstacles, if any.
    def collision_check(self, paths, obstacles):
        """Returns a bool array on whether each path is collision free.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]

        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            collision_dist_array: A list of float values which represents
                the distance between the ego vehicle and an obstacle, if it exists.
                The ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
 
        collision_check_array = np.zeros(len(paths), dtype=bool)
        collision_dist_array = np.zeros(len(paths), dtype=float)

        for i in range(len(paths)):
            collision_free = True
            path           = paths[i]

            # Given the car extension on the x axis as threshold, ignore all the indices below it. 
            offset_to_ignore = CAR_RADII_X_EXTENT
            dist = 0
            path_index = 0
            while dist < offset_to_ignore and path_index < (len(path[0])-1):
                temp_dist = np.sqrt((path[0][path_index+1]-path[0][path_index])**2+(path[1][path_index+1]-path[1][path_index])**2)
                dist += temp_dist
                path_index += 1

            if path_index != len(path[0])-1:
                # Iterate over the points in the path.
                dist_from_obstacle = dist
                for j in range(path_index, len(path[0])):
                    # Compute the circle locations along this point in the path.
                    # These circle represent an approximate collision
                    # border for the vehicle, which will be used to check
                    # for any potential collisions along each path with obstacles.

                    # The circle offsets are given by self._circle_offsets.
                    # The circle offsets need to placed at each point along the path,
                    # with the offset rotated by the yaw of the vehicle.
                    # Each path is of the form [[x_values], [y_values],
                    # [theta_values]], where each of x_values, y_values, and
                    # theta_values are in sequential order.

                    # Thus, we need to compute:
                    # circle_x = point_x + circle_offset*cos(yaw)
                    # circle_y = point_y circle_offset*sin(yaw)
                    # for each point along the path.
                    # point_x is given by path[0][j], and point _y is given by
                    # path[1][j]. 
                    circle_locations = np.zeros((len(self._circle_offsets), 2))

                    circle_offset = np.array(self._circle_offsets)
                    circle_locations[:, 0] = path[0][j] + circle_offset * cos(path[2][j])
                    circle_locations[:, 1] = path[1][j] + circle_offset * sin(path[2][j])

                    # Assumes each obstacle is approximated by a collection of
                    # points of the form [x, y].
                    # Here, we will iterate through the obstacle points, and check
                    # if any of the obstacle points lies within any of our circles.
                    # If so, then the path will collide with an obstacle and
                    # the collision_free flag should be set to false for this flag
                    for k in range(len(obstacles)):
                        collision_dists = \
                            scipy.spatial.distance.cdist(obstacles[k], 
                                                        circle_locations)
                        collision_dists = np.subtract(collision_dists, 
                                                    self._circle_radii)
                        collision_free = collision_free and \
                                        not np.any(collision_dists < 0)

                        ### [ADDITION] [MODIFIED]
                        # If an obstacle collides with the current path, update the distance from the obstacle.
                        if not collision_free:
                            dist_from_obstacle += np.sqrt((path[0][j]-path[0][path_index])**2+(path[1][j]-path[1][path_index])**2)
                            break
                    if not collision_free:
                        break

                collision_check_array[i] = collision_free

                # Correct the distance from the obstacle by taking into account the ego vehicle extension.
                dist_from_obstacle = self._correct_distance(dist_from_obstacle)
                collision_dist_array[i] = dist_from_obstacle
            
            # If the path is shorter than the ego vechile extension on the x axis, set all paths to free and
            # the distance from obstacles to infinite.
            else:
                collision_check_array = [True] * len(paths)
                collision_dist_array = [math.inf] * len(paths)
                
        # Set the distance from obstacle to infinite to all the free paths.
        for i in range(len(collision_check_array)):
            if collision_check_array[i]:
                collision_dist_array[i] = math.inf
        
        return collision_check_array, collision_dist_array

    ### [ADDITION] [MODIFIED]
    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline.
    # Disqualifies paths that collide with obstacles from the selection
    # process, except if the behavioraul planner is in the "STOP_AT_OBSTACLE" state.    <----- [MODIFIED]
    # In that case consider also the colliding paths in order to not run out of paths.  <----- [MODIFIED]
    # collision_check_array contains True at index i if paths[i] is
    # collision-free, otherwise it contains False.
    def select_best_path_index(self, paths, collision_check_array, goal_state, behavioural_planner_state):
        """Returns the path index which is best suited for the vehicle to
        traverse.

        Selects a path index which is closest to the center line as well as far
        away from collision paths.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
            behavioural_planner_state: behavioural planner state
        useful variables:
            self._weight: Weight that is multiplied to the best index score.
        returns:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of colliding paths.
            if behavioural_planner_state != STOP_AT_OBSTACLE and not collision_check_array[i]:
                score = float('Inf')
            
            # Handle the case of collision-free paths.
            else:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                score = np.sqrt((paths[i][0][-1]-goal_state[0])**2+(paths[i][1][-1]-goal_state[1])**2)

            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index

    ### [ADDITION] [MODIFIED]
    def _correct_distance(self, distance):
        """Corrects the distance from the obstacle by taking into account the
        ego vehicle extension on the long side.

        args:
            distance: not corrected distance between the obstacle and the ego vehicle.
        returns:
            new_distance: corrected distance.
        """

        # Computes the new distance by subtracting half of the extension of the ego
        # vehicle on the long side. 
        new_distance = distance - CAR_RADII_X_EXTENT
        new_distance = 0 if new_distance < 0 else new_distance
        return new_distance