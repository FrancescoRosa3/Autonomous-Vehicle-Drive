import os
from traffic_light_detector import TrafficLightDetector
import numpy as np
import main
from math import cos, sin, pi,tan

BASE_DIR = os.path.dirname(__file__)

# The following enumeration represent the possibile states of the traffic light.
# In particular, an additional UNKNOWN class is provided to classify the lack
# of traffic light detection. 
TRAFFIC_LIGHT_CLASSES = ["go", "stop", "UNKNOWN"]
GO = TRAFFIC_LIGHT_CLASSES[0]
STOP = TRAFFIC_LIGHT_CLASSES[1]
UNKNOWN = TRAFFIC_LIGHT_CLASSES[2]

# Numeric label of the traffic light provided by Carla documentation
TRAFFIC_SIGN_LABEL = 12

# The following constants are used to represent the consecutive number of frames
# needed to define the transition between the different traffic light states.
# In particular, if the current state is GO or UNKOWN, 3 consecutive RED classified frames
# are needed to set the traffic light state to RED. Otherwise, 4 consective frames
# are needed for the transition.
STOP_COUNTER = 4
OTHERS_COUNTER = 3


class TrafficLightsManager:

    def __init__(self, camera_parameters, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module\\config.json')):
        self.config_path = config_path
        self.tl_det = TrafficLightDetector()
        
        self.new_state_counter = 0
        self.true_state = UNKNOWN
        self.prev_state = self.true_state
        self.distance = None
        self.vehicle_frame_list = []
        self._create_intrinsic_matrix(camera_parameters)
        self._image_to_camera_frame_matrix()

    def get_tl_state(self, image, depth_img = None, semantic_img = None):
        """Updates the current frames and use them to compute the true traffic light state
           and the its distance from the ego vehicle.
        
        args:
            image: the current RGB image. 
            depth_img: the current depth image.
            semantic_img: the current semantic image.
        returns:
            self.true_state: True state of the detected traffic light.
            self.distance: Computed distance of the traffic light from the ego vehicle.
            self.vehicle_frame_list: Set of traffic light point's position with respect 
                to vehicle frame.
        """

        self._set_current_frame(image, depth_img, semantic_img)
        self._update_state()
        self._update_distance()

        return self.true_state, self.distance, self.vehicle_frame_list

    def _set_current_frame(self, image, depth_img, semantic_img):
        """Updates the current frames.
        
        args:
            image: the current RGB image. 
            depth_img: the current depth image.
            semantic_img: the current semantic image.
        """

        self.curr_img = image
        self.curr_depth_img = depth_img
        self.curr_semantic_img = semantic_img

    def _update_distance(self):
        """
            Compute the traffic light distance from the ego vehicle.
        """
        
        # compute the distance from the traffic light if its state is red and a bounding box exists
        if self.true_state == STOP and self.curr_bb != None:
            
            image_h, image_w = self.curr_depth_img.shape
            # compute the bounding box coordinates
            xmin = int(self.curr_bb.xmin * image_w)
            ymin = int(self.curr_bb.ymin * image_h)
            xmax = int(self.curr_bb.xmax * image_w)
            ymax = int(self.curr_bb.ymax * image_h)

            # get the pixel coordinates corresponding to the traffic light position in the image
            traffic_light_pixels = self._get_traffic_light_slice_from_semantic_segmentation(xmin, xmax, ymin, ymax)

            # check for false positive bounding box
            if len(traffic_light_pixels) == 0:
                self.distance = None
                return

            # get the traffic light pixels from the depth image
            x_distance = 0
            self.vehicle_frame_list = []
            for pixel in traffic_light_pixels:
                
                # convert depth image value in meters
                depth = 1000 * self.curr_depth_img[pixel[0]][pixel[1]]

                ###
                # Compute the pixel position in vehicle frame
                ###

                pixel = [pixel[1] , pixel[0], 1]
                pixel = np.reshape(pixel, (3,1))
                # Projection "Pixel to Image Frame"
                image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth
                # Creation of the extended vector
                image_frame_vect_extended = np.zeros((4,1))
                image_frame_vect_extended[:3] = image_frame_vect 
                image_frame_vect_extended[-1] = 1

                # Projection "Image to Camera Frame"
                camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
                camera_frame = camera_frame[:3]
                camera_frame = np.asarray(np.reshape(camera_frame, (1,3)))

                camera_frame_extended = np.zeros((4,1))
                camera_frame_extended[:3] = camera_frame.T 
                camera_frame_extended[-1] = 1

                # Projection "Camera to Vehicle Frame"
                camera_to_vehicle_frame = np.zeros((4,4))
                camera_to_vehicle_frame[:3,:3] = main.to_rot([self.cam_pitch, self.cam_yaw, self.cam_roll])
                camera_to_vehicle_frame[:,-1] = [self.cam_x_pos, -self.cam_y_pos, self.cam_height, 1]
                
                vehicle_frame = np.dot(camera_to_vehicle_frame,camera_frame_extended )
                vehicle_frame = vehicle_frame[:3]
                vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))
                
                # Add the computed pixel position with respect to vehicle frame to the pixel positions list
                self.vehicle_frame_list.append([vehicle_frame[0][0], -vehicle_frame[0][1]])
                                
                # Update the distance between the traffic light and the vehicle by takin into account
                # only the x axis contribute.   
                x_distance += vehicle_frame[0][0]
                
            # compute the avarage distance
            self.distance = x_distance/len(traffic_light_pixels)
            # correct the distance by taking into account the ego vehicle extension on the long side.
            self.distance = self._correct_perpendicular_distance(self.distance)
        
        # if the traffic light state is not red or its bounding box doesn't exists set the distance to None
        else:
            self.distance = None

    def _update_state(self):
        """
            Updates the true traffic light state.
        """

        # Get the traffic light bounding box from the detector given the current RGB image.
        self.curr_bb = self.tl_det.detect_on_image(self.curr_img)

        # If the bounding box doesn't exist, the current state is set to UNKNOWN.
        # Otherwise it is set to the detected state.  
        if self.curr_bb == None:
            curr_state = UNKNOWN
        else:
            curr_state = TRAFFIC_LIGHT_CLASSES[self.curr_bb.get_label()]
        
        # If the detected state is equal to the previous current state, increment a counter that
        # keeps track of the consecutive frames classified with the same state. (?) 
        if curr_state == self.prev_state:
            self.new_state_counter += 1
            
            # Given the true state, set a different threshold, for the consecutive number of frames
            # classified in the same state, to change the true state.
            threshold = STOP_COUNTER if self.true_state == "stop" else OTHERS_COUNTER
            
            # If the counter is equal or greater than the threshold, set the true state to the current state.
            if self.new_state_counter >= threshold:
                    self.true_state = self.prev_state
        
        # If the current detected state is different from the previous detected state, set the
        # counter to 0 and update the current state.
        else:
            self.new_state_counter = 0
            self.prev_state = curr_state

    def _get_traffic_light_slice_from_semantic_segmentation(self, x_min, x_max, y_min, y_max):
        """Get the pixels coordinates corresponding to the traffic light from the semantic image,
        given the detected traffic light bounding box.

        args:
            x_min, x_max, y_min, y_max: bounding box coordinates.
        returns:
            traffic_light_pixels: set of pixels coordinates corresponding to the traffic light. 
        """

        # Since the bounding box returned by the detector is not always actually centered on
        # the traffic light, you need to check it's neighborhood to find the actual traffic light pixels.
        # The neighborhood is defined by the x_offset and y_offset variables.
        width = x_max - x_min
        x_offset = int(width + ((width*30)/100))
        y_offset = 0

        # Prevent indices from going out of image borders
        start_y = 0 if y_min - y_offset < 0 else y_min-y_offset
        end_y = self.curr_semantic_img.shape[0] if (y_max + y_offset) > self.curr_semantic_img.shape[0] else y_max + y_offset
        start_x = 0 if x_min - x_offset < 0 else x_min-x_offset
        end_x = self.curr_semantic_img.shape[1] if (x_max + x_offset) > self.curr_semantic_img.shape[1] else x_max + x_offset

        # get the pixels belonging to traffic sign using the semantic image and the corresponding class. 
        traffic_light_pixels = []
        for v in range(start_y, end_y):
            for u in range(start_x, end_x):
                if self.curr_semantic_img[v][u] == TRAFFIC_SIGN_LABEL:
                    traffic_light_pixels.append((v, u))

        return traffic_light_pixels

    def _create_intrinsic_matrix(self, camera_parameters):
        """Computes the intrinsic matrix given the camera parameters.

        args:
            camera_parameters: parameters of the used camera (position, orientation, dimensions, fov). 
        """

        ### Compute the transformation matrices from image to camera frame
        self.cam_height = camera_parameters['z']
        self.cam_x_pos = camera_parameters['x']
        self.cam_y_pos = camera_parameters['y']

        self.cam_yaw = camera_parameters['yaw'] 
        self.cam_pitch = camera_parameters['pitch'] 
        self.cam_roll = camera_parameters['roll']
        
        camera_width = camera_parameters['width']
        camera_height = camera_parameters['height']

        camera_fov = camera_parameters['fov']

        # Calculate Intrinsic Matrix
        f = camera_width /(2 * tan(camera_fov * pi / 360))
        Center_X = camera_width / 2.0
        Center_Y = camera_height / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                     [0, f, Center_Y],
                                     [0, 0, 1]])
                                      
        self.inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    
    def _image_to_camera_frame_matrix(self):
        """
            Define the function that, given as input the homogenous transformation matrix of a
            point in the image frame, produce the point defined with respect to the camera frame.
        """
        # Rotation matrix to align image frame to camera frame
        rotation_image_camera_frame = np.dot(main.rotate_z(-90 * pi /180), main.rotate_x(-90 * pi /180))

        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]
        # Lambda Function for transformation of image frame in camera frame 
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame , object_camera_frame)

    def _correct_perpendicular_distance(self, distance):
        """Corrects the distance from the traffic light by taking into account the
        ego vehicle extension on the long side.

        args:
            distance: distance parallel to the lane between the traffic light and the ego vehicle.
        returns:
            new_distance: the corrected distance.
        """

        # Computes the new distance by subtracting half of the extension of the ego
        # vehicle on the long side. 
        new_distance = distance - main.CAR_RADII_X_EXTENT
        new_distance = 0 if new_distance < 0 else new_distance
        return new_distance