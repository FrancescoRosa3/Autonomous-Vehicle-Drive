import os
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector
import numpy as np
from math import cos, sin, pi,tan
import main

BASE_DIR = os.path.dirname(__file__)

TRAFFIC_LIGHT_CLASSES = ["go", "stop", "UNKNOWN"]

GO = TRAFFIC_LIGHT_CLASSES[0]
STOP = TRAFFIC_LIGHT_CLASSES[1]
UNKNOWN = TRAFFIC_LIGHT_CLASSES[2]

X_OFFSET = 30
Y_OFFSET = 30
TRAFFIC_SIGN_LABEL = 12
STOP_COUNTER = 4
OTHERS_COUNTER = 3

class trafficLightsManager:

    def __init__(self, camera_parameters, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module\\config.json')):
        self.config_path = config_path
        self.tl_det = trafficLightDetector()
        
        self.new_state_counter = 0
        self.true_state = UNKNOWN
        self.curr_state = UNKNOWN
        self.distance = None
        self.vehicle_frame_list = []
        self.create_intrinsic_matrix(camera_parameters)
        self.image_to_camera_frame_matrix()

    def get_tl_state(self, image, depth_img = None, semantic_img = None):
        
        self._set_current_frame(image, depth_img, semantic_img)
        self._update_state()
        self._update_distance()

        return self.true_state, self.distance, self.vehicle_frame_list

    def _set_current_frame(self, image, depth_img, semantic_img):
        self.curr_img = image
        self.curr_depth_img = depth_img
        self.curr_semantic_img = semantic_img

    def _update_distance(self):
        # compute the distance from the traffic light if its state is red
        if self.true_state == STOP and self.curr_bb != None:
            
            image_h, image_w = self.curr_depth_img.shape
            # compute the bb coordinates
            xmin = int(self.curr_bb.xmin * image_w)
            ymin = int(self.curr_bb.ymin * image_h)
            xmax = int(self.curr_bb.xmax * image_w)
            ymax = int(self.curr_bb.ymax * image_h)

            # take the pixel coordinates for the traffic light
            traffic_light_pixels = self._get_traffic_light_slice_from_semantic_segmentation(xmin, xmax, ymin, ymax)

            # false positive bounding box
            if len(traffic_light_pixels) == 0:
                self.distance = None
                return

            # take the traffic light pixels from the depth image
            x_distance = 0
            temp_avg = 0
            i = 0
            self.vehicle_frame_list = []
            for pixel in traffic_light_pixels:
                i += 1
                # convert depth image value in meters
                depth = 1000 * self.curr_depth_img[pixel[0]][pixel[1]]
                #depth_sum = depth_sum + depth

                ### Compute the pixel position in vehicle frame
                
                # From pixel to waypoint
                pixel = [pixel[1] , pixel[0], 1]
                pixel = np.reshape(pixel, (3,1))

                # Projection Pixel to Image Frame
                image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth

                # Create extended vector
                image_frame_vect_extended = np.zeros((4,1))
                image_frame_vect_extended[:3] = image_frame_vect 
                image_frame_vect_extended[-1] = 1
                
                # Projection Camera to Vehicle Frame
                camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
                camera_frame = camera_frame[:3]
                camera_frame = np.asarray(np.reshape(camera_frame, (1,3)))

                camera_frame_extended = np.zeros((4,1))
                camera_frame_extended[:3] = camera_frame.T 
                camera_frame_extended[-1] = 1

                camera_to_vehicle_frame = np.zeros((4,4))
                camera_to_vehicle_frame[:3,:3] = main.to_rot([self.cam_pitch, self.cam_yaw, self.cam_roll])
                camera_to_vehicle_frame[:,-1] = [self.cam_x_pos, self.cam_y_pos, self.cam_height, 1]

                vehicle_frame = np.dot(camera_to_vehicle_frame,camera_frame_extended )
                vehicle_frame = vehicle_frame[:3]
                vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))

                self.vehicle_frame_list.append([vehicle_frame[0][0], -vehicle_frame[0][1]])
                
                x_distance += vehicle_frame[0][0]

            self.distance = x_distance/len(traffic_light_pixels)
        else:
            self.distance = None

    def _update_state(self):
        self.curr_bb = self.tl_det.detect_on_image(self.curr_img)

        if self.curr_bb == None:
            bb_state = UNKNOWN
        else:
            bb_state = TRAFFIC_LIGHT_CLASSES[self.curr_bb.get_label()]
        # print(bb_state)
        if bb_state == self.curr_state:
            self.new_state_counter += 1
            
            # set the threshold to a different value based on the detected traffic light state
            threshold = STOP_COUNTER if self.true_state == "stop" else OTHERS_COUNTER
            if self.new_state_counter >= threshold:
                    self.true_state = self.curr_state
        else:
            self.new_state_counter = 0
            self.curr_state = bb_state

    def _get_traffic_light_slice_from_semantic_segmentation(self, x_min, x_max, y_min, y_max):
        
        # neighborhood of the bounding box
        width = x_max - x_min
        x_offset = int(width + ((width*30)/100))
        y_offset = 0

        # prevent indices from going out of image borders
        start_y = 0 if y_min - y_offset < 0 else y_min-y_offset
        end_y = self.curr_semantic_img.shape[0] if (y_max + y_offset) > self.curr_semantic_img.shape[0] else y_max + y_offset
        start_x = 0 if x_min - x_offset < 0 else x_min-x_offset
        end_x = self.curr_semantic_img.shape[1] if (x_max + x_offset) > self.curr_semantic_img.shape[1] else x_max + x_offset

        # get the pixels belonging to traffic sign
        traffic_light_pixels = []
        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                if self.curr_semantic_img[i][j] == TRAFFIC_SIGN_LABEL:
                    traffic_light_pixels.append((i, j))

        return traffic_light_pixels

    def create_intrinsic_matrix(self, camera_parameters):
        ### Compute the transformation matrices from image to camera frame
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
    
    def image_to_camera_frame_matrix(self):
        # Rotation matrix to align image frame to camera frame
        rotation_image_camera_frame = np.dot(main.rotate_z(-90 * pi /180), main.rotate_x(-90 * pi /180))

        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        # Lambda Function for transformation of image frame in camera frame 
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame , object_camera_frame)