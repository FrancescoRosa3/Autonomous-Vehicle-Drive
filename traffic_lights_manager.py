from math import pi
import os
from numpy.lib.arraypad import _slice_at_axis

from numpy.lib.function_base import select
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CLASSES = ["go", "stop", "UNKNOWN"]
UNKNOWN = 2
STOP = CLASSES[1]
X_OFFSET = 30
Y_OFFSET = 30
TRAFFIC_SIGN_LABEL = 12

class trafficLightsManager:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module\\config.json')):
        self.config_path = config_path
        self.tl_det = trafficLightDetector()
        
        self.new_state_counter = 0
        self.true_state = UNKNOWN
        self.curr_state = UNKNOWN
        self.distance = None

    def get_tl_state(self, image, depth_img = None, semantic_img = None):
        
        self._set_current_frame(image, depth_img, semantic_img)
        self._update_state()
        self._update_distance()

        return self.true_state, self.distance

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

            width = xmax - xmin
            height = ymax - ymin
            
            # take the pixel coordinates for the traffic light
            traffic_light_pixels = self._slice_traffic_light_from_semantic_segmentation(xmin, xmax, ymin, ymax)
            
            # false positivo bounding box
            if len(traffic_light_pixels) == 0:
                self.distance = None
                return

            # take the traffic light pixels from the depth image
            depth_sum = 0
            for pixel in traffic_light_pixels:
                # convert depth image value in meters
                depth_sum = depth_sum + 1000 * self.curr_depth_img[pixel[0]][pixel[1]]

            self.distance = depth_sum/len(traffic_light_pixels)

            print(self.distance)
        else:
            self.distance = None

    def _update_state(self):
        self.curr_bb = self.tl_det.detect_on_image(self.curr_img)

        if self.curr_bb == None:
            bb_state = UNKNOWN
        else:
            bb_state = self.curr_bb.get_label()
        
        if bb_state == self.curr_state:
            self.new_state_counter += 1
            if self.new_state_counter >= 3:
                self.true_state = CLASSES[self.curr_state]
        else:
            self.new_state_counter = 0
            self.curr_state = bb_state

    def _slice_traffic_light_from_semantic_segmentation(self, x_min, x_max, y_min, y_max):
        # neighborhood of the bounding box
        x_offset = X_OFFSET
        y_offset = Y_OFFSET

        # get the pixels belonging to traffic sign
        traffic_light_pixels = []

        if(y_min - y_offset < 0):
            start_y = 0
        else:
            start_y = y_min-y_offset
        
        print(self.curr_semantic_img.shape[0])
        if((y_max + y_offset) > self.curr_semantic_img.shape[0]):
            end_y = self.curr_semantic_img.shape[0]
        else:
            end_y = y_max + y_offset

        if(x_min - x_offset < 0):
            start_x = 0
        else:
            start_x = x_min-x_offset
        
        if((x_max + x_offset) > self.curr_semantic_img.shape[1]):
            end_x = self.curr_semantic_img.shape[1]
        else:
            end_x = x_max + x_offset

        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                if self.curr_semantic_img[i][j] == TRAFFIC_SIGN_LABEL:
                    # traffic light pixel 
                    traffic_light_pixels.append((i,j))
        
        return traffic_light_pixels