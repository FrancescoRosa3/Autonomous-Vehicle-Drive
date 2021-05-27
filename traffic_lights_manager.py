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
            traffic_light_pixels_y, traffic_light_pixels_x = self._slice_traffic_light_from_semantic_segmentation(xmin, xmax, ymin, ymax)

            # take the traffic light pixels from the depth image
            bb_depth = self.curr_depth_img[traffic_light_pixels_y[0]:traffic_light_pixels_y[-1], 
                                            traffic_light_pixels_x[0]:traffic_light_pixels_x[-1]]

            cv2.imshow("traffic light", self.curr_img[traffic_light_pixels_y[0]:traffic_light_pixels_y[-1], 
                                                      traffic_light_pixels_x[0]:traffic_light_pixels_x[-1]])
            in_meters = 1000 * bb_depth
            
            depth_mat = np.matrix(in_meters)
            self.distance = np.average(depth_mat)
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
        traffic_light_pixels_x = []
        traffic_light_pixels_y = []

        for i in range(y_min-y_offset, y_max+y_offset):
            raw_with_traffic_light = False
            for j in range(x_min-x_offset, x_max+x_offset):
                
                if self.curr_semantic_img[i][j] == TRAFFIC_SIGN_LABEL:
                    # traffic light pixel 
                    traffic_light_pixels_x.append(j)
                    raw_with_traffic_light = True
            # append the raw index if the it contains a traffic sign pixel
            if raw_with_traffic_light:
                traffic_light_pixels_y.append(i)

        return traffic_light_pixels_y, traffic_light_pixels_x