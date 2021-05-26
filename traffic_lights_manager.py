import os
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CLASSES = ["go", "stop", "UNKNOWN"]
UNKNOWN = 2
STOP = CLASSES[1]
OFFSET = 30

class trafficLightsManager:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module\\config.json')):
        self.config_path = config_path
        self.tl_det = trafficLightDetector()
        
        self.new_state_counter = 0
        self.true_state = UNKNOWN
        self.curr_state = UNKNOWN
        self.distance = None

    def get_tl_state(self, image, depth_img = None):
        
        self._set_current_frame(image, depth_img)
        self._update_state()
        self._update_distance()

        return self.true_state, self.distance

    def _set_current_frame(self, image, depth_img):
        self.curr_img = image
        self.curr_depth_img = depth_img

    def _update_distance(self):
        if self.true_state == STOP and self.curr_bb != None:
            
            image_h, image_w = self.curr_depth_img.shape

            xmin = int(self.curr_bb.xmin * image_w)
            ymin = int(self.curr_bb.ymin * image_h)
            xmax = int(self.curr_bb.xmax * image_w)
            ymax = int(self.curr_bb.ymax * image_h)

            width = xmax - xmin
            height = ymax - ymin

            x_offset = OFFSET
            y_offset = OFFSET

            bb_depth = self.curr_depth_img[ymin - y_offset : ymin + height + y_offset, xmin - x_offset: xmin + width + x_offset]

            in_meters = 1000 * bb_depth
            
            depth_mat = np.matrix(in_meters)
            self.distance = np.min(depth_mat)
            
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