import os
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector
import cv2
import math

BASE_DIR = os.path.dirname(__file__)
CLASSES = ["go", "stop", "UNKNOWN"]
UNKNOWN = 2
STOP = CLASSES[1]

class trafficLightsManager:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module\\config.json')):
        self.config_path = config_path
        self.tl_det = trafficLightDetector()
        
        self.new_state_counter = 0
        self.true_state = UNKNOWN
        self.curr_state = UNKNOWN

    def get_tl_state(self, image, depth_img = None):
        
        self._set_current_frame(image, depth_img)
        self._update_state()

        if self.true_state == STOP and self.bb != None:
            
            image_h, image_w = depth_img.shape

            xmin = int(self.bb.xmin * image_w)
            ymin = int(self.bb.ymin * image_h)
            xmax = int(self.bb.xmax * image_w)
            ymax = int(self.bb.ymax * image_h)

            width = xmax - xmin
            height = ymax - ymin
            print("width: "+ str(width))
            print("height: "+ str(height))
            print("\n")

            bb_depth = depth_img[xmin : xmin + width, ymin : ymin + height]
            cv2.imshow("bb", bb_depth)
            cv2.waitKey(1)

            in_meters = 1000 * bb_depth
            
            average_depth = 0

            for row in in_meters:
                for elem in row:
                    average_depth += elem

            self.depth = average_depth/(width * height)

        else:
            self.depth = math.inf

        return self.true_state, self.depth


    def _set_current_frame(self, image, depth_img):
        self.curr_img = image
        self.curr_depth_img = depth_img

    

    def _update_state(self):
        self.bb = self.tl_det.detect_on_image(self.curr_img)

        if self.bb == None:
            bb_state = UNKNOWN
        else:
            bb_state = self.bb.get_label()
        
        if bb_state == self.curr_state:
            self.new_state_counter += 1
            if self.new_state_counter >= 3:
                self.true_state = CLASSES[self.curr_state]
        else:
            self.new_state_counter = 0
            self.curr_state = bb_state