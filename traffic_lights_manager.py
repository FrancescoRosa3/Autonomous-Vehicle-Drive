import os
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector

BASE_DIR = os.path.dirname(__file__)
UNKNOWN = "UNKNOWN"

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

        self.distance = None
        return self.true_state, self.distance

    def _set_current_frame(self, image, depth_img):
        self.curr_img = image
        self.curr_depth_img = depth_img

    def _update_state(self):
        bb = self.tl_det.detect_on_image(self.curr_img)

        if bb == None:
            bb_state = UNKNOWN
        else:
            bb_state = bb.get_label()
        
        if bb_state == self.curr_state:
            self.new_state_counter += 1
            if self.new_state_counter >= 3:
                self.true_state = self.curr_state
        else:
            self.new_state_counter = 0
            self.curr_state = bb_state
        