from traffic_light_detection_module.traffic_light_detector import traffic_light_detector

BASE_DIR = os.path.dirname(__file__)

class traffic_lights_manager:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'config.json')):
        self.config_path = config_path
        self.tl_det = traffic_light_detector()
        
    def set_current_frame(self, image):
        self.curr_frame = image
        

    def get_traffic_light_state(self):
        pass
        
        