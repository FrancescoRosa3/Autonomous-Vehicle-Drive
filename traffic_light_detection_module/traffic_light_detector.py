import json
import os
import argparse
from traffic_light_detection_module.predict import get_model, predict_with_model_from_image
import cv2

import main

from traffic_light_detection_module.postprocessing import bbox_iou, draw_boxes

BASE_DIR = os.path.dirname(__file__)

class trafficLightDetector:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'config.json')):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())
        
        self.config = config
        self.model = get_model(self.config)
        self.i = 0

        ### da eliminare
        self.frame_counter = 0
        
    def detect_on_image(self, image, save_image = False):
        
        netout = predict_with_model_from_image(self.config, self.model, image)
        best_bb = self.get_best_bb(netout)
        if best_bb != None:
            image = draw_boxes(image, [best_bb], self.config['model']['classes'])

        # Show and save image
        if main.SHOW_LIVE_PLOTTER:
            cv2.imshow('demo', image)
            cv2.waitKey(1)
        
        ### da eliminare
        if main.PRODUCE_VIDEO:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            main.save_video_image(image, "tl_camera", self.frame_counter)
            self.frame_counter += 1

        # return the bounding box with the higher score
        return best_bb

    def get_best_bb(self, boxes):
        if len(boxes) > 0:
            chosen_box = boxes[0]
            chosen_box_score = chosen_box.get_score()
            for box in boxes:
                box_score = box.get_score()
                if box_score > chosen_box_score:
                    chosen_box_score = box_score
                    chosen_box = box

            return chosen_box
        return None
