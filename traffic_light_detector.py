import json
import os
import argparse
from traffic_light_detection_module.predict import get_model, predict_with_model_from_image
import cv2
from traffic_light_detection_module.postprocessing import bbox_iou, draw_boxes

BASE_DIR = os.path.dirname(__file__)

class TrafficLightDetector:

    def __init__(self, config_path = os.path.join(BASE_DIR, 'traffic_light_detection_module', 'config.json')):
        
        # Load the detector configuration from file
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())
        
        self.config = config

        # Get the detection model
        self.model = get_model(self.config)

    def detect_on_image(self, image):
        """
        Detect the traffic light bounding box from the given image.

        args:
            image: Camera RGB image.
        
        returns:
            best_bb: Bounding box with the higher score.
        """
        
        # Get the traffic light bounding boxes.
        netout = predict_with_model_from_image(self.config, self.model, image)
        
        # Get the best bounding box from the previous set.
        best_bb = self.get_best_bb(netout)

        # If a bounding box exists draw it over the camera image.
        if best_bb != None:
            image = draw_boxes(image, [best_bb], self.config['model']['classes'])

        # Show camera image with produced bounding box
        cv2.imshow('demo', image)
        cv2.waitKey(1)
        
        # return the bounding box with the higher score
        return best_bb

    def get_best_bb(self, boxes):
        """
        Find the higher score bounding box between the given set.

        args:
            boxes: set of bounding box.
        """
        
        # Check if the set of bounding box length is greater than 0
        if len(boxes) > 0:
            
            # Initialize the chosen bounding box
            chosen_box = boxes[0]
            chosen_box_score = chosen_box.get_score()
            
            # Update the chosen bounding box with the one which has the higher score.
            for box in boxes:
                box_score = box.get_score()
                if box_score > chosen_box_score:
                    chosen_box_score = box_score
                    chosen_box = box

            # return the chosen bounding box
            return chosen_box
        
        # return None if no bounding boxes exist.
        return None
