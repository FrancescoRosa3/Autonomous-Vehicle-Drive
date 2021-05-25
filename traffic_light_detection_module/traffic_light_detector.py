import json
import os
import argparse
from predict import get_model
import cv2

from postprocessing import draw_boxes
from predict import predict_with_model_from_image


BASE_DIR = os.path.dirname(__file__)

class traffic_light_detector:

    def __init__(self, config = 'config.json'):
        self.config = config

    def detect_on_image(self, image):
        model = get_model(self.config)
        
        netout = predict_with_model_from_image(self.config, model, image)
        print(f"NETOUT: {netout}")
        plt_image = draw_boxes(image, netout, self.config['model']['classes'])

        cv2.imshow('demo', plt_image)
        #cv2.waitKey(0)

        # cv2.imwrite(os.path.join(OUT_IMAGES_DIR, 'out' + str(img_num) + '.png'), plt_image)