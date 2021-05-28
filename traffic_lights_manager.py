from math import pi
import os
from numpy.lib.arraypad import _slice_at_axis

from numpy.lib.function_base import select
from traffic_light_detection_module.traffic_light_detector import trafficLightDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
            traffic_light_pixels, traffic_light_pixels2 = self.get_traffic_light_slice_from_semantic_segmentation(xmin, xmax, ymin, ymax)
            
            N = M = 416

            img = np.zeros([N,M,3])

            img[:,:,0] = np.ones([N,M])*255/255.0
            img[:,:,1] = np.ones([N,M])*255/255.0
            img[:,:,2] = np.ones([N,M])*255/255.0
            #data = np.full((N, M, 3), [255, 255, 255], dtype=int)
            #data = [[[255, 255, 255] for i in range(416)] for j in range(416)]
            for elem in traffic_light_pixels:
                img[elem[0], elem[1], 0] = 1
                img[elem[0], elem[1], 1] = 1
                img[elem[0], elem[1], 2] = 0
            '''
            for elem in traffic_light_pixels2:
                if elem not in traffic_light_pixels:
                    img[elem[0], elem[1], 0] = 0
                    img[elem[0], elem[1], 1] = 0.5
                    img[elem[0], elem[1], 2] = 0
            ''' 

            

            # false positive bounding box
            if len(traffic_light_pixels) == 0:
                self.distance = None
                return
                
            # take the traffic light pixels from the depth image
            depth_sum = 0
            temp_avg = 0
            i = 0
            for pixel in traffic_light_pixels:
                i += 1
                # convert depth image value in meters
                in_meter_val = 1000 * self.curr_depth_img[pixel[0]][pixel[1]]
                #print(in_meter_val)
                depth_sum = depth_sum + in_meter_val
                temp_avg = depth_sum/i
                print(f"val: {in_meter_val} - avg + offset: {temp_avg + ((temp_avg *30) / 100)}")     
                if in_meter_val > temp_avg + ((temp_avg * 30) / 100):
                    img[pixel[0], pixel[1], 0] = 0
                    img[pixel[0], pixel[1], 1] = 0
                    img[pixel[0], pixel[1], 2] = 1
                    

            
            cv2.imshow("test", img)
            cv2.waitKey(1)

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

    def get_traffic_light_slice_from_semantic_segmentation(self, x_min, x_max, y_min, y_max):
        # neighborhood of the bounding box
        x_offset = X_OFFSET
        y_offset = Y_OFFSET

        # get the pixels belonging to traffic sign
        traffic_light_pixels = []

        # prevent indices from going out of image borders
        start_y = 0 if y_min - y_offset < 0 else y_min-y_offset
        end_y = self.curr_semantic_img.shape[0] if (y_max + y_offset) > self.curr_semantic_img.shape[0] else y_max + y_offset
        start_x = 0 if x_min - x_offset < 0 else x_min-x_offset
        end_x = self.curr_semantic_img.shape[1] if (x_max + x_offset) > self.curr_semantic_img.shape[1] else x_max + x_offset

        traffic_light_pixels = {}

        # find traffic light pixels
        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                if self.curr_semantic_img[i][j] == TRAFFIC_SIGN_LABEL:
                    if i not in traffic_light_pixels:
                        traffic_light_pixels[i] = []
                    traffic_light_pixels[i].append(j)
        
        print(f"BEFORE: {traffic_light_pixels}")
        temp2 = []
        for k, v in traffic_light_pixels.items():
            for elem in v:
                temp2.append((k, elem))
        
        # remove border from traffic light segment 
        min_raw = min(traffic_light_pixels.keys())
        print(f"min_raw: {min_raw}")
        max_raw = max(traffic_light_pixels.keys())
        print(f"max_raw: {max_raw}")
        del traffic_light_pixels[min_raw]
        del traffic_light_pixels[max_raw]
        for k in traffic_light_pixels.keys():
            if len(traffic_light_pixels[k])>1: 
                traffic_light_pixels[k] = traffic_light_pixels[k][1:-1]
            else:
                traffic_light_pixels[k] = []
        
        print(f"AFTER: {traffic_light_pixels}")

        temp1 = []
        for k, v in traffic_light_pixels.items():
            for elem in v:
                temp1.append((k, elem))
        #print(f"AFTER: {temp}")
        
        return temp1, temp2