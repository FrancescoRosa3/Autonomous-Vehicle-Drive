import os
import shutil
from PIL import Image
import behavioural_planner
import main

HIGH_QUALITY = True 
main_folder = "HDVideos" if HIGH_QUALITY else "Videos"

def create_video_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder+ "/Temp")

def save_video_graph(graph, name, frame_counter):
    create_video_output_dir(f"{main_folder}/{main.PARAMS_STRING}")
    graph.savefig(f'{main_folder}/{main.PARAMS_STRING}/Temp/{name}_{frame_counter}{main.PARAMS_STRING}.png')

def save_video_image(img, name, frame_counter):
    create_video_output_dir(f"{main_folder}/{main.PARAMS_STRING}")
    im = Image.fromarray(img)
    im.save(f"{main_folder}/{main.PARAMS_STRING}/Temp/{name}_{frame_counter}{main.PARAMS_STRING}.jpeg")

def copy_state_image(state, frame_counter):
    create_video_output_dir(f"{main_folder}/{main.PARAMS_STRING}")
    out_path = f"{main_folder}/{main.PARAMS_STRING}/Temp/fsm_{frame_counter}{main.PARAMS_STRING}.png"
    in_path = ""
    if state == behavioural_planner.STOP_AT_OBSTACLE:
        in_path = "fsm_imgs\\fsm_stop_at_obstacle.png"
    elif state == behavioural_planner.FOLLOW_LANE:
        in_path = "fsm_imgs\\fsm_follow_lane.png"
    elif state == behavioural_planner.STOP_AT_TRAFFIC_LIGHT:
        in_path = "fsm_imgs\\fsm_stop_at_traffic_light.png"
    elif state == behavioural_planner.APPROACHING_RED_TRAFFIC_LIGHT:
        in_path = "fsm_imgs\\fsm_approaching_red_traffic_light.png"
    shutil.copy(in_path, out_path)