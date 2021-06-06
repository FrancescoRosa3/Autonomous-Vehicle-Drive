import os
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip
parameters_string = "_2-20-1000-100-500-0_"
camera_out_path = os.path.join("Videos", parameters_string, "Reports", "camera" + parameters_string + ".mp4")
trajectory_out_path = os.path.join("Videos", parameters_string, "Reports", "trajectory" + parameters_string + ".mp4")
tl_out_path = os.path.join("Videos", parameters_string, "Reports", "tl_camera" + parameters_string + ".mp4")
fsm_out_path = os.path.join("Videos", parameters_string, "Reports", "fsm" + parameters_string + ".mp4")

result_out_path = os.path.join("Videos", parameters_string, "Reports", "result.mp4")

clip1 = VideoFileClip(camera_out_path)
clip2 = VideoFileClip(trajectory_out_path)
clip3 = VideoFileClip(tl_out_path)
clip4 = VideoFileClip(fsm_out_path)
clip2 = clip2.resize(0.45)
clip3 = clip3.resize(0.70)
clip4 = clip4.resize(0.35)
final_clip = CompositeVideoClip([clip1, clip2.set_position(("left", "bottom")), clip3.set_position(("right", "top")), clip4.set_position(("right", "center"))])#.set_duration(clip1)
final_clip.write_videofile(result_out_path)
