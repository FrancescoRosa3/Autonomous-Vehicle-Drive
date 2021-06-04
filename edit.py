import os
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip
parameters_string = "_91-148-400-0-123-0_"
camera_out_path = os.path.join("Videos", parameters_string, "Reports", "camera" + parameters_string + ".mp4")
trajectory_out_path = os.path.join("Videos", parameters_string, "Reports", "trajectory" + parameters_string + ".mp4")
tl_out_path = os.path.join("Videos", parameters_string, "Reports", "tl_camera" + parameters_string + ".mp4")

result_out_path = os.path.join("Videos", parameters_string, "Reports", "result.mp4")

clip1 = VideoFileClip(camera_out_path)
clip2 = VideoFileClip(trajectory_out_path)
clip3 = VideoFileClip(tl_out_path)
clip2 = clip2.resize(0.45)
clip3 = clip3.resize(0.70)
final_clip = CompositeVideoClip([clip1, clip2.set_position(("left", "bottom")), clip3.set_position(("right", "top"))])#.set_duration(clip1)
final_clip.write_videofile(result_out_path)
