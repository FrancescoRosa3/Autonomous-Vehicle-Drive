import os
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip
import argparse

def produce_video(args):

    video_directory = os.fsencode("Videos") if (args.quality == "Low") else os.fsencode("HDVideos")

    for ep_dir in os.listdir(video_directory):
        print(f"dir: {ep_dir}")
        skip_folder = False
        filename = os.fsdecode(ep_dir)
        for sub_dir in os.listdir(os.path.join(video_directory, ep_dir)):
            print(f"sub_dir: {sub_dir}")
            if sub_dir.decode("utf-8") == "Reports":
                skip_folder = True

        if not skip_folder:
            ep_dir_path = os.path.join(video_directory.decode("utf-8"), ep_dir.decode("utf-8"))
            parameters_string = ep_dir.decode("utf-8")

            try:
                os.makedirs(os.path.join(ep_dir_path, "Reports"))
            except:
                pass

            in_path = os.path.join(ep_dir_path, "Temp")

            camera_out_path = os.path.join(ep_dir_path, "Reports", "camera" + parameters_string + ".mp4")
            trajectory_out_path = os.path.join(ep_dir_path, "Reports", "trajectory" + parameters_string + ".mp4")
            forward_speed_out_path = os.path.join(ep_dir_path, "Reports", "forward_speed" + parameters_string + ".mp4")
            tl_out_path = os.path.join(ep_dir_path, "Reports", "tl_camera" + parameters_string + ".mp4")
            fsm_out_path = os.path.join(ep_dir_path, "Reports", "fsm" + parameters_string + ".mp4")

            # os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i Temp/camera_%d.jpeg -c:v libx264 -pix_fmt yuv420p -r 15 " + out_path + " >nul 2>&1")
            os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i " + in_path + "\camera_%d" + parameters_string + ".jpeg -c:v libx264 -pix_fmt yuv420p -b 10000k -r 15 " + camera_out_path)
            os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i " + in_path + "\\trajectory_%d" + parameters_string + ".png -c:v libx264 -pix_fmt yuv420p -b 10000k -r 15 " + trajectory_out_path)
            os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i " + in_path + "\\tl_camera_%d" + parameters_string + ".jpeg -c:v libx264 -pix_fmt yuv420p -b 10000k -r 15 " + tl_out_path)
            os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i " + in_path + "\\fsm_%d" + parameters_string + ".png -vf \"scale=420:420\" -c:v libx264 -pix_fmt yuv420p -b 10000k -r 15 " + fsm_out_path)
            os.system("echo y | ffmpeg -r 15 -pattern_type sequence -i " + in_path + "\\forward_speed_%d" + parameters_string + ".png -c:v libx264 -pix_fmt yuv420p -b 10000k -r 15 " + forward_speed_out_path)

            ### editing video
            result_out_path = os.path.join(ep_dir_path, "Reports", "result.mp4")

            clip1 = VideoFileClip(camera_out_path)
            clip2 = VideoFileClip(trajectory_out_path)
            clip3 = VideoFileClip(tl_out_path)
            clip4 = VideoFileClip(fsm_out_path)
            clip5 = VideoFileClip(forward_speed_out_path)
            if args.quality == "Low":
                clip2 = clip2.resize(0.45)
                clip3 = clip3.resize(0.70)
                clip4 = clip4.resize(0.33)
                clip5 = clip5.resize(0.60)
            else:
                clip2 = clip2.resize(height = 648)
                clip3 = clip3.resize(height = 545)
                clip4 = clip4.resize(height = 350)
                clip5 = clip5.resize(height = 250)
            final_clip = CompositeVideoClip([clip1, clip2.set_position(("left", "bottom")), clip3.set_position(("right", "top")), clip4.set_position(("right", "center")), clip5.set_position(("left", "top"))])#.set_duration(clip1)
            final_clip.write_videofile(result_out_path)

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '-q', '--quality',
    choices=['Low', 'High'],
    type=lambda s: s.title(),
    default='Low',
    help='graphics quality level.')
args = argparser.parse_args()

produce_video(args)
