import os

def produce_video():
    parameters_string = "_000_"
    out_path = ("Reports/iterations-sequence" + parameters_string + ".mp4")
    os.system("echo y | ffmpeg -r 2 -pattern_type sequence -i Temp/%d.jpeg -c:v libx264 -pix_fmt yuv420p -r 2 " + out_path + " >nul 2>&1")

produce_video()
