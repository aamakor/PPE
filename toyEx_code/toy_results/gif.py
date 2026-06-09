# Change this line
from moviepy import VideoFileClip



# Load your video

for i in range(6):
    clip = VideoFileClip(f"./toyEx_code/images/video_dtlz{i+1}.mp4")
    clip.write_gif(f"./toyEx_code/images/video_dtlz{i+1}.gif", fps=15)
