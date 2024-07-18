from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip,clips_array
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from PIL import Image

# Load the two video clips
clip1 = VideoFileClip("/home/mommymythra/Videos/For Research Crew/DP8189FULLRUN.mp4").margin(5)
clip1 = clip1.resize(0.25)
clip2 = VideoFileClip("/home/mommymythra/Videos/For Research Crew/DP8189GARUN.mp4").margin(5)
clip2 = clip2.resize(0.25)
clip3 = VideoFileClip("/home/mommymythra/Videos/For Research Crew/SG9795FULLRUN.mp4").margin(5)
clip3 = clip3.resize(0.25)
clip4 = VideoFileClip("/home/mommymythra/Videos/For Research Crew/SG9795GARUN.mp4").margin(5)
clip4 = clip4.resize(0.25)

# Get the duration of the shorter video
duration = max(clip1.duration, clip2.duration, clip3.duration, clip4.duration)

# Load and place the images on top of the videos
final_clip = clips_array([[clip1,clip2],[clip3,clip4]])

# Save the final video
final_clip.write_videofile("output.mp4", codec="libx264", fps=24)
