from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Load the two video clips
clip1 = VideoFileClip("/home/mommymythra/Videos/Fall 2022 Crash Videos/Room 354/TT4525final.webm")
clip2 = VideoFileClip("/home/mommymythra/Videos/Fall 2022 Crash Videos/Room 354/LF6983final.webm")

# Load the two images
image1 = "/home/mommymythra/CarlaTera2/Virtuous-Vehicle/color_trajectories/final/TT4525final.png"
image2 = "/home/mommymythra/CarlaTera2/Virtuous-Vehicle/color_trajectories/final/LF6983final.png"

# Get the duration of the shorter video
duration = min(clip1.duration, clip2.duration)

# Place the videos side by side
final_clip = CompositeVideoClip([clip1.crossfadein(0.5).set_position(("left", "center")),
                                 clip2.crossfadein(0.5).set_position(("right", "center"))],
                                size=(1920, 1080)).set_duration(duration)

# Load and place the images on top of the videos
image_clip1 = ImageClip(image1, duration=duration).set_position(("left", "top"))
image_clip2 = ImageClip(image2, duration=duration).set_position(("right", "top"))
final_clip = CompositeVideoClip([final_clip, image_clip1, image_clip2])

# Save the final video
final_clip.write_videofile("output.mp4", codec="libx264", fps=24)
