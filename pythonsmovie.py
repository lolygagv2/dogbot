import os
import pathlib

# Path to your videos
video_folder = "/home/morgan/dog2/dogvids"

# Get all video files (mp4 and mov)
videos = []
for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV']:
    videos.extend(pathlib.Path(video_folder).glob(ext))

# Sort them so you can review in order
videos = sorted(videos)

# Print them with numbers so you can label them
print("Review your videos and note which behavior each shows:")
for i, vid in enumerate(videos, 1):
    print(f"{i}. {vid.name}")

print("\nNow create a text file mapping video number to behavior:")
print("Example format:")
print("1 sit")
print("2 sit")
print("3 stand")
print("etc...")