import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:
    name = os.path.splitext(file)[0]
    print(name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{name}.mp3"])

