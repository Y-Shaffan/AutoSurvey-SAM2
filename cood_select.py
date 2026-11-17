import os
import matplotlib.pyplot as plt
from PIL import Image

videoFolder = "D:/Shaffan/AutoSurvey/media/vids/Hithadhoo_dir_Fuvahmulah_Dec2025/frames"

frame_names = [
    p for p in os.listdir(videoFolder)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".JPG", "JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

frame_idx = 41435
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(videoFolder, frame_names[frame_idx])))
plt.show()