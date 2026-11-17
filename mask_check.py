import os
import cv2
import numpy as np
from PIL import Image
m = np.array(Image.open("output/test_run2/masks/00560.png"))
print(np.unique(m), (m > 0).sum())

# def save_viewable(g, lab):
#     os.makedirs("output/masks_view", exist_ok=True)
#     cv2.imwrite(f"output/masks_view/{g:05d}.png", (lab > 0).astype(np.uint8) * 255)