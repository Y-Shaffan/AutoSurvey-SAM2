import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import io

# # Cut video into frames manually:
# # ffmpeg -i input.mp4 -q:v 2 -start_number 0 %05d.jpg

# GPU/CPU selection
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__() # Use bfloat16 for inference

# SAM model
sam2_checkpoint = "D:/Shaffan/AutoSurvey/sam2/checkpoints/sam2.1_hiera_tiny.pt"

# Config file for SAM model
model_cfg = "D:/Shaffan/AutoSurvey/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

# Create predictor object
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Mask function
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Show points
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)

# Path to video frames
videoFolder = "D:/Shaffan/AutoSurvey/media/vids/Hithadhoo_dir_Fuvahmulah_Dec2025/frames"

frame_names = [
    p for p in os.listdir(videoFolder)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".JPG", "JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# # Manually find the coordinates of the object in the first frame
# frame_idx = 19480
# plt.figure(figsize=(12, 8))
# plt.title(f"frame {frame_idx}")
# plt.imshow(Image.open(os.path.join(videoFolder, frame_names[frame_idx])))
# plt.show()
# print("Press Enter to continue...")
# input()
# print("Continuing...")

# Initialization
inference_state = predictor.init_state(
    video_path=videoFolder,
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
)
predictor.reset_state(inference_state)

ann_frame_idx = 0 # Frame index for annotation
ann_obj_id = 1 # Object ID for annotation

points = np.array([[507, 637], [1909, 1047]], dtype=np.float32)
labels = np.array([1, 1], dtype=np.int32) # 1 for positive click, 0 for negative click
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels
)

# Visualize results on the annotated frame
# plt.figure(figsize=(12, 8))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(videoFolder, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()
# print("Press Enter to continue...")
# input()
# print("Continuing...")

#  Run propagation throught video and results in dict
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")

# Define figure outside loop
print("Saving processed images")
# fig = plt.figure(figsize=(16, 12))
fig, ax = plt.subplots(figsize=(10,6), dpi=150)
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    # plt.title(f"frame {out_frame_idx}")
    im=plt.imshow(Image.open(os.path.join(videoFolder, frame_names[out_frame_idx])), animated=True, aspect="auto")
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    print("Saving image", out_frame_idx)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        f'output/s{out_frame_idx}.png',
        bbox_inches = 'tight',
        pad_inches = 0,
        dpi = 150,
    )
#     # print(torch.cuda.memory.memory_summary())
#     # torch.cuda.empty_cache()
#     # plt.show()
print("Saving processed images")

# # # Combine frames into video:
# # # ffmpeg -framerate 24 -i s%d.png -c:v libx264 -r 24 output.mp4