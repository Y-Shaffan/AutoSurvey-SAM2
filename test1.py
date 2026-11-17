import os, gc, shutil, tempfile
import numpy as np, torch, cv2
from sam2.build_sam import build_sam2_video_predictor

# # Cut video into frames manually:
# # ffmpeg -i input.mp4 -q:v 2 -start_number 0 %05d.jpg

CHUNK = 500 # frames per chunk, ~12.6 MB RAM each (500 ≈ 6.3 GB), approx 2MB per frame for 1080p
VIDEO_FOLDER = "D:/Shaffan/AutoSurvey/media/vids/Hithadhoo_dir_Fuvahmulah_Dec2025/frames"
OUT_MASKS = "output/test_run2/masks"
OUT_VIDEO = "output/test_run2/overlay.mp4"
sam2_checkpoint, model_cfg = "D:/Shaffan/AutoSurvey/sam2/checkpoints/sam2.1_hiera_tiny.pt", "D:/Shaffan/AutoSurvey/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
points = np.array([[507, 637], [1909, 1047]], dtype=np.float32)
labels = np.array([1, 1], dtype=np.int32)

os.makedirs(OUT_MASKS, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

frame_names = sorted(
    [p for p in os.listdir(VIDEO_FOLDER)
     if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")],
    key=lambda p: int(os.path.splitext(p)[0]))
n = len(frame_names)
H, W = cv2.imread(os.path.join(VIDEO_FOLDER, frame_names[0])).shape[:2]
writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

def make_chunk_dir(names):
    d = tempfile.mkdtemp(prefix="sam2_chunk_")
    for i, name in enumerate(names):
        src = os.path.abspath(os.path.join(VIDEO_FOLDER, name))
        dst = os.path.join(d, f"{i:05d}.jpg")
        try:
            os.symlink(src, dst) # no disk copy
        except OSError:
            shutil.copy2(src, dst) # fallback (e.g. Windows without symlink rights)
    return d

prev_masks, start = {}, 0
OBJ_ID = 1
# Manual re-prompts for when the object is lost: global frame index -> (points, labels).
REPROMPTS = {
    0:(np.array([[516, 645], [522, 639], [1901, 973], [1905, 1054]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    470:(np.array([[631, 1072], [724, 1054], [1513, 15], [1471, 19]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    605:(np.array([[623, 1049], [768, 1068], [929, 531], [846, 513]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    900:(np.array([[631, 1070], [776, 1072], [832, 267], [786, 271]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    15465:(np.array([[662, 1045], [945, 1064], [1059, 116], [986, 129]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    16715:(np.array([[862, 944], [980, 874], [924, 583], [858, 608]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    17505:(np.array([[869, 936], [953, 878], [947, 459], [918, 465]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    18660:(np.array([[718, 1062], [1085, 1068], [1263, 34], [1050, 44]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    25150:(np.array([[827, 921], [922, 930], [865, 610], [832, 616]], np.float32), np.array([1, 1, 1, 1], np.int32)),
    41435:(np.array([[848, 822], [1026, 816], [896, 89], [865, 92]], np.float32), np.array([1, 1, 1, 1], np.int32))
}

def write_frame(g, masks):
    lab = np.zeros((H, W), np.uint8) # pixel value = object id
    for oid, m in masks.items():
        lab[m] = oid
    cv2.imwrite(os.path.join(OUT_MASKS, f"{g:05d}.png"), lab)
    img = cv2.imread(os.path.join(VIDEO_FOLDER, frame_names[g]))
    for oid, m in masks.items():
        c = np.array(COLORS[oid % len(COLORS)], np.float32)
        img[m] = (0.6 * img[m] + 0.4 * c).astype(np.uint8)
    writer.write(img)

prev_masks, start, written = {}, 0, set()
while True:
    end = min(start + CHUNK, n)
    chunk_dir = make_chunk_dir(frame_names[start:end])
    state = predictor.init_state(video_path=chunk_dir,
                                 offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)
    seeded = False

    if start == 0:
        predictor.add_new_points(inference_state=state, frame_idx=0, obj_id=OBJ_ID,
                                 points=points, labels=labels)
        seeded = True
    else:
        for oid, m in prev_masks.items(): # carry-over from previous chunk
            if m.any():
                predictor.add_new_mask(inference_state=state, frame_idx=0,
                                       obj_id=oid, mask=torch.from_numpy(m))
                seeded = True

    for g, (pts, lbs) in REPROMPTS.items(): # manual re-seeds inside this chunk
        if start <= g < end:
            predictor.add_new_points(inference_state=state, frame_idx=g - start,
                                     obj_id=OBJ_ID, points=pts, labels=lbs)
            seeded = True

    chunk_masks, last = {}, {}
    if seeded:
        for local_idx, obj_ids, logits in predictor.propagate_in_video(state):
            masks = {oid: (logits[i] > 0).cpu().numpy()[0] for i, oid in enumerate(obj_ids)}
            chunk_masks[start + local_idx] = masks
            last = masks

    for g in range(start, end): # write in order; blank where nothing was tracked
        if g not in written:
            write_frame(g, chunk_masks.get(g, {}))
            written.add(g)

    lost = not any(m.any() for m in last.values())
    print(f"frames {start}-{end-1}: seeded={seeded}, object at last frame={'NO' if lost else 'yes'}")
    if lost and end < n:
        print(f"  -> object lost; add a REPROMPTS entry for a frame after {end-1}")

    predictor.reset_state(state)
    del state
    shutil.rmtree(chunk_dir, ignore_errors=True)
    gc.collect(); torch.cuda.empty_cache()

    prev_masks = last
    if end >= n:
        break
    start = end - 1

writer.release()

# # # Combine frames into video:
# # # ffmpeg -framerate 24 -i s%d.png -c:v libx264 -r 24 output.mp4