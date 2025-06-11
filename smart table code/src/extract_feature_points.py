import numpy as np
import cv2 as cv
import customtkinter as ctk
import os
import time
from natsort import natsorted
from plantcv import plantcv as pcv
# GUI to select a folder
root = ctk.CTk()
img_folder = ctk.filedialog.askdirectory()
root.withdraw()

# Create output directory
output_dir = os.path.join(img_folder, 'motion_features')
os.makedirs(output_dir, exist_ok=True)

# Get sorted list of image files in the folder
img_lst = [
    os.path.normpath(os.path.join(img_folder, f))
    for f in os.listdir(img_folder)
    if os.path.isfile(os.path.join(img_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
img_lst = natsorted(img_lst)  # Make sure they are in temporal order

# Ensure at least 2 images
if len(img_lst) < 2:
    raise ValueError("Need at least two images to track motion.")

# Parameters
feature_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

# Load the first frame
old_frame = cv.imread(img_lst[0])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
feature_ids = np.arange(len(p0)).tolist()
mask = np.zeros_like(old_frame)

# Add a variable to control fading
fade_trails = False   # Set to False for no fading
fade_alpha = 0.1     # 0.0 = no fade, 1.0 = full fade each frame

#frames = [] # Store frames for video creation

# Process the rest of the frames
for img_path in img_lst[1:]:
    frame = cv.imread(img_path)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        st = st.flatten()
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        good_ids = [fid for fid, alive in zip(feature_ids, st) if alive == 1]

        for i, (new, old, fid) in enumerate(zip(good_new, good_old, good_ids)):
            a, b = new.ravel()
            c, d = old.ravel()
            filename = os.path.basename(img_path)
            # Extract date, time, and rep from filename
            try:
                base = filename.split("-Rep")[0]
                date_part, time_part = base.split("_")
                file_date = time.strftime("%m/%d/%Y", time.strptime(date_part, "%m-%d-%Y"))
                file_time = time_part.replace("-", ":")
                file_name = file_date + " " + file_time
                # Extract rep number
                rep_part = filename.split("-Rep")[1]
                rep = rep_part.split(".")[0]  # e.g., '2'
            except Exception as e:
                file_date = ""
                file_time = ""
                rep = ""
            with open(os.path.join(output_dir, f"feature_{fid}.csv"), "a") as f:
                f.write(f"{a},{b},{file_name},{rep}\n")
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[fid % len(color)].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[fid % len(color)].tolist(), -1)

        p0 = good_new.reshape(-1, 1, 2)
        feature_ids = good_ids

    # --- Fading effect for trails ---
    if fade_trails:
        mask = cv.addWeighted(mask, 1 - fade_alpha, np.zeros_like(mask), 0, 0)

    img_display = cv.add(frame, mask)
    #frames.append(img_display.copy())  # Store frame for video creation
    cv.imshow('frame', img_display)
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()

cv.destroyAllWindows()

'''output_path = os.path.join(output_dir, 'PlantMotionNoFade.mp4')
height, width, layers = frames[0].shape
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
video = cv.VideoWriter(output_path, fourcc, 24, (width, height))

for frame in frames:
    video.write(frame)
video.release()'''
