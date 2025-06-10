import numpy as np
import cv2 as cv
import customtkinter as ctk
import os

root = ctk.CTk()
img_path = ctk.filedialog.askopenfilename()
output_dir = os.path.join(os.path.dirname(img_path), 'motion_features')
os.makedirs(output_dir, exist_ok=True)
root.withdraw()

cap = cv.VideoCapture(img_path)

feature_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Assign a unique ID to each initial feature
feature_ids = np.arange(len(p0)).tolist()  # [0, 1, 2, ..., N-1]

mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Only keep features that are still tracked
    if p1 is not None:
        st = st.flatten()
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        good_ids = [fid for fid, alive in zip(feature_ids, st) if alive == 1]

        # Draw and write only for features that are still tracked
        for i, (new, old, fid) in enumerate(zip(good_new, good_old, good_ids)):
            a, b = new.ravel()
            c, d = old.ravel()
            with open(os.path.join(output_dir, f"feature_{fid}.csv"), "a") as f:
                f.write(f"{a},{b}\n")
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[fid % len(color)].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[fid % len(color)].tolist(), -1)

        # Update only the tracked features and their IDs
        p0 = good_new.reshape(-1, 1, 2)
        feature_ids = good_ids

    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()

cv.destroyAllWindows()