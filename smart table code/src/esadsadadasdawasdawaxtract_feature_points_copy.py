import numpy as np
import cv2 as cv
import customtkinter as ctk
import os
import time
from natsort import natsorted
from plantcv import plantcv as pcv
import shutil

def TRACKMOTION(img_folder, delete_bin_files=False):
    if os.path.basename(img_folder) == 'processed_images':
        parent_dir = os.path.dirname(img_folder)
        output_dir = os.path.join(parent_dir, 'motion_features')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(img_folder, 'motion_features')
        img_folder = os.path.join(img_folder, 'processed_images')
        os.makedirs(output_dir, exist_ok=True)

    img_lst = [
        os.path.normpath(os.path.join(img_folder, f))
        for f in os.listdir(img_folder)
        if os.path.isfile(os.path.join(img_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    img_lst = natsorted(img_lst)

    if len(img_lst) < 2:
        raise ValueError("Need at least two images to track motion.")

    feature_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    old_frame = cv.imread(img_lst[0])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)
    fade_trails = True
    fade_alpha = 0.1

    frames = []

    for i in range(1, len(img_lst)):
        new_frame = cv.imread(img_lst[i])
        new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # Get features in the previous (old) frame
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        if p0 is not None:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

            if p1 is not None:
                st = st.flatten()
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for fid, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    filename = os.path.basename(img_lst[i])

                    try:
                        base = filename.split("-Rep")[0]
                        date_part, time_part = base.split("_")
                        file_date = time.strftime("%m/%d/%Y", time.strptime(date_part, "%m-%d-%Y"))
                        file_time = time_part.replace("-", ":")
                        file_name = file_date + " " + file_time
                        rep_part = filename.split("-Rep")[1]
                        rep = rep_part.split(".")[0]
                    except Exception:
                        file_date = ""
                        file_time = ""
                        rep = ""

                    with open(os.path.join(output_dir, f"feature_{fid}.csv"), "a") as f:
                        f.write(f"{a},{b},{file_name},{rep}\n")

                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)),
                                   color[fid % len(color)].tolist(), 2)
                    new_frame = cv.circle(new_frame, (int(a), int(b)), 5,
                                          color[fid % len(color)].tolist(), -1)

        if fade_trails:
            mask = cv.addWeighted(mask, 1 - fade_alpha, np.zeros_like(mask), 0, 0)

        img_display = cv.add(new_frame, mask)
        frames.append(img_display.copy())
        cv.imshow('frame', img_display)
        k = cv.waitKey(10) & 0xff
        if k == 27:
            break

        old_gray = new_gray.copy()

    cv.destroyAllWindows()

    output_path = os.path.join(output_dir, 'PlantMotionFade.mp4')
    height, width, layers = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(output_path, fourcc, 24, (width, height))

    # Optional: uncomment to write video
    # for frame in frames:
    #     video.write(frame)
    # video.release()

    print(output_dir)
    return output_dir
if __name__ == '__main__':
    pass