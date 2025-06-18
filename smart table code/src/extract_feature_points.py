import numpy as np
import cv2 as cv
import customtkinter as ctk
import os
import time
from natsort import natsorted
from plantcv import plantcv as pcv
import re
# GUI to select a folder
def TRACKMOTION(img_folder, delete_bin_files=False):
    """
    Track motion in a sequence of images and save the motion features to CSV files.
    
    Parameters:
    img_folder (str): Path to the folder containing images.
    """
    output_dir = None # Initialize output_dir

    if os.path.basename(img_folder) == 'processed_images':
        print("DEBUG TRACKMOTION: img_folder IS 'processed_images'. Calculating versioned output_dir.")
        parent_dir = os.path.dirname(img_folder)
        print(f"DEBUG TRACKMOTION: Parent directory: {parent_dir}")

        num_ext = 1 # Default to 1 if no existing numbered folders are found

        # List all items in the parent directory to find existing 'motion_features_X' folders
        existing_motion_dirs = []
        try:
            for item in os.listdir(parent_dir):
                if os.path.isdir(os.path.join(parent_dir, item)):
                    match = re.match(r'motion_features_(\d+)', item)
                    if match:
                        existing_motion_dirs.append(int(match.group(1)))
        except Exception as e:
            print(f"ERROR: Could not list directory {parent_dir} to find existing motion features: {e}")

        print(f"DEBUG TRACKMOTION: Found existing motion_features versions: {existing_motion_dirs}")

        if existing_motion_dirs:
            num_ext = max(existing_motion_dirs) + 1
        
        output_dir = os.path.join(parent_dir, f'motion_features_{num_ext}')
        print(f"DEBUG TRACKMOTION: Determined output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # If the input img_folder itself is 'processed_images',
        # ensure subsequent image processing logic correctly targets the images INSIDE this folder.
        # This modification of img_folder is for internal use within TRACKMOTION only.
        # If your image loading logic already handles this correctly, you might not need this line here.
        # It depends on how the rest of TRACKMOTION uses 'img_folder'.
        # For clarity, let's assume img_folder means the directory containing the actual images.
        # If it's already 'processed_images', then it's fine.
        # If your TRACKMOTION's core image loading loop looks for images directly in `img_folder`,
        # then this branch is already set up correctly to find those.

    else:
        print("DEBUG TRACKMOTION: img_folder IS NOT 'processed_images'. Using default 'motion_features'.")
        output_dir = os.path.join(img_folder, 'motion_features')
        print(f"DEBUG TRACKMOTION: Determined output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # In this branch, `img_folder` (e.g., 'Plant1') is the parent of 'processed_images'.
        # So, the images to process are likely inside 'img_folder/processed_images'.
        # This line ensures the image loading part of TRACKMOTION looks in the correct subfolder.
        # Only modify `img_folder` if it's used internally to point to where the images are.
        img_folder = os.path.join(img_folder, 'processed_images')
        print(f"DEBUG TRACKMOTION: Adjusted img_folder for image scanning: {img_folder}")
    
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
    fade_trails = True   # Set to False for no fading
    fade_alpha = 0.1     # 0.0 = no fade, 1.0 = full fade each frame

    frames = [] # Store frames for video creation

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
        frames.append(img_display.copy())  # Store frame for video creation
        cv.imshow('frame', img_display)
        k = cv.waitKey(10) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()

    cv.destroyAllWindows()

    output_path = os.path.join(output_dir, 'PlantMotionFade.mp4')
    height, width, layers = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    video = cv.VideoWriter(output_path, fourcc, 24, (width, height))



    '''for frame in frames:
        video.write(frame)
    video.release()'''
    print(output_dir)
    return os.path.normpath(output_dir)
'''
root = ctk.CTk()
root.withdraw()
folder = ctk.filedialog.askdirectory()
TRACKMOTION(folder)'''

if __name__ == "__main__":
    pass
