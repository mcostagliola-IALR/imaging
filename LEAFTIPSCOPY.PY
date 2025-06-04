from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import csv

root = ctk.CTk()
folder_path = ctk.filedialog.askdirectory()

def find_leaf_tips(skeleton):
    """Find tip coordinates from a skeleton image."""
    original_sample_label = pcv.params.sample_label
    pcv.params.sample_label = 'default'
    tips_mask = pcv.morphology.find_tips(skel_img=skeleton)
    pcv.params.sample_label = original_sample_label

    tip_points = np.column_stack(np.where(tips_mask > 0))
    print("\nFound Tip Coordinates (x, y) on PRUNED skeleton:")
    for tip_y, tip_x in tip_points:
        print(f"  ({tip_x}, {tip_y})")

    return tip_points


def recursive_prune(skeleton, max_iterations=5, size=150):
    """Recursively prune the skeleton to avoid 'too many tips' error."""
    pcv.params.sample_label = 'default'
    pruned_skeleton = skeleton
    for i in range(max_iterations):
        pruned_skeleton, segmented_img, obj = pcv.morphology.prune(skel_img=pruned_skeleton, size=size)
        try:
            # Try calculating lengths to check if pruning is sufficient
            pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=obj)
            # No error means pruning is fine
            return pruned_skeleton, segmented_img, obj
        except RuntimeError as e:
            if "Too many tips found per segment" in str(e):
                print(f"Pruning iteration {i+1}: too many tips, pruning again...")
                continue
            else:
                raise e
    print("Warning: Maximum pruning iterations reached, results may be unreliable.")
    return pruned_skeleton, segmented_img, obj

def ExtractLeafTips(inDirectoryPath):
    tips_all = []


    image_files = pcv.io.read_dataset(inDirectoryPath)
    for img_path in image_files:
        extensions = ('.jpg', '.png')
        ext = os.path.splitext(img_path)[-1].lower()
        if ext not in extensions:
            print(f"Unsupported file type for {img_path}, skipping.")
            continue

        output_dir = os.path.join(os.path.dirname(img_path), "plantcv_results")
        os.makedirs(output_dir, exist_ok=True)

        name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        results_filename = os.path.join(output_dir, f"{name_without_ext}_results.csv")

        args = WorkflowInputs(
            images=[img_path],
            names=img_path,
            result=results_filename,
            writeimg=True,
            debug='none',
            sample_label=''
        )

        pcv.params.line_thickness = 2
        pcv.params.debug = args.debug
        pcv.params.sample_label = 'plant'

        img, filename, path = pcv.readimage(filename=img_path)

        hh, ww = img.shape[:2]
        maxdim = max(hh, ww)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        sigma = int(5 * maxdim / 300)
        gaussian = cv2.GaussianBlur(y, (3, 3), sigma, sigma)
        y = (y - gaussian + 100)
        ycrcb = cv2.merge([y, cr, cb])
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        corrected_img2 = pcv.white_balance(img=output, mode='hist', roi=[5, 5, 80, 80])

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected_img2, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(
            gray_img=bin_mask1_2,
            kernel=np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]])
        )
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_2 = pcv.opening(clean_mask1_2)

        # --- SKELETONIZE ---
        skeleton = pcv.morphology.skeletonize(clean_mask1_2)

        # --- RECURSIVE PRUNE ---
        pruned_skeleton, segmented_img, obj = recursive_prune(skeleton, max_iterations=10, size=70)

        # --- SWITCH BACK to 'plant' ---
        pcv.params.sample_label = 'plant'

        # === TIP FINDING ===
        tip_points = find_leaf_tips(pruned_skeleton)
        tips_all.append((name_without_ext, tip_points))


    # Save tips to CSV
    tips_csv_path = os.path.join(inDirectoryPath, "all_tips.csv")
    with open(tips_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Name", "Tip_X", "Tip_Y"])
        for img_name, tip_points in tips_all:
            for tip_y, tip_x in tip_points:
                writer.writerow([img_name, tip_x, tip_y])
    print(f"All tip points saved to {tips_csv_path}")


    return tips_all

if __name__ == "__main__":
    tips= ExtractLeafTips(folder_path)
    print("Processing complete.")
