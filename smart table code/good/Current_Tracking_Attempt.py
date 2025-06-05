import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull # Still useful for comparing if needed
import customtkinter as ctk
from skimage.morphology import convex_hull_image
from skimage import measure # For finding contours
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import os
from tqdm import tqdm  # <-- Add tqdm for progress bar
import csv

# --- File Dialog and Image Loading ---
root = ctk.CTk()
root.withdraw()
path = ctk.filedialog.askdirectory()

outDirectoryPath = os.path.join(path, 'convex_hulls').replace("/", "\\") # Elimintes confusion with python escape codes
os.makedirs(outDirectoryPath, exist_ok=True)
os.chdir(outDirectoryPath)
img_lst = pcv.io.read_dataset(path)
extensions = ('.jpg', '.png')

all_vertices_csv = os.path.join(outDirectoryPath, "all_hull_vertices.csv")
with open(all_vertices_csv, "w", newline="") as all_csvfile:
    writer = csv.writer(all_csvfile)
    writer.writerow(["Image_Name", "Tip_X", "Tip_Y"])

    for img_name in tqdm(img_lst, desc="Processing images"):
        ext = os.path.splitext(img_name)[-1].lower()
        if ext in extensions:
            img_path = os.path.join(path, img_name)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Failed to load: {img_path}")
                continue

            binary = (img_gray > 250).astype(np.uint8)
            ys, xs = np.nonzero(binary)
            points = np.column_stack((xs, ys))

            chull_skimage = convex_hull_image(binary)
            chull_skimage_uint8 = chull_skimage.astype(np.uint8) * 255
            contours, _ = cv2.findContours(chull_skimage_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            skimage_hull_vertices = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour, returnPoints=True)
                # Simplify the hull to reduce the number of vertices
                epsilon_ratio = 0.02  # Adjust this value for more/less simplification
                epsilon = epsilon_ratio * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                skimage_hull_vertices = approx.reshape(-1, 2)

            # Write all hull vertices for this image to the single CSV
            filename_no_ext = os.path.splitext(os.path.basename(img_name))[0]
            for vx, vy in skimage_hull_vertices:
                writer.writerow([filename_no_ext, int(vx), int(vy)])

            print(f"Vertices extracted from skimage.convex_hull_image: {len(skimage_hull_vertices)} points")

        '''plt.figure(figsize=(img_gray.shape[1]/80, img_gray.shape[0]/80))
        plt.imshow(img_gray, cmap='gray', origin='upper')
        plt.plot(points[:, 0], points[:, 1], 'w.', markersize=1, alpha=0.2, label='Skeleton Points')
        plt.imshow(chull_skimage, cmap='hot', alpha=0.3)

        if len(skimage_hull_vertices) > 0:
            plt.plot(skimage_hull_vertices[:, 0], skimage_hull_vertices[:, 1], 'bo', markersize=6, label='Hull Vertices')
            plt.plot(np.append(skimage_hull_vertices[:, 0], skimage_hull_vertices[0, 0]),
                     np.append(skimage_hull_vertices[:, 1], skimage_hull_vertices[0, 1]), 'b--', linewidth=1)

        plt.title("Convex Hull Overlay")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("on")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')

        # Create valid save path using filename only (no directories)
        basename = os.path.basename(img_name)
        filename_no_ext = os.path.splitext(basename)[0]
        save_path = os.path.join(outDirectoryPath, f'convex_hull_overlay_{filename_no_ext}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()'''
