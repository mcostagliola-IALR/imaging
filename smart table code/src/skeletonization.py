import dearpygui.dearpygui as dpg
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
from skimage.morphology import convex_hull_image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# GUI setup

dpg.create_context()
selected_path = None

def callback(sender, app_data):
    global selected_path
    if isinstance(app_data, dict):
        if 'file_path_name' in app_data:
            selected_path = app_data['file_path_name']
            print("Selected file:", selected_path)
        elif 'file_path' in app_data:
            selected_path = app_data['file_path']
            print("Selected directory:", selected_path)
        dpg.stop_dearpygui()

with dpg.file_dialog(directory_selector=True, show=False, callback=callback, tag="file_dialog_tag", width=700, height=400):
    dpg.add_file_extension(".py", color=(0, 255, 0, 255))
    dpg.add_file_extension(".csv", color=(255, 0, 0, 255))

with dpg.window(label="Tutorial", width=800, height=300):
    dpg.add_button(label="File/Directory Selector", callback=lambda: dpg.show_item("file_dialog_tag"))

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

# Paths
outDirectoryPathSkel = os.path.join(selected_path, 'processed_images_skel')
os.makedirs(outDirectoryPathSkel, exist_ok=True)
outDirectoryPath = os.path.join(selected_path, 'convex_hull_images_2')
os.makedirs(outDirectoryPath, exist_ok=True)

img_lst = pcv.io.read_dataset(selected_path)
extensions = ('.jpg', '.png')

all_vertices_csv = os.path.join(selected_path, "all_hull_vertices.csv")
with open(all_vertices_csv, "w", newline="") as all_csvfile:
    writer = csv.writer(all_csvfile)
    writer.writerow(["Image_Name", "Tip_X", "Tip_Y"])

    for img_name in tqdm(img_lst, desc="Processing images"):
        ext = os.path.splitext(img_name)[-1].lower()
        if ext in extensions:
            args = WorkflowInputs(
                images=[img_name],
                names=img_name,
                result='',
                writeimg=True,
                outdir='',
                debug='',
                sample_label=''
            )
            pcv.params.line_thickness = 2
            pcv.params.debug = args.debug
            img, filename, path = pcv.readimage(filename=img_name)

            hh, ww = img.shape[:2]
            maxdim = max(hh, ww)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            sigma = int(5 * maxdim / 300)
            gaussian = cv2.GaussianBlur(y, (3, 3), sigma, sigma)
            y = (y - gaussian + 100)
            ycrcb = cv2.merge([y, cr, cb])
            output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            output_g = pcv.transform.gamma_correct(img=output, gamma=2.5, gain=5)
            output_g = pcv.gaussian_blur(img=output_g, ksize=(5, 5), sigma_x=0, sigma_y=None)

            lab = pcv.rgb2gray_lab(output_g, 'a')
            binl = pcv.threshold.otsu(gray_img=lab, object_type='dark')
            mask1l = pcv.closing(gray_img=binl, kernel=np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]))
            mask2l = pcv.fill(mask1l, 300)
            mask2cl = pcv.opening(mask2l)
            mask1l = pcv.closing(gray_img=mask2cl, kernel=np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]))
            mask2l = pcv.fill(mask1l, 300)
            mask1l = pcv.closing(gray_img=mask2l, kernel=np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
            mask1l_inv = cv2.bitwise_not(mask1l)

            binary = (mask1l > 0).astype(np.uint8)
            img_gray = mask1l
            ys, xs = np.nonzero(binary)
            points = np.column_stack((xs, ys))
            chull_skimage = convex_hull_image(binary)
            chull_skimage_uint8 = chull_skimage.astype(np.uint8) * 255
            contours, _ = cv2.findContours(chull_skimage_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            skimage_hull_vertices = []

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour, returnPoints=True)
                epsilon = 0.005 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                skimage_hull_vertices = approx.reshape(-1, 2)

                M = cv2.moments(binary)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = np.array([cx, cy])
                else:
                    centroid = np.mean(skimage_hull_vertices, axis=0)

                if len(skimage_hull_vertices) >= 2:
                    clustering = DBSCAN(eps=30, min_samples=1).fit(skimage_hull_vertices)
                    labels = clustering.labels_
                    unique_labels = np.unique(labels)
                    filtered_tip_points = []
                    for label in unique_labels:
                        cluster_points = skimage_hull_vertices[labels == label]
                        if len(cluster_points) > 0:
                            dists = [np.linalg.norm(pt - centroid) for pt in cluster_points]
                            tip = cluster_points[np.argmax(dists)]
                            filtered_tip_points.append(tip)
                    filtered_tip_points = np.array(filtered_tip_points)
                else:
                    filtered_tip_points = skimage_hull_vertices

                filename_no_ext = os.path.splitext(os.path.basename(img_name))[0]
                for vx, vy in filtered_tip_points:
                    writer.writerow([filename_no_ext, int(vx), int(vy)])

            plt.figure(figsize=(img_gray.shape[1]/80, img_gray.shape[0]/80))
            plt.imshow(img_gray, cmap='gray', origin='upper')
            plt.plot(points[:, 0], points[:, 1], 'w.', markersize=1, alpha=0.2)
            plt.imshow(chull_skimage, cmap='hot', alpha=0.3)

            if len(skimage_hull_vertices) > 0:
                plt.plot(skimage_hull_vertices[:, 0], skimage_hull_vertices[:, 1], 'bo', markersize=6)
                plt.plot(np.append(skimage_hull_vertices[:, 0], skimage_hull_vertices[0, 0]),
                         np.append(skimage_hull_vertices[:, 1], skimage_hull_vertices[0, 1]), 'b--', linewidth=1)

            plt.title("Convex Hull Overlay")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.axis("on")
            plt.gca().set_aspect('equal', adjustable='box')

            basename = os.path.basename(img_name)
            filename_no_ext = os.path.splitext(basename)[0]
            save_path = os.path.join(outDirectoryPath, f'convex_hull_overlay_{filename_no_ext}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
