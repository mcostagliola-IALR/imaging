from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt

root = ctk.CTk()
folder_path = ctk.filedialog.askopenfilename()

def ExtractLeafTips(inDirectoryPath) -> str:



    extensions = ('.jpg', '.png')

    img_path = inDirectoryPath

    #outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images').replace("/", "\\") # Elimintes confusion with python escape codes
    #os.makedirs(outDirectoryPath, exist_ok=True)
    #print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')
    #image_files = pcv.io.read_dataset(inDirectoryPath)

    ext = os.path.splitext(img_path)[-1].lower()
    if ext in extensions:
        # Process each image as before
        args = WorkflowInputs(
            images=[img_path],
            names=img_path,
            result='',
            writeimg=True,
            debug='none',
            sample_label=''
        )
        pcv.params.line_thickness = 2

        # Debug Params
        pcv.params.debug = args.debug
        img, filename, path = pcv.readimage(filename=img_path)

        hh, ww = img.shape[:2]
        maxdim = max(hh, ww)

        # illumination normalize
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # separate channels
        y, cr, cb = cv2.split(ycrcb)

        # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
        # account for size of input vs 300
        sigma = int(5 * maxdim / 300)
        #print('sigma: ',sigma)
        gaussian = cv2.GaussianBlur(y, (3, 3), sigma, sigma)

        # subtract background from Y channel
        y = (y - gaussian + 100)

        # merge channels back
        ycrcb = cv2.merge([y, cr, cb])

        #convert to BGR
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        corrected_img2 = pcv.white_balance(img=output, mode='hist')
        corrected_img2 = pcv.white_balance(img=corrected_img2, mode='hist')

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected_img2, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_2 = pcv.opening(clean_mask1_2)

        #pcv.params.sample_label = "plant"

        skeleton = pcv.morphology.skeletonize(clean_mask1_2)
        pruned_skeleton, segmented_img, obj = pcv.morphology.prune(skel_img=skeleton, size=100)
        tips_coords = pcv.morphology.find_tips(skel_img=pruned_skeleton)
        euclidean_distance = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=obj)

        

        #euclidean_lengths = pcv.outputs.observations['plant']['segment_eu_length']['value']
        #print(euclidean_lengths)

        shape = pcv.analyze.size(corrected_img2, bin_mask1_2, 1)
        color_scatter, _ = pcv.visualize.pixel_scatter_plot(paths_to_imgs = [img_path] , x_channel = "a", y_channel ="b" )
        hori2 = np.concatenate((pruned_skeleton, tips_coords), axis=1)

        clean_mask_3dim = np.stack([clean_mask1_2, clean_mask1_2, clean_mask1_2], axis=-1)
        hori1 = np.concatenate((corrected_img2, clean_mask_3dim), axis=1)
        hori2 = np.stack([hori2, hori2, hori2], axis=-1)
        final_img = np.concatenate((hori1, hori2), axis=0)

        cv2.imshow(None, final_img)
        cv2.waitKey(0)
        cv2.imshow(None, shape)
        cv2.waitKey(0)
        cv2.imshow(None, euclidean_distance)
        cv2.waitKey(0)
         # Convert the Matplotlib Figure (color_scatter) to a NumPy array for OpenCV display
        if isinstance(color_scatter, plt.Figure):
            color_scatter.canvas.draw()
            rgba_image = np.array(color_scatter.canvas.renderer.buffer_rgba())
            # Convert RGBA to BGR and remove alpha channel for OpenCV
            color_scatter_np = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Pixel Scatter Plot", color_scatter_np)
            cv2.waitKey(0)
            plt.close(color_scatter) # Close the Matplotlib figure
        else:
            print("Warning: color_scatter was not a Matplotlib Figure.")



        cv2.destroyAllWindows()


ExtractLeafTips(folder_path)
