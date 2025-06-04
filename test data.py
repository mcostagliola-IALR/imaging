from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk

root = ctk.CTk()
folder_path = ctk.filedialog.askopenfilename()

def ProcessImagesToBinary(inDirectoryPath) -> str:



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

        corrected_img1 = pcv.white_balance(img=img, mode='hist', roi=[5, 5, 80, 80])
        corrected_img2 = pcv.white_balance(img=output, mode='hist', roi=[5, 5, 80, 80])

        # Image Processing - example operations
        a_gray = pcv.rgb2gray_lab(rgb_img=output, channel='a')
        bin_mask1 = pcv.threshold.otsu(gray_img=a_gray, object_type='dark')
        bin_mask2 = pcv.threshold.triangle(gray_img=a_gray, object_type='dark')
        clean_mask1 = pcv.closing(gray_img=bin_mask1, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        clean_mask2 = pcv.closing(gray_img=bin_mask2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))

        a_gray1 = pcv.rgb2gray_lab(rgb_img=corrected_img1, channel='a')
        bin_mask1_1 = pcv.threshold.otsu(gray_img=a_gray1, object_type='dark')
        bin_mask2_1 = pcv.threshold.triangle(gray_img=a_gray1, object_type='dark')
        clean_mask1_1 = pcv.closing(gray_img=bin_mask1_1, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        clean_mask2_1 = pcv.closing(gray_img=bin_mask2_1, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected_img2, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        bin_mask2_2 = pcv.threshold.triangle(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        clean_mask2_2 = pcv.closing(gray_img=bin_mask2_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))


        size_img1 = np.count_nonzero(clean_mask1)
        size_img2 = np.count_nonzero(clean_mask2)
        size_img1_1 = np.count_nonzero(clean_mask1_1)
        size_img2_1 = np.count_nonzero(clean_mask2_1)
        size_img1_2 = np.count_nonzero(clean_mask1_2)
        size_img2_2 = np.count_nonzero(clean_mask2_2)


        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1
        
        # Blue color in BGR
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.putText() method
        clean_mask1 = cv2.putText(clean_mask1, f'No Correction, Pixel Count:{size_img1}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        clean_mask2 = cv2.putText(clean_mask2, f'No Correction, Pixel Count:{size_img2}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        clean_mask1_1 = cv2.putText(clean_mask1_1, f'Pixel Count:{size_img1_1}\n Otsu Thresholding', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        clean_mask2_1 = cv2.putText(clean_mask2_1, f'Pixel Count:{size_img2_1}, \n Triangle Thresholding', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        clean_mask1_2 = cv2.putText(clean_mask1_2, f'WCPG Pixel Count:{size_img1_2}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        clean_mask2_2 = cv2.putText(clean_mask2_2, f'WCPG Pixel Count:{size_img2_2}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        corrected_img1 = cv2.putText(corrected_img1, 'White Corrected Pre-Gaussian', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        corrected_img2 = cv2.putText(corrected_img2, 'White Corrected Post-Gaussian', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        clean_mask1_dim3 = np.stack([clean_mask1, clean_mask1, clean_mask1], axis=-1)
        clean_mask2_dim3 = np.stack([clean_mask2, clean_mask2, clean_mask2], axis=-1)
        clean_mask1_1_dim3 = np.stack([clean_mask1_1, clean_mask1_1, clean_mask1_1], axis=-1)
        clean_mask2_1_dim3 = np.stack([clean_mask2_1, clean_mask2_1, clean_mask2_1], axis=-1)
        clean_mask1_2_dim3 = np.stack([clean_mask1_2, clean_mask1_2, clean_mask1_1], axis=-1)
        clean_mask2_2_dim3 = np.stack([clean_mask2_2, clean_mask2_2, clean_mask2_1], axis=-1)

        hori1 = np.concatenate((clean_mask1, clean_mask2), axis=1)
        hori2 = np.concatenate((clean_mask1_1, clean_mask2_1), axis=1)
        hori3 = np.concatenate((clean_mask1_2, clean_mask2_2), axis=1)
        hori4 = np.concatenate((corrected_img1, corrected_img2), axis=1)
        hori_13 = np.concatenate((hori1, hori3), axis=0)
        #hori_23 = np.concatenate((hori2, hori3), axis=0)
        cv2.imshow(None , hori_13)
        #cv2.imshow(None , hori_23)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        



ProcessImagesToBinary(folder_path)