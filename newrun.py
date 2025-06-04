from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt

root = ctk.CTk()
root.withdraw()
folder_path = ctk.filedialog.askdirectory()

def ExtractLeafTips(inDirectoryPath) -> str:

        # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 255, 0)

    # Line thickness of 2 px
    thickness = 2

    extensions = ('.jpg', '.png')

    img_path = inDirectoryPath

    outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images').replace("/", "\\") # Elimintes confusion with python escape codes
    os.makedirs(outDirectoryPath, exist_ok=True)
    print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')
    #image_files = pcv.io.read_dataset(inDirectoryPath)

    image_files = pcv.io.read_dataset(inDirectoryPath)

    for img_path in image_files:
        root.update()
        ext = os.path.splitext(img_path)[-1].lower()
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
        output_g = pcv.transform.gamma_correct(img=output, gamma=2.5, gain=5)
        output_g = pcv.gaussian_blur(img=output_g, ksize=(5, 5), sigma_x=0, sigma_y=None)

        '''cmyk = pcv.rgb2gray_cmyk(output_g, 'y')
        cmyk = pcv.white_balance(cmyk, mode='hist')
        cmyk = pcv.opening(cmyk)'''
        #cmyk = pcv.gaussian_blur(img=cmyk, ksize=(51, 51), sigma_x=0, sigma_y=None)
        '''        cmyk = pcv.invert(cmyk)
        binc = pcv.threshold.otsu(gray_img=cmyk, object_type='dark')
        mask1c = pcv.closing(gray_img=binc, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2c = pcv.fill(mask1c, 300)
        mask2cc = pcv.opening(mask2c)
        mask2cc = pcv.closing(gray_img=mask2cc, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2c = pcv.fill(mask2cc, 300)'''

        lab = pcv.rgb2gray_lab(output_g, 'a')
        binl = pcv.threshold.otsu(gray_img=lab, object_type='dark')
        mask1l = pcv.closing(gray_img=binl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2l = pcv.fill(mask1l, 300)
        mask2cl = pcv.opening(mask2l)
        mask1l = pcv.closing(gray_img=mask2cl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2l = pcv.fill(mask1l, 300)
        mask1l = pcv.closing(gray_img=mask2l, kernel=np.array([[1, 0, 1],[0, 1, 0],[1, 0, 1]]))
        #mask1l = pcv.gaussian_blur(img=mask1l, ksize=(51, 51), sigma_x=0, sigma_y=None)
        pixl = np.count_nonzero(mask2l)


        #pixc = np.count_nonzero(cmyk)
        #print(f'Plant Size CMYK: {pixc} ({pixc/(640*480)*100:0.2f}%)', "-"*5, f'Plant Size LAB: {pixl} ({pixl/(640*480)*100:0.2f}%)', "-"*5, f'Difference: {abs(pixc-pixl)}')
        print(f'Plant Size CIELAB: {pixl} ({pixl/(640*480)*100:0.2f}%)')
        
        pixl = cv2.putText(mask2cl, f'CIELAB {pixl}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        pixl = np.stack([pixl, pixl, pixl], axis=-1)
        hori = np.concatenate((pixl, output), axis=1)

        filename = os.path.basename(img_path)
        pcv.print_image(hori, rf'{outDirectoryPath}\{filename}')

    img_list = pcv.io.read_dataset(outDirectoryPath)

    print(outDirectoryPath)
    print(os.path.basename(os.path.normpath(outDirectoryPath)))
    pcv.visualize.time_lapse_video(img_list=img_list, out_filename=outDirectoryPath, fps=24)

ExtractLeafTips(folder_path)