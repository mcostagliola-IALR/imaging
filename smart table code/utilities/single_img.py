from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt

root = ctk.CTk()
root.withdraw()
folder_path = ctk.filedialog.askopenfilename()

def ExtractLeafTips(inDirectoryPath) -> str:
    img_path = inDirectoryPath
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


    root.update()
    ext = os.path.splitext(img_path)[-1].lower()
    # Process each image as before
    args = WorkflowInputs(
        images=[img_path],
        names=img_path,
        result='',
        writeimg=True,
        outdir='',
        debug='',
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

    lab = pcv.rgb2gray_lab(output_g, 'a')
    #fin_img = pcv.hist_equalization(gray_img=lab)
    binl = pcv.threshold.otsu(gray_img=lab, object_type='dark')
    mask1l = pcv.closing(gray_img=binl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
    mask2l = pcv.fill(mask1l, 300)
    mask2cl = pcv.opening(mask2l)
    mask1l = pcv.closing(gray_img=mask2cl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
    mask2l = pcv.fill(mask1l, 300)
    mask1l = pcv.closing(gray_img=mask2l, kernel=np.array([[1, 0, 1],[0, 1, 0],[1, 0, 1]]))

    #con = np.concatenate((fin_img, mask1l), axis=1)

    cv2.imshow(None, mask1l)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ExtractLeafTips(folder_path)