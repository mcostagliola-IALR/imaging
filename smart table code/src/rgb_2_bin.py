from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2

def RGB2BIN(inDirectoryPath) -> str:

    extensions = ('.jpg', '.png')

    img_path = inDirectoryPath

    outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images')
    os.makedirs(outDirectoryPath, exist_ok=True)

    print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')

    image_files = pcv.io.read_dataset(inDirectoryPath)

    for img_path in image_files:
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
        #pcv.params.debug = 'plot'
        img, filename, path = pcv.readimage(filename=img_path)

        hh, ww = img.shape[:2]
        maxdim = max(hh, ww)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        sigma = int(5 * maxdim / 300)
        gaussian = cv2.GaussianBlur(y, (3, 3), sigma, sigma)
        y = (y - gaussian + 100)
        ycrcb = cv2.merge([y, cr, cb])

        #convert to BGR
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        output_g = pcv.transform.gamma_correct(img=output, gamma=2.5, gain=5)
        output_g = pcv.gaussian_blur(img=output_g, ksize=(5, 5), sigma_x=0, sigma_y=None)

        lab = pcv.rgb2gray_lab(output_g, 'a')
        binl = pcv.threshold.otsu(gray_img=lab, object_type='dark')
        mask1l = pcv.closing(gray_img=binl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2l = pcv.fill(mask1l, 300)
        mask2cl = pcv.opening(mask2l)
        mask1l = pcv.closing(gray_img=mask2cl, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        mask2l = pcv.fill(mask1l, 300)
        mask1l = pcv.closing(gray_img=mask2l, kernel=np.array([[1, 0, 1],[0, 1, 0],[1, 0, 1]]))

        filename = os.path.basename(img_path)
        pcv.print_image(mask1l, rf'{outDirectoryPath}\{filename}')

    return outDirectoryPath

if __name__ == "__main__":
# Example usage (replace with a real path if you want to test directly)
# TRACKMOTION(r"C:\path\to\your\images")
    pass