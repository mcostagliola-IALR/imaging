from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import time
import pandas as pd

def RGB2BIN(inDirectoryPath) -> str:
    data = []
    extensions = ('.jpg', '.png')

    outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images')
    os.makedirs(outDirectoryPath, exist_ok=True)

    print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')

    image_files = pcv.io.read_dataset(inDirectoryPath)

    for img_path in image_files:
        ext = os.path.splitext(img_path)[-1].lower()

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
        pcv.params.debug = args.debug

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

        size = np.count_nonzero(mask1l)

        try:
            # Remove extension
            base = os.path.splitext(os.path.basename(path))[0]  # "03-21-2024_08-58-Rep1"

            # Split on "-Rep"
            datetime_part, rep_part = base.split("-Rep")  # → "03-21-2024_08-58", "1"

            # Split date and time
            date_part, time_part = datetime_part.split("_")  # → "03-21-2024", "08-58"

            # Format for output
            file_date = time.strftime("%m/%d/%Y", time.strptime(date_part, "%m-%d-%Y"))  # → "03/21/2024"
            file_time = time_part.replace("-", ":")  # → "08:58"
            file_name = f"{file_date} {file_time}"
            rep = rep_part
        except Exception as e:
            print(f"Filename parsing failed for '{path}': {e}")
            file_name = ""
            rep = ""


        # ✅ FIXED: Correct usage of append
        data.append([file_name, rep, size])

        out_file_path = os.path.join(outDirectoryPath, os.path.basename(img_path))
        pcv.print_image(mask1l, out_file_path)

    # ✅ Convert data to DataFrame
    df = pd.DataFrame(data, columns=["datetime", "rep", "size"])
    excel_path = os.path.join(inDirectoryPath, "mask_sizes.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"Data written to Excel: {excel_path}")
    return outDirectoryPath

if __name__ == "__main__":
    # RGB2BIN(r"C:\your\image\folder")
    pass
