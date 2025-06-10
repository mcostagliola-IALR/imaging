import os
from PIL import Image
import numpy as np
import cv2
import xlsxwriter
folder = r"C:\Users\dbrimmer\Downloads\Plant1_Peppermovement"

lower_hsv = np.array([20, 50, 50])    # Yellowish-green lower
upper_hsv = np.array([85, 255, 255])  # Deep green upper

image_files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
folder_name = os.path.basename(folder)
workbook = xlsxwriter.Workbook(folder_name + '.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write(0, 0, "Filename")
worksheet.write(0, 1, "Plant Pixel Count")
worksheet.write(0, 2, "Plant Pixel Percentage")

row = 1
for filename in image_files:
    path = os.path.join(folder, filename)
    image_bgr = cv2.imread(path)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Create mask for green/yellow pixels
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    pixel_count = cv2.countNonZero(mask)
    total_pixels = mask.size
    percentage = (pixel_count / total_pixels) * 100

    print(f"{filename}: {pixel_count} plant pixels ({percentage:.2f}%)")
    worksheet.write(row, 0, filename)
    worksheet.write(row, 1, pixel_count)
    worksheet.write(row, 2, percentage)
    row += 1
workbook.close()





