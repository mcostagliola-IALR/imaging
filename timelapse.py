from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt

root = ctk.CTk()
path = ctk.filedialog.askdirectory()
img_list = pcv.io.read_dataset(path)

pcv.visualize.time_lapse_video(img_list=img_list, out_filename=os.path.basename(os.path.normpath(path)), fps=24)