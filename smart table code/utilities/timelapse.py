from plantcv import plantcv as pcv
import os
import customtkinter as ctk

root = ctk.CTk()
path = ctk.filedialog.askdirectory()
img_list = pcv.io.read_dataset(path)

os.chdir(path)

pcv.visualize.time_lapse_video(img_list=img_list, out_filename=os.path.basename(os.path.normpath(path)), fps=24)