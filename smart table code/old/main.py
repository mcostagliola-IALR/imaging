import customtkinter as ctk
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import time
import numpy as np
import os
import time
import xlsxwriter
import shutil
from natsort import natsorted
import sys
import cv2

root = ctk.CTk()

def run_program(path, pathCount):

    if batch_trigger == False:
        folder_path = path_var.get()
    
    elif batch_trigger == True:
        folder_path = path

    extensions = ('.jpg', '.png')
    def ProcessImagesToBinary(inDirectoryPath) -> str:

        outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images').replace("/", "\\") # Elimintes confusion with python escape codes
        os.makedirs(outDirectoryPath, exist_ok=True)
        print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')
        image_files = pcv.io.read_dataset(inDirectoryPath)

        for img_path in image_files:
            root.update()
            ext = os.path.splitext(img_path)[-1].lower()
            if ext in extensions:
                # Process each image as before
                args = WorkflowInputs(
                    images=[img_path],
                    names=img_path,
                    result='',
                    outdir=rf'{outDirectoryPath}',
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

                # Image Processing - example operations
                a_gray = pcv.rgb2gray_lab(rgb_img=output, channel='a')
                bin_mask = pcv.threshold.otsu(gray_img=a_gray, object_type='dark')
                clean_mask = pcv.closing(gray_img=bin_mask, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))

                filename = os.path.basename(img_path)
                pcv.print_image(clean_mask, rf'{outDirectoryPath}\{filename}')

        return outDirectoryPath


    def formatCellInfoFromFilename(filename):
        """
        Extracts the information from the filename to create a clear and more presentable string.

        :param filename: for the current photo, expected to be in the following format: 06-29-2023_13-52-Rep1.jpg
        :return: formatted string with extracted data, ex: 07/08/2023 09:33AM
        """

        try:

            shortenedname = filename.split("-Rep")[0]
            timeinfo = time.strptime(shortenedname, "%m-%d-%Y_%H-%M")  # struct as defined in time module
            return time.strftime("%m/%d/%Y", timeinfo)
        except:
            # formatted name could not be parsed, use filename instead
            return filename


    def writeData(row, col, filename, newPic, oldPic=None):
        """
        Outputs data for a specific plant and rep to the excel sheet.

        :param row: uppermost row of cell range where work is being written
        :param col: leftmost column of cell range where work is being written
        :param filename: name of the file, containing information to extract for cells
        :param newPic: the matrix representation of green pixels for the current plant
        :param oldPic: the matrix representation of green pixels for the previous plant,
            set to None in the case of the first plant
        :return: nothing
        """

        # Writes the rep number
        Size_Worksheet.write(row, col, row - 1)
        Motion_Worksheet.write(row, col, row - 1)

        # Writes the date and time in a more readable format
        Size_Worksheet.write(row, col + 1, formatCellInfoFromFilename(filename))
        Size_Worksheet.set_column(col + 1, col + 1,
                                len(filename))  # Sets col width to fit width of filename, TODO: refactor to use only once...
        Motion_Worksheet.write(row, col + 1, formatCellInfoFromFilename(filename))
        Motion_Worksheet.set_column(col + 1, col + 1, len(filename))

        # writes the Size value for all plants
        Size_Worksheet.write(row, col + 2, np.count_nonzero(newPic))

        # writes the growth and motion values for all plants aside from the first
        if oldPic is None:
            # N/A when there is no previous data to calculate with
            Motion_Worksheet.write(row, col + 2, "N/A")
        else:
            # Counts the number of pixels (under mask) different from previous capture. Done by XOR operation on image matrices
            Motion_Worksheet.write(row, col + 2, np.count_nonzero(newPic ^ oldPic))
            # Counts the difference in pixels between the new capture and the previous capture.


    ###########################
    ## START OF MAIN PROCEDURE
    ###########################
    temp_time = time.time()

    # stores the desired Excel save name
    savename = f"MotionSize_{pathCount}"
    # creates the workbook file w/ savename
    workbook = xlsxwriter.Workbook(savename + '.xlsx')

    inDirectoryPath = ProcessImagesToBinary(folder_path)
    os.chdir(folder_path)

    headerformat = workbook.add_format({
        "bold": 1,
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "fg_color": "yellow",
    })

    pixelsPerPic = 0 # Set after first pic is collected tuple of dimensions (x, y, z = 3)

    Motion_Worksheet = workbook.add_worksheet("Motion_Worksheet")
    Size_Worksheet = workbook.add_worksheet("Size_Worksheet")

    row = 0
    col = 0
    total_picture_count = 0

    oldPic = None  # referenced when determining if we should skip the comparison calculation (for first plant of batch)

    directory = inDirectoryPath
    # Adding headers for Size_Worksheet and Motion_Worksheet
    Size_Worksheet.merge_range(row, col, row, col + 2, directory, headerformat)
    Size_Worksheet.write(row + 1, col, "Num", headerformat)
    Size_Worksheet.write(row + 1, col + 1, "Time", headerformat)
    Size_Worksheet.write(row + 1, col + 2, "Green", headerformat)

    Motion_Worksheet.merge_range(row, col, row, col + 2, directory, headerformat)
    Motion_Worksheet.write(row + 1, col, "Num", headerformat)
    Motion_Worksheet.write(row + 1, col + 1, "Time", headerformat)
    Motion_Worksheet.write(row + 1, col + 2, "Motion", headerformat)

    # Set a fixed column width for the 'Time' column
    Size_Worksheet.set_column(col + 1, col + 1, 20)
    Motion_Worksheet.set_column(col + 1, col + 1, 20)

    row += 2
    # cycle for each image in the folder
    for file in (os.listdir(inDirectoryPath)):
        root.update()
        total_picture_count += 1
        try:

            newPic, _, _ = pcv.readimage(os.path.join(inDirectoryPath, file))
            if newPic is not None:

                # Set pixelsPerPic, only do this for the first pic
                if pixelsPerPic == 0:

                    pixelsPerPic = newPic.shape[0] * newPic.shape[1]  # newPic.shape is a tuple of (height, width, 3(RGB))

                # take out first line below to test speed change
                newIsolatedImage = newPic

                # only evaluates for non-black images in processing
                if np.count_nonzero(newIsolatedImage) > 500:

                    # when reading the first picture, prevents a comparison
                    # attempt with another picture
                    if oldPic is None:

                        writeData(row, col, file, newIsolatedImage)
                    else:

                        writeData(row, col, file, newIsolatedImage, oldIsolatedImage)

                    # add in to test speed change
                    oldIsolatedImage = newIsolatedImage

                    row += 1
                    oldPic = newPic

            else:

                pass
        except Exception as e:

            pass

    # Add chart for Motion_Worksheet
    chart_motion = workbook.add_chart({'type': 'line'}) # Configure type of chart for motion
    chart_motion.set_title({'name': 'Plant Motion Over Time'})
    chart_motion.set_legend({'position': 'none'})

    # Configure the series of the chart from Motion_Worksheet
    chart_motion.add_series({
        'name': '=Motion_Worksheet!$C$1',
        'categories': '=Motion_Worksheet!$B$3:$B$' + str(row),
        'values': '=Motion_Worksheet!$C$3:$C$' + str(row),
        'line': {'width': 1},
    })

    chart_motion.set_x_axis({
        'name': 'Capture Date',
        'name_font': {'size': 18, 'bold': True},
        'date_axis': True,  # Set the x-axis as a date axis
        'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
    })

    chart_motion.set_y_axis({
        'name': 'Pixel Fluctuaton',
        'name_font': {'size': 18, 'bold': True},
    })

    # Insert the chart into the worksheet
    Motion_Worksheet.insert_chart('D2', chart_motion, {'x_scale': 2, 'y_scale': 1.5}) # x_scale & y_scale control size of chart
    # Chart Size can be specificed to a resolution by: {'width': 1920, 'height': 1080}, and then adjusted as needed

    # Add chart for Size_Worksheet
    chart_size = workbook.add_chart({'type': 'line'}) # Configure type of chart for size
    chart_size.set_title({'name': 'Plant Size Over Time'})
    chart_size.set_legend({'position': 'none'})

    # Configure the series of the chart from Size_Worksheet
    chart_size.add_series({
        'name': '=Size_Worksheet!$C$1',
        'categories': '=Size_Worksheet!$B$3:$B$' + str(row),
        'values': '=Size_Worksheet!$C$3:$C$' + str(row),
        'line': {'width': 1},
    })

    chart_size.set_x_axis({
        'name': 'Capture Date',
        'name_font': {'size': 18, 'bold': True},
        'date_axis': True,  # Set the x-axis as a date axis
        'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
    })

    chart_size.set_y_axis({
        'name': 'Pixel Count',
        'name_font': {'size': 18, 'bold': True},
    })

    # Insert the chart into the worksheet
    Size_Worksheet.insert_chart('D2', chart_size, {'x_scale': 2, 'y_scale': 1.5}) # x_scale & y_scale control size of chart
    # Chart Size can be specificed to a resolution by: {'width': 1920, 'height': 1080}, and then adjusted as needed

    workbook.close()

    if save_images == False:
        shutil.rmtree(inDirectoryPath) #deletes the created folder if save_images is set to false

    print(f'~Process Time: {"%0.3f" % (time.time() - temp_time)}\n~Total Time: {"%0.3f" % (time.time()-start)}')
    return total_picture_count

#########################
#~~~~~~~~Widgets~~~~~~~~#
#########################

class DebugWindow:
    def __init__(self, root):
        self.debug_window = ctk.CTkTextbox(master=root, state='disabled', 
                                           border_color='gray40', border_width=2,
                                           fg_color='gray18',  
                                           corner_radius=8, width=600, height=190)
        self.debug_window.place(x=250, y=60)
        # Redirect stdout to the text widget
        
        sys.stdout = TextRedirector(self.debug_window, "stdout")
        sys.stderr = TextRedirector(self.debug_window, "stderr")

    def clear(self):
        '''Clear the contents of the debug window'''
        self.debug_window.configure(state=ctk.NORMAL)  # Enable editing
        self.debug_window.delete('1.0', ctk.END)  # Delete all text
        self.debug_window.configure(state=ctk.DISABLED)  # Disable editing
        self.debug_window.update_idletasks() # Update & display/reflect changes on window (if any)

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state=ctk.NORMAL)  # Enable editing
        self.widget.insert(ctk.END, string, (self.tag,))
        self.widget.see(ctk.END)  # Scroll to the end of the text
        self.widget.configure(state=ctk.DISABLED)  # Disable editing
        self.widget.update_idletasks()  # Update the widget


# Globals
save_images = False
batch_trigger = False

# Widget functionality code -- enables image saving
def save_image_trigger():

    global save_images

    if save_images:
        save_images = False
    else:
        save_images = True
    return save_images


# Widget functionality code -- enables batch processing
def trigger_batch_status():

    global batch_trigger

    if batch_trigger:
        batch_trigger = False
    else:
        batch_trigger = True
    return batch_trigger

# Widget functionality code -- handles directories
def choose_folder():

    folder_path = ctk.filedialog.askdirectory()
    path_var.set(folder_path)

# Widget functionality code -- runs the main function 
def batch():

    global start
    global imCount
    imCount = 0
    pathCount = 1
    start = time.time()

    if batch_trigger == True:

        paths = [os.path.normpath(os.path.join(path_var.get(), f.name)) for f in os.scandir(path_var.get()) if f.is_dir()] # extracts paths from parent folder in proper format
        for path in natsorted(paths): # Natural sorts paths, so that the second path would be "Folder 2", not "Folder 10"
            imCount += run_program(path, pathCount)
            pathCount += 1
        
        print(f"~Processed {imCount} images in {"%0.3f" % (time.time() - start)} seconds.\n~Average Process Time: {"%0.3f" % ((time.time() - start) / len(paths))} seconds.")

    else:
        imCount += run_program(path_var.get(), 0)

        print(f"~Processed {imCount} images in {"%0.3f" % (time.time() - start)} seconds.")


##########################
#~~~~~~~~~~GUI~~~~~~~~~~~#
##########################

# Create the main window
root.title("Smart Table Plant Image Analysis")
root.minsize(880, 260) # minimum resolution
root.resizable(width=False, height=False) # window is NOT resizeable

# Create a label and entry to display the selected folder path
path_var = ctk.StringVar() 
path_label = ctk.CTkLabel(root, text="Selected Folder Path:")
path_label.place(x=30, y=10)
path_entry = ctk.CTkEntry(root, textvariable=path_var, width=690, corner_radius=8)
path_entry.place(x=160, y=10)

# Create a button to choose the folder path
choose_button = ctk.CTkButton(master=root, text="Choose Folder", command=choose_folder, width=200, corner_radius=8).place(x=30, y=60)

# Create a button to run the full analysis
run_button = ctk.CTkButton(master=root, text="Run Analysis", command=batch, width=200, corner_radius=8).place(x=30, y=100)

# Create a checkbox to enable batch processing
batch_box = ctk.CTkCheckBox(master=root, text="Enable Batch Processing", command=trigger_batch_status, checkbox_width=30, checkbox_height=30, corner_radius=5)
batch_box.place(x=30, y=140)

# Create a checkbox to enable image saving
save_box = ctk.CTkCheckBox(master=root, text="Enable Image Saving", command=save_image_trigger, checkbox_width=30, checkbox_height=30, corner_radius=5)
save_box.place(x=30, y=180)

# Initialize the debug window within the root window
debug_window = DebugWindow(root=root)

def clear():
    debug_window.clear()

# This button is only down here because the debug window needs to be intialized
# Along with the clear() method which also requires the debug window to be intialized
clear_debug = ctk.CTkButton(master=root, text="Clear Debug Window", command=clear, width=200, corner_radius=8)
clear_debug.place(x=30, y=220)

# Start the Tkinter event loop
root.mainloop()