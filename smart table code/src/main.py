# -*- coding: utf-8 -*-
import customtkinter as ctk
import time
from natsort import natsorted
import os
from win32api import GetSystemMetrics
import sys
import shutil
import threading
from rgb_2_bin import RGB2BIN
from extract_feature_points import TRACKMOTION
from extract_vectors_from_csv import export_to_excel

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state=ctk.NORMAL)
        self.widget.insert(ctk.END, string, (self.tag,))
        self.widget.see(ctk.END)
        self.widget.configure(state=ctk.DISABLED)
        self.widget.update_idletasks()

    def flush(self):
        pass  # Needed for compatibility with sys.stdout/sys.stderr

class DebugWindow:
    def __init__(self, root):
        self.debug_window = ctk.CTkTextbox(master=root, state='disabled',
                                           border_color='gray40', border_width=2,
                                           #fg_color='gray18',
                                           corner_radius=8, width=800, height=270)
        self.debug_window.place(x=250, y=60)
        #sys.stdout = TextRedirector(self.debug_window, "stdout")
        #sys.stderr = TextRedirector(self.debug_window, "stderr")

    def clear(self):
        self.debug_window.configure(state=ctk.NORMAL)
        self.debug_window.delete('1.0', ctk.END)
        self.debug_window.configure(state=ctk.DISABLED)
        self.debug_window.update_idletasks()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        self.geometry("%dx%d" % (1080, 360))
        self.title("SMART Table Analysis")
        self.state('zoomed')  # For Windows

        # Debug window (create FIRST)
        self.debug_window = DebugWindow(root=self)
        sys.stdout = TextRedirector(self.debug_window.debug_window, "stdout")
        sys.stderr = TextRedirector(self.debug_window.debug_window, "stderr")

        # Instance variables
        self.save_images = False
        self.batch_trigger = False

        # Path variable and widgets
        self.path_var = ctk.StringVar()
        self.BIN_paths = []
        self.CSV_paths = []
        path_label = ctk.CTkLabel(self, text="Selected Folder Path:")
        path_label.place(x=30, y=10)
        path_entry = ctk.CTkEntry(self, border_color='gray40', textvariable=self.path_var, width=890, corner_radius=8)
        path_entry.place(x=160, y=10)

        # Widget functionality code -- enables image saving
        self.save_images_var = ctk.BooleanVar(value=False)

        def save_image_trigger():
            self.save_images = self.save_images_var.get()

        # Widget functionality code -- enables batch processing
        def trigger_batch_status():
            self.batch_trigger = not self.batch_trigger
            self.choose_text.set("Choose Folder" if self.batch_trigger else "Choose File")
            return self.batch_trigger

        # Widget functionality code -- handles directories
        def choose_folder():
            if self.batch_trigger:
                folder_path = ctk.filedialog.askdirectory()
                self.path_var.set(os.path.normpath(folder_path))
            else:
                folder_path = ctk.filedialog.askdirectory()
                self.path_var.set(os.path.normpath(folder_path))

        # Widget functionality code -- runs the main function
        def batch():
            start = time.time()
            self.BIN_paths.clear()  # Clear previous paths before new batch
            if self.batch_trigger:
                paths = [os.path.normpath(os.path.join(self.path_var.get(), f.name))
                         for f in os.scandir(self.path_var.get()) if f.is_dir()]
                for path in natsorted(paths):
                    bin_path = RGB2BIN(path)
                    self.BIN_paths.append(bin_path if bin_path else path)
                print(f"~Processed images in {'%0.3f' % (time.time() - start)} seconds.\n~Average Process Time: {'%0.3f' % ((time.time() - start) / len(paths))} seconds.")
            else:
                bin_path = RGB2BIN(self.path_var.get())
                #self.BIN_paths = [bin_path if bin_path else self.path_var.get()]
                print(f"~Processed images in {'%0.3f' % (time.time() - start)} seconds.")

        def track_motion():
            start = time.time()
            self.CSV_paths.clear()  # Clear previous paths before new batch
            if self.batch_trigger:
                for path in natsorted(self.BIN_paths):
                    print(path)
                    self.CSV_paths.append(TRACKMOTION(path, delete_bin_files=self.save_images))
                    shutil.rmtree(os.path.normpath(path), ignore_errors=True)
                print(f"~Motion tracking in {'%0.3f' % (time.time() - start)} seconds.\n~Average: {'%0.3f' % ((time.time() - start) / len(self.BIN_paths))} seconds.")
            else:
                print(os.access(self.path_var.get(), os.W_OK))
                print(os.path.basename(self.path_var.get()))
                TRACKMOTION(self.path_var.get(), delete_bin_files=self.save_images)
                if os.path.basename(self.path_var.get()) == "processed_images":
                    shutil.rmtree(self.path_var.get(), ignore_errors=True)
                else:
                    shutil.rmtree(os.path.join(self.path_var.get(), 'processed_images'), ignore_errors=True)
                print(f"~Motion tracking in {'%0.3f' % (time.time() - start)} seconds.")

        def export_to_excel_gui():
            start = time.time()
            if self.batch_trigger:
                if self.CSV_paths != []:
                    for dir in natsorted(self.CSV_paths):
                        if os.path.isdir(dir):
                            csv_files = os.listdir(dir)
                            self.CSV_paths.extend([os.path.normpath(os.path.join(dir, f)) for f in csv_files if f.endswith('.csv')])
                        export_to_excel(True, csv_dir=dir)
                elif self.CSV_paths == []:
                    os.chdir(self.path_var.get())
                    csv_files = os.listdir(self.path_var.get())
                    self.CSV_paths = [os.path.normpath(os.path.join(self.path_var.get(), f)) for f in csv_files if f.endswith('.csv')]
                    
                print(self.CSV_paths)
                for path in natsorted(self.CSV_paths):
                    export_to_excel(True, csv_dir=path)
                    print(f"~Excel written in {'%0.3f' % (time.time() - start)} seconds.\n~Average: {'%0.3f' % ((time.time() - start) / len(self.BIN_paths))} seconds.")
            else:
                export_to_excel(False, csv_path=self.path_var.get())
                print(f"~Excel written in {'%0.3f' % (time.time() - start)} seconds.")

        def batch_threaded_RGB2BIN():
            threading.Thread(target=batch, daemon=True).start()

        def batch_threaded_MOTION():
            threading.Thread(target=track_motion, daemon=True).start()

        def batch_threaded_EXPORT():
            threading.Thread(target=export_to_excel_gui, daemon=True).start()

        # Create a button to choose the folder path
        self.choose_text = ctk.StringVar(value="Choose Folder" if self.batch_trigger else "Choose File")

        choose_button = ctk.CTkButton(
            master=self,
            textvariable=self.choose_text,
            command=choose_folder,
            width=200,
            corner_radius=8
        )
        choose_button.place(x=30, y=60)

        # Create a button to run the full analysis (RGB2BIN)
        run_button = ctk.CTkButton(master=self, text="Run Analysis", command=batch_threaded_RGB2BIN, width=200, corner_radius=8)
        run_button.place(x=30, y=100)

        # Create a button to run motion tracking
        motion_button = ctk.CTkButton(master=self, text="Run Motion Tracking", command=batch_threaded_MOTION, width=200, corner_radius=8)
        motion_button.place(x=30, y=140)

        # Create a button to export results to Excel
        export_button = ctk.CTkButton(master=self, text="Export to Excel", command=batch_threaded_EXPORT, width=200, corner_radius=8)
        export_button.place(x=30, y=180)

        # Create a checkbox to enable batch processing
        batch_box = ctk.CTkCheckBox(master=self, text="Enable Batch Processing", command=trigger_batch_status, checkbox_width=30, checkbox_height=30, corner_radius=5)
        batch_box.place(x=30, y=220)

        # Create a checkbox to enable image saving
        save_box = ctk.CTkCheckBox(
            master=self,
            text="Enable Image Saving",
            variable=self.save_images_var,
            command=save_image_trigger,
            checkbox_width=30,
            checkbox_height=30,
            corner_radius=5
        )
        save_box.place(x=30, y=260)

        def clear():
            self.debug_window.clear()

        clear_debug = ctk.CTkButton(master=self, text="Clear Debug Window", command=clear, width=200, corner_radius=8)
        clear_debug.place(x=30, y=300)

app = App()
app.mainloop()