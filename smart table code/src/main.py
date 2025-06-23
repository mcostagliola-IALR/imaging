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
from Plant_Wilt_detection_script import classify_images_in_folder

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
                                           corner_radius=8, width=800, height=310)
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
        # GetSystemMetrics is a Windows-specific function; for cross-platform,
        # you might want to use self.winfo_screenwidth() and self.winfo_screenheight()
        # For simplicity, keeping your original for now if this is Windows-only.
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        self.geometry("%dx%d" % (1080, 400)) # This will be overridden by 'zoomed' state on Windows
        self.title("SMART Table Analysis")
        self.state('zoomed')  # For Windows
        self._set_appearance_mode("system")

        # Debug window (create FIRST)
        self.debug_window = DebugWindow(root=self)
        sys.stdout = TextRedirector(self.debug_window.debug_window, "stdout")
        sys.stderr = TextRedirector(self.debug_window.debug_window, "stderr")

        # Instance variables
        self.save_images = False
        self.batch_trigger = False
        self.auto = True # Initial state is auto mode

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

        def auto_trigger():
            self.auto = not self.auto
            self.update_run_buttons_visibility() # Call this function to update buttons
            return self.auto

        def save_image_trigger():
            self.save_images = self.save_images_var.get()

        # Widget functionality code -- enables batch processing
        def trigger_batch_status():
            self.batch_trigger = not self.batch_trigger
            self.choose_text.set("Choose Folder" if self.batch_trigger else "Choose File")
            return self.batch_trigger

        # Widget functionality code -- handles directories
        def choose_folder():
            # It seems both batch and non-batch modes ask for a folder.
            # If "Choose File" implies selecting a single file, you'd need ctk.filedialog.askopenfilename()
            # For now, it consistently asks for a directory.
            folder_path = ctk.filedialog.askdirectory()
            if folder_path: # Ensure a path was selected
                self.path_var.set(os.path.normpath(folder_path))

        # Widget functionality code -- runs the main function
        def batch():
            """Convert RGB images to binary in batch or single mode."""
            start = time.time()
            self.BIN_paths.clear()  # Clear previous paths before new batch
            
            try:
                if self.batch_trigger:
                    # Get all subdirectories
                    # Using glob.glob is often more robust for pattern matching if needed,
                    # but os.scandir is fine for direct subdirectories.
                    paths = [
                        os.path.normpath(os.path.join(self.path_var.get(), f.name))
                        for f in os.scandir(self.path_var.get()) if f.is_dir()
                    ]
                    
                    if not paths:
                        print("No valid directories found to convert to binary.")
                        return False
                        
                    for path in natsorted(paths):
                        print(f"Processing RGB2BIN: {os.path.basename(path)}")
                        bin_path = RGB2BIN(path)
                        if bin_path:
                            self.BIN_paths.append(bin_path)
                            
                    print(f"~Processed images in {'%0.3f' % (time.time() - start)} seconds.")
                    if paths: # Avoid division by zero
                        print(f"~Average: {'%0.3f' % ((time.time() - start) / len(paths))} seconds.")
                    
                else:  # Single folder processing
                    print(f"Processing RGB2BIN: {os.path.basename(self.path_var.get())}")
                    bin_path = RGB2BIN(self.path_var.get())
                    if bin_path:
                        self.BIN_paths.append(bin_path)
                    print(f"~Processed images in {'%0.3f' % (time.time() - start)} seconds.")
                    
                return bool(self.BIN_paths)  # Return True if any paths were processed
                
            except Exception as e:
                print(f"Error in RGB2BIN processing: {e}")
                return False
        def batch_classify_images_in_folder():
            if self.batch_trigger:
                # Get all subdirectories
                subdirs = [
                    d.path for d in os.scandir(self.path_var.get()) 
                    if d.is_dir() and not d.name.startswith('.')
                ]
                
                if not subdirs:
                    print("No valid subdirectories found for classification.")
                    return False
                
                for subdir in natsorted(subdirs):
                    print(f"Classifying images in: {os.path.basename(subdir)}")
                    excel_path = os.path.join(subdir, "Health_results.xlsx")
                    classify_images_in_folder(subdir, excel_path)
                    
                print("Classification completed for batch.")
                
            else:
                    excel_path = os.path.join(self.path_var.get(), "Health_results.xlsx")
                    classify_images_in_folder(self.path_var.get(),excel_path)
                    print("Completed for folder: ", os.path.basename(self.path_var.get()))

        def track_motion():
            """Process images and track motion features."""
            start = time.time()
            self.CSV_paths.clear()
            
            try:
                if self.batch_trigger:  # Handle multiple folders
                    # Get all subdirectories
                    subdirs = [
                        d.path for d in os.scandir(self.path_var.get()) 
                        if d.is_dir() and not d.name.startswith('.')
                    ]
                    
                    if not subdirs:
                        print("No valid subdirectories found for motion tracking.")
                        return
                        
                    for subdir in natsorted(subdirs):
                        print(f"Processing motion tracking: {os.path.basename(subdir)}")
                        csv_path = TRACKMOTION(subdir, delete_bin_files=self.save_images)
                        if csv_path:
                            self.CSV_paths.append(csv_path)
                            # Clean up processed images
                            # The logic here seems to assume 'processed_images' is a direct child
                            # of either the selected directory or a subdir within the selected directory.
                            # It's a bit ambiguous how 'delete_bin_files' relates to this cleanup.
                            # Usually, delete_bin_files would mean deleting the intermediate binary images.
                            # The shutil.rmtree part cleans up the 'processed_images' *folder*.
                            processed_images_dir = os.path.join(subdir, 'processed_images')
                            if os.path.exists(processed_images_dir):
                                print(f"Cleaning up {processed_images_dir}...")
                                shutil.rmtree(processed_images_dir, ignore_errors=True)
                                
                    print(f"~Motion tracking completed in {'%0.3f' % (time.time() - start)} seconds")
                    if subdirs: # Avoid division by zero
                        print(f"~Average: {'%0.3f' % ((time.time() - start) / len(subdirs))} seconds")
                    
                else:  # Handle single folder
                    print(f"Processing motion tracking: {os.path.basename(self.path_var.get())}")
                    csv_path = TRACKMOTION(self.path_var.get(), 
                                            delete_bin_files=self.save_images)
                    if csv_path:
                        self.CSV_paths.append(csv_path)
                        processed_images_dir = os.path.join(self.path_var.get(), 'processed_images')
                        if os.path.exists(processed_images_dir):
                            print(f"Cleaning up {processed_images_dir}...")
                            shutil.rmtree(processed_images_dir, ignore_errors=True)
                                            
                    print(f"~Motion tracking completed in {'%0.3f' % (time.time() - start)} seconds")
                    
            except Exception as e:
                print(f"Error in motion tracking: {e}")
                
            return bool(self.CSV_paths)  # Return True if any paths were processed

        def export_to_excel_gui(self=self):
                """Export motion tracking data to Excel files."""
                start = time.time()
                self.export_complete = False
                    
                try:
                    export_targets = []
                    
                    print(f"DEBUG EXPORT: Batch Trigger: {self.batch_trigger}")
                    print(f"DEBUG EXPORT: CSV_paths (from TRACKMOTION): {self.CSV_paths}")

                    if self.batch_trigger:
                        if self.CSV_paths: # Prefer CSV_paths if populated from previous step
                            # --- Logic Branch A: Batch mode, CSV_paths populated ---
                            print("DEBUG EXPORT: Entering Batch Mode (CSV_paths populated)")
                            
                            # *** MODIFICATION HERE ***
                            # Use the paths in self.CSV_paths directly as they already point to the 'motion_features' directories
                            # Use a set to handle potential duplicates, then convert back to list for natsorted
                            export_targets = natsorted(list(set(self.CSV_paths)))
                            print(f"DEBUG EXPORT: Export targets from CSV_paths directly: {export_targets}")

                        else: # Fallback: scan current path for motion_features subfolders
                            # --- Logic Branch B: Batch mode, CSV_paths empty (fallback) ---
                            print("DEBUG EXPORT: Entering Batch Mode (CSV_paths EMPTY, using fallback scan)")
                            subdirs = [d.path for d in os.scandir(self.path_var.get())
                                        if d.is_dir() and not d.name.startswith('.')]
                            print(f"DEBUG EXPORT: Subdirectories found in {self.path_var.get()}: {subdirs}")
                            
                            export_targets = []
                            for subdir in natsorted(subdirs):
                                potential_motion_features_path = os.path.join(subdir, 'motion_features')
                                print(f"DEBUG EXPORT: Checking for 'motion_features' at: {potential_motion_features_path}")
                                if os.path.isdir(potential_motion_features_path):
                                    export_targets.append(potential_motion_features_path)
                                    print(f"DEBUG EXPORT: Found 'motion_features' at: {potential_motion_features_path}")
                                else:
                                    print(f"DEBUG EXPORT: 'motion_features' NOT found at: {potential_motion_features_path}")

                    else: # Single mode
                        # --- Logic Branch C: Single mode ---
                        print("DEBUG EXPORT: Entering Single Mode")
                        current_path = self.path_var.get()
                        print(f"DEBUG EXPORT: Current path in single mode: {current_path}")
                        if not current_path:
                            print("DEBUG EXPORT: Please select a valid directory or CSV file first.")
                            return False

                        csv_dir = os.path.join(current_path, 'motion_features')
                        print(f"DEBUG EXPORT: Checking for 'motion_features' directory: {csv_dir}")
                        if os.path.isdir(csv_dir):
                            export_targets = [csv_dir]
                            print(f"DEBUG EXPORT: Found 'motion_features' directory for single export: {export_targets}")
                        elif os.path.isfile(current_path) and current_path.lower().endswith('.csv'):
                            export_targets = [current_path] # It's a single CSV file directly selected
                            print(f"DEBUG EXPORT: Found single CSV file for export: {export_targets}")
                        else:
                            print(f"DEBUG EXPORT: No 'motion_features' folder or CSV file found at {current_path}")
                            return False

                    # --- DEBUGGING POINT 2: Final list of targets before processing ---
                    total_items = len(export_targets)
                    print(f"DEBUG EXPORT: Final list of export targets: {export_targets}")
                    print(f"DEBUG EXPORT: Total items for export: {total_items}")

                    if total_items == 0:
                        print("DEBUG EXPORT: No items to process for Excel Export after all checks.")
                        return False

                    for i, target_path in enumerate(natsorted(export_targets)):
                        #progress_value = (i + 1) / total_items
                        #self.update_progress(progress_value, f"Exporting {os.path.basename(target_path)} to Excel...")
                        print(f"Exporting to Excel from: {os.path.basename(target_path)}")
                        
                        if os.path.isdir(target_path): # If it's a 'motion_features' directory
                            print(f"DEBUG EXPORT: Calling export_to_excel (directory mode) for: {target_path}")
                            export_to_excel(True, csv_dir=target_path)
                        else: # If it's a direct CSV file
                            print(f"DEBUG EXPORT: Calling export_to_excel (file mode) for: {target_path}")
                            export_to_excel(False, csv_path=target_path)
                                        
                    self.export_complete = True
                    print(f"~Excel export completed in {'%0.3f' % (time.time() - start)} seconds")
                    if total_items > 0:
                        print(f"~Average per item: {'%0.3f' % ((time.time() - start) / total_items)} seconds")
                    print("Export completed successfully!")
                        
                    return self.export_complete
                        
                except Exception as e:
                    print(f"Export failed: {e}")
                    self.export_complete = False
                    return False
                
        def processing():
            # This is your chained process for the "Auto Run Program" button
            # You might want to add some visual feedback (e.g., disable buttons, show progress)
            # during these long-running operations.
            print("Starting Auto Run process...")
            if batch():
                if track_motion():
                    export_to_excel_gui(self)
                if batch_classify_images_in_folder():
                    print("Batch classification completed.")
            print("Auto Run process finished.")

        # Threaded functions
        def batch_threaded_RGB2BIN():
            threading.Thread(target=batch, daemon=True).start()

        def batch_threaded_MOTION():
            threading.Thread(target=track_motion, daemon=True).start()

        def batch_threaded_EXPORT():
            threading.Thread(target=export_to_excel_gui, daemon=True).start()

        def auto_process_threaded():
            threading.Thread(target=processing, daemon=True).start()

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

        # Create all buttons, but manage their visibility
        self.auto_run_button = ctk.CTkButton(master=self, text="Auto Run Program", command=auto_process_threaded, width=200, height=108 , corner_radius=8)
        
        self.run_analysis_button = ctk.CTkButton(master=self, text="Run Analysis", command=batch_threaded_RGB2BIN, width=200, corner_radius=8)
        self.motion_button = ctk.CTkButton(master=self, text="Run Motion Tracking", command=batch_threaded_MOTION, width=200, corner_radius=8)
        self.export_button = ctk.CTkButton(master=self, text="Export to Excel", command=batch_threaded_EXPORT, width=200, corner_radius=8)

        # Call the update function initially to set correct visibility based on self.auto
        self.update_run_buttons_visibility()

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
        clear_debug.place(x=30, y=340)

        # The checkbox to toggle auto mode
        self.enable_manual_run_checkbox = ctk.CTkCheckBox(master=self, text="Disable Auto Run", command=auto_trigger, checkbox_width=30, checkbox_height=30, corner_radius=5)
        self.enable_manual_run_checkbox.place(x=30, y=300)

    def update_run_buttons_visibility(self):
        """Manages the visibility of the run buttons based on the 'auto' state."""
        if self.auto:
            self.auto_run_button.place(x=30, y=100) # Show auto button
            # Hide manual buttons
            self.run_analysis_button.place_forget()
            self.motion_button.place_forget()
            self.export_button.place_forget()
        else:
            self.auto_run_button.place_forget() # Hide auto button
            # Show manual buttons
            self.run_analysis_button.place(x=30, y=100)
            self.motion_button.place(x=30, y=140)
            self.export_button.place(x=30, y=180)
        self.update_idletasks() # Refresh the GUI


app = App()
app.mainloop()