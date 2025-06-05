import os
import sys
import pandas as pd
from natsort import natsorted
import customtkinter as ctk

'''
TODO
ADD A METHOD OF COLLECTING TIME DATA FROM A MEDIAN SHEET - DONE
ADD A DYNAMIC INTERVAL WHICH IS BASED OFF THE NUMBER OF TIME DATA AVAILABLE TO AVOID CROWDING ON LARGER PLOTS - DONE
'''


class GUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()  # Initialize the main window
        self.minsize(880, 500) # Sets window resolution
        self.resizable(width=False, height=False) # Disables resizing
        self.title('Excel Merger')  # Set the title of the main window
        
        self.path_var = ctk.StringVar() # Initialize string var
        self.path_label = ctk.CTkLabel(self, text="Selected Folder Path:") # Create label 
        self.path_label.place(x=30, y=10)
        
        # Entry box for path
        self.path_entry = ctk.CTkEntry(self, textvariable=self.path_var, width=690, corner_radius=8, border_color='gray40', border_width=2, fg_color='gray18')
        self.path_entry.place(x=160, y=10)
        
        # Initialize debug window
        self.debug_window = ctk.CTkTextbox(self, state='disabled', 
                                           border_color='gray40', border_width=2,
                                           fg_color='gray18',  
                                           corner_radius=8, width=600, height=189)
        self.debug_window.place(x=250, y=60)
        
        # Redirect stdout to the text widget
        sys.stdout = TextRedirector(self.debug_window, "stdout")
        sys.stderr = TextRedirector(self.debug_window, "stderr")
        
        # Create a button to choose folder path
        self.choose_folder_button = ctk.CTkButton(self, text='Choose Folder', width=200, height=89, corner_radius=8, command=self.getPath)
        self.choose_folder_button.place(x=30, y=60)
        
        # Create a button to run the main function
        self.merge_sheets_button = ctk.CTkButton(self, text='Merge Sheets', width=200, height=89, corner_radius=8, command=self.mergeData)
        self.merge_sheets_button.place(x=30, y=159)

        # Create a button to enable a custom title for motion page
        self.change_title_motion_button = ctk.CTkCheckBox(self, text='Custom Motion Chart Title', width=20, height=50, corner_radius=8, command=self.getMotionTitle)
        self.change_title_motion_button.place(x=30, y=250)
        self.change_title_motion_trigger = False

        # Create a button to enable a custom title for size page
        self.change_title_size_button = ctk.CTkCheckBox(self, text='Custom Size Chart Title', width=20, height=50, corner_radius=8, command=self.getSizeTitle)
        self.change_title_size_button.place(x=30, y=370)
        self.change_title_size_trigger = False

        # The rest of this code is just widgets for custom chart features like labels and titles
        self.change_xaxis_motion_button = ctk.CTkCheckBox(self, text='Custom Motion x-axis Label', width=20, height=50, corner_radius=8, command=self.getXMotionLabel)
        self.change_xaxis_motion_button.place(x=30, y=290)
        self.change_xaxis_motion_trigger = False

        self.change_xaxis_size_button = ctk.CTkCheckBox(self, text='Custom Size x-axis Label', width=20, height=50, corner_radius=8, command=self.getXSizeLabel)
        self.change_xaxis_size_button.place(x=30, y=410)
        self.change_xaxis_size_trigger = False

        self.change_yaxis_motion_button = ctk.CTkCheckBox(self, text='Custom Motion y-axis Label', width=20, height=50, corner_radius=8, command=self.getYMotionLabel)
        self.change_yaxis_motion_button.place(x=30, y=330)
        self.change_yaxis_motion_trigger = False

        self.change_yaxis_size_button = ctk.CTkCheckBox(self, text='Custom Size y-axis Label', width=20, height=50, corner_radius=8, command=self.getYSizeLabel)
        self.change_yaxis_size_button.place(x=30, y=450)
        self.change_yaxis_size_trigger = False

        # Initialize all variables related to the customization widgets
        self.titleMotionVar = ctk.StringVar(self)
        self.titleMotion = 'Plant Motion In Pixel Fluctuations Over Time'
        self.titleSizeVar = ctk.StringVar(self)
        self.titleSize = 'Plant Size In Pixel Count Over Time'
        self.xaxisLabelMotionVar = ctk.StringVar(self)
        self.xaxisLabelMotion = 'Capture Date'
        self.xaxisLabelSizeVar = ctk.StringVar(self)
        self.xaxisLabelSize = 'Capture Date'
        self.yaxisLabelMotionVar = ctk.StringVar(self)
        self.yaxisLabelMotion = 'Pixel Fluctuations'
        self.yaxisLabelSizeVar = ctk.StringVar(self)
        self.yaxisLabelSize = 'Pixel Count'

    # General rundown
    # def someFunc(self):
    # if some_trigger == True (meaning that someFunc(self) has already been ran)
    # set that trigger back to false
    # destroy the widget since this idicates that the button was checked again, ie: unchecking the button
    # if some_trigger == False (meaning someFunc(self) has not been ran or was reset)
    # create the entry widget
    # place the widget
    # set the trigger to true since someFunc(self) has now been ran

    def getMotionTitle(self):
        if self.change_title_motion_trigger == True:
            self.change_title_motion_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.titleMotionVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=260)
            self.change_title_motion_trigger = True

    def getSizeTitle(self):
        if self.change_title_size_trigger == True:
            self.change_title_size_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.titleSizeVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=380)
            self.change_title_size_trigger = True 

    def getXMotionLabel(self):
        if self.change_xaxis_motion_trigger == True:
            self.change_xaxis_motion_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.xaxisLabelMotionVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=300)
            self.change_xaxis_motion_trigger = True 

    def getXSizeLabel(self):
        if self.change_xaxis_size_trigger == True:
            self.change_xaxis_size_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.xaxisLabelSizeVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=420)
            self.change_xaxis_size_trigger = True 

    def getYMotionLabel(self):
        if self.change_yaxis_motion_trigger == True:
            self.change_yaxis_motion_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.yaxisLabelMotionVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=340)
            self.change_yaxis_motion_trigger = True 

    def getYSizeLabel(self):
        if self.change_yaxis_size_trigger == True:
            self.change_title_size_trigger = False
            self.entry.destroy()
        else:
            self.entry = ctk.CTkEntry(self, textvariable=self.yaxisLabelSizeVar, width=600, border_color='gray40', border_width=2, fg_color='gray18')
            self.entry.place(x=250, y=460)
            self.change_yaxis_size_trigger = True 
        
    # Logic for clearing the debug window 
    def clear(self):
        '''Clear the contents of the debug window'''
        self.debug_window.configure(state=ctk.NORMAL)  # Enable editing
        self.debug_window.delete('1.0', ctk.END)  # Delete all text
        self.debug_window.configure(state=ctk.DISABLED)  # Disable editing
        self.debug_window.update_idletasks()  # Update & display/reflect changes on window (if any)

    # Logic for obtaining the folder path
    def getPath(self):
        self.path_var.set("")  # Clear the path entry
        self.path = ctk.filedialog.askdirectory()
        self.path_var.set(self.path)  # Update path entry with selected path

    # Main Function
    def mergeData(self):

        # List cols for xlsxwriter/excel - despite only going to z, there is no limit to the length of folders due to some later logic
        cols = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        path = self.path_var.get()
        
        if not path:
            print("Please select a folder first.")
            return
        
        # Set working directory to path, this will cause the generated Excel file to be within the path folder
        os.chdir(path)

        excelFileList = []
        pathList = natsorted([os.path.normpath(os.path.join(path, f.name)) for f in os.scandir(path) if f.is_dir()]) # list comprehension to isolate paths in parent path
        legend = [] # Contains Titles for the legend
        # Isolates end of path to get legend titles
        for path in pathList:
            folder_name = os.path.basename(path)
            legend.append(folder_name)

        # Isolates all .xlsx files
        for path in pathList:
            lst = os.listdir(path)
            for file in lst:
                if file.endswith(".xlsx"):
                    excelFileList.append(os.path.join(path, file))

        print("Excel files found:")
        print(excelFileList)

        merged_df_Motion = pd.DataFrame()  # Initialize empty DataFrame for Motion_Worksheet
        merged_df_Size = pd.DataFrame()    # Initialize empty DataFrame for Size_Worksheet
        timeDataFrame = pd.DataFrame()     # Initialize empty DataFrame for story time data

        for file in natsorted(excelFileList): # natsored erased the issue of for example: 1, 11, 2, being the output of sort()
            print(f'Processing file: {file}')
            try:

                self.update() # To ensure the window stays responding
                # Read Motion_Worksheet
                dfM = pd.read_excel(file, sheet_name='Motion_Worksheet', skiprows=[0, 2], usecols='C')
                # Concatenate horizontally to merged DataFrame
                merged_df_Motion = pd.concat([merged_df_Motion, dfM], axis=1)

                # Read Size_Worksheet
                dfS = pd.read_excel(file, sheet_name='Size_Worksheet', skiprows=[0, 1], usecols='C')
                # Concatenate horizontally to merged DataFrame
                merged_df_Size = pd.concat([merged_df_Size, dfS], axis=1)

                rowM = dfM.shape[0] # Returns num rows in 'Motion_Worksheet'
                rowS = dfS.shape[0] # Returns num rows in 'Size_Worksheet'

            except Exception as e:
                print(f'Error processing {file}: {e}')

        # Get the time values from a sheet somewhere in the middle of the list to have an median time spread
        # Read the Excel file and extract the specified column
        timeDataFrame = pd.read_excel(excelFileList[(len(excelFileList) // 2) - 1], sheet_name='Motion_Worksheet', skiprows=[0, 1], usecols='B')

        # Write merged DataFrames to Excel
        with pd.ExcelWriter('Merged.xlsx', engine='xlsxwriter') as writer:
            self.update() # To ensure the window stays responding
            merged_df_Motion.to_excel(writer, sheet_name='Motion_Worksheet', index=False, startcol=1)
            merged_df_Size.to_excel(writer, sheet_name='Size_Worksheet', index=False, startrow=1, startcol=1)
            timeDataFrame.to_excel(writer, sheet_name='Motion_Worksheet', index=False)
            timeDataFrame.to_excel(writer, sheet_name='Size_Worksheet', index=False, startrow=1)

            workbook = writer.book
            worksheetMotion = writer.sheets['Motion_Worksheet']
            worksheetSize = writer.sheets['Size_Worksheet']

            chartMotion = workbook.add_chart({'type': 'line'})
            if self.change_title_motion_trigger == True:
                chartMotion.set_title({'name': f'{self.titleMotionVar.get()}'})
            else:
                chartMotion.set_title({'name': f'{self.titleMotion}'})
            chartSize = workbook.add_chart({'type': 'line'})
            if self.change_title_size_trigger == True:
                chartSize.set_title({'name': f'{self.titleSizeVar.get()}'})
            else:
                chartSize.set_title({'name': f'{self.titleSize}'})
            # Adjust chartMotion add_series

            # Somewhat complicated xlsxwriter code below
            # Basically loops through each folder or dataframe and creates a separate plot on the same chart
            # Complicated part is the logic to ensure proper excel format for cols, ie: AA, AB, AC, once going beyond Z
            # It works, don't touch it

            for i in range(len(legend)):
                prefixCol: int = i // 26 # Variable that cols[i] will be multiplied by to generate proper column, only returns integer
                if prefixCol < 1:
                    chartMotion.add_series({
                        'name': legend[i],
                        'categories': f'=Motion_Worksheet!${cols[1]}$2:${cols[1]}${rowM-1}',  # Sets dates as x-axis
                        'values': f'=Motion_Worksheet!${cols[prefixCol] + cols[i + 1]}$2:${cols[prefixCol] + cols[i + 1]}${rowM-1}',  # Sets plot values
                        'line': {'width': 1},
                    })

                    # Adjust chartSize add_series
                    chartSize.add_series({
                        'name': legend[i],
                        'categories': f'=Size_Worksheet!${cols[1]}$2:${cols[1]}${rowS-1}',  # Sets dates as x-axis
                        'values': f'=Size_Worksheet!${cols[prefixCol] + cols[i + 1]}$2:${cols[prefixCol] + cols[i + 1]}${rowS-1}',  # Sets plot values
                        'line': {'width': 1},
                    })
                else:
                    temp_i = i % 26 # temp variable for to colSuffix, only as a temp for the 'name': lengend[i] line
                    chartMotion.add_series({
                        'name': legend[i],
                        'categories': f'=Motion_Worksheet!${cols[1]}$2:${cols[1]}${rowM-1}',  # Sets dates as x-axis
                        'values': f'=Motion_Worksheet!${cols[prefixCol] + cols[temp_i + 1]}$2:${cols[prefixCol] + cols[temp_i + 1]}${rowM-1}',  # Sets plot values
                        'line': {'width': 1},
                    })

                    # Adjust chartSize add_series
                    chartSize.add_series({
                        'name': legend[i],
                        'categories': f'=Size_Worksheet!${cols[1]}$2:${cols[1]}${rowS-1}',  # Sets dates as x-axis
                        'values': f'=Size_Worksheet!${cols[prefixCol] + cols[temp_i + 1]}$2:${cols[prefixCol] + cols[temp_i + 1]}${rowS-1}',  # Sets plot values
                        'line': {'width': 1},
                    })


            if self.change_xaxis_motion_trigger == True:
                chartMotion.set_x_axis({
                    'name': f'{self.xaxisLabelMotionVar.get()}',
                    'name_font': {'size': 18, 'bold': True},
                    'num_font': {'rotation': -45},
                    'date_axis': True,  # Set the x-axis as a date axis
                    'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
                })
            else:
                chartMotion.set_x_axis({
                    'name': 'Capture Date',
                    'name_font': {'size': 18, 'bold': True},
                    'num_font': {'rotation': -45},
                    'date_axis': True,  # Set the x-axis as a date axis
                    'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
                })

            if self.change_yaxis_motion_trigger == True:
                chartMotion.set_y_axis({
                    'name': f'{self.yaxisLabelMotionVar.get()}',
                    'name_font': {'size': 18, 'bold': True},
                })
            else:
                chartMotion.set_y_axis({
                    'name': 'Pixel Fluctuaton',
                    'name_font': {'size': 18, 'bold': True},
                })


            if self.change_xaxis_size_trigger == True:
                chartSize.set_x_axis({
                    'name': f'{self.xaxisLabelSizeVar.get()}',
                    'name_font': {'size': 18, 'bold': True},
                    'num_font': {'rotation': -45},
                    'date_axis': True,  # Set the x-axis as a date axis
                    'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
                })
            else:
                chartSize.set_x_axis({
                    'name': 'Capture Date',
                    'name_font': {'size': 18, 'bold': True},
                    'num_font': {'rotation': -45},
                    'date_axis': True,  # Set the x-axis as a date axis
                    'major_unit': 12,  # Specify the interval count (display every 2 days, for example)
                })

            if self.change_yaxis_size_trigger == True:
                chartSize.set_y_axis({
                    'name': f'{self.yaxisLabelSizeVar.get()}',
                    'name_font': {'size': 18, 'bold': True},
                })
            else:
                chartSize.set_y_axis({
                    'name': 'Pixel Count',
                    'name_font': {'size': 18, 'bold': True},
                })

            worksheetMotion.insert_chart('F5', chartMotion, {'x_scale': 2.5, 'y_scale': 2})
            worksheetSize.insert_chart('F5', chartSize, {'x_scale': 2.5, 'y_scale': 2})

        print("Merged.xlsx created successfully.")



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

app = GUI()
app.mainloop()