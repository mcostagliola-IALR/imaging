~To run the analysis program, ensure you have a valid install of python.
~Your Python version must be at least v3.12.0
~To check your version, open the command prompt.
~This can be done with windows + 'command prompt'
~Or with holding windows + r, then typing 'cmd' into the prompt
~If your Python version is up to date, you can continue
~Otherwise, install the latest version of Python from https://www.python.org/downloads/

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~MAIN.PY~~~~~~~~~~~~~ This Section Only Contains Documentation For main.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Documentation For mergeData.py Can Be Found In The Next Section

~Right click inside of the unzipped folder, and click open in terminal
~From here, type 'python getModules.py'
~This will install all required modules for main.py
~Once this script has finished executing, type 'python main.py' to run the analysis software

Note: getModules.py only needs to be ran on time as a setup.
      Subsequent attempts to use main.py do not need to be superceeded by getModules.py

Features:
~By Default - main.py will create & delete a folder called 'processed_images' once processing is complete
~This can be disabled to allow for the folder to save and not be deleted by clicking 'Enable Image Saving'
~By Default - main.py is configured to process only one folder which directly contains RGB processed_images
~This can be changed to allow for Batch Processing by clicking 'Enable Batch Processing'
~This requires the format of directories to be as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~Correct Structure~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
-Parent Folder
---Sub Folder
-----RGB Images

~~~~~~~~~~~~~~~~~~~~~~~~~
~~~Incorrect Structure~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
-Parent Folder
---Sub Folder
-----RGB Images
-------Sub Folder
---------Binary/Non RGB Images


Note: There must be no sub folders within the sub folders
~Additionally, the above incorrect structure will be the result if the imaging analysis is ran-
-with Save Images Enable. If you need to rerun that folder, make sure to delete the processed_images folder within it
~One window-The GUI- will open once main.py is ran
~The top line is where the input file path will be displayed for user confirmation
~A large box is where any and all debug information will be shown, this includes
~~Current Folder
~~Current Folder Output Location
~~Per Folder Processing Time
~~Totat Concurrent Processing Time
~~Total Processing Time
~~Any errors generated
~Additionally, if the program is closed, or another window is focused, the analysis WILL BE SLOWER
~On average, 500 images on a 8 core, 16 thread CPU take 8.5-9.5 seconds
~A test on a 16 core, 24 thread CPU showed times as low as 5.8, averaging 6.5-7.2 seconds

~Once completed, an excel file containing motion and size data will be stored in each sub folder

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~IMPORTANT~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~If you get an error which contains "::scn 1" or something similar,
~check your folders, this message means that there are additional folders containing non RGB images.
~this can happen if the program gets interrupted and is closed mid processing.
~simply delete the unwanted folder(s) and move on.

~Additionally, if the program stops working, try to revert to python v3.12.4, which is the current version
~-being used to develop the programs

########################################################################################################################
~The Following Section Contains Documentation For Proper Use of mergeData.py

~Once mergeData.py is ran in the terminal, a window-The GUI- will open
~The top line is where the input folder path will be displayed for user confirmation
~The large box is where any and all debug information will be displayed, this includes:
~~List of all found excel files in parent directory
~~Current excel file in use
~~Any errors

~To select a folder, click on the 'Choose Folder' button:
~~This will open a file explorer window
~To Run the program, click 'Merge Sheets'
~Once completed, an excel file containing all the merged data from all excel files within the parent directory-
~-will be found inside of the parent directory, ie: (path = c:\user\yourPath, output: c:\user\yourPath\merged.xlsx)
~There are six customization options all relating to the generated charts:
~~Custom Motion page chart title
~~Custom Motion page x-axis label
~~Custom Motion page y-axis label
~~Custom Size page chart title
~~Custom Size page x-axis label
~~Custom Size page y-axis label
~Simply type the desired text into the box that will appear after checking a box-
~-there is no need to press enter, the text bar will still be blinking, but your title has been saved
~Additionally, if you no longer wish to have a custom title, simply uncheck the box

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~Default Titles/labels~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~Motion Page Chart Title: 'Plant Motion In Pixel Fluctuations Over Time'
~Motion Page X-axis Label: 'Capture Date'
~Motion Page Y-axis Label: 'Pixel Fluctuations'
~
~Size Page Chart Title: 'Plant Size In Pixel Count Over Time'
~Size Page X-axis Label: 'Capture Date'
~Size Page Y-axis Label: 'Pixel Count'

###########################################################################################################