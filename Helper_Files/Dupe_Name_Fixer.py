import os
import re

# Set your folder path and replacement string
folder = r"C:\Users\dbrimmer\Downloads\Plant1"
replacement = 'BornAgain'  # Replace with what you want

for filename in os.listdir(folder):
    # Replace 'rep' followed by numbers at the end (before file extension)
    new_name = re.sub(r'rep\d+(?=\.\w+$)', replacement, filename, flags=re.IGNORECASE)
    if new_name != filename:
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
        print(f'Renamed: {filename} -> {new_name}')