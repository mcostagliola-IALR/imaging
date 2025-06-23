#This is just the python version of the notebook of the same name.
#For documentation on the code, go to the notebook version of this script.
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.notebook import tqdm
from torchvision.models import ResNet50_Weights
import torch.optim as optim
import pandas as pd
import re
from datetime import datetime
import os
import xlsxwriter
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

data_dir = r"c:\Users\dbrimmer\Downloads\plant_dataset_split_v4"
epochs = 10
batch_size = 32
learning_rate = 0.1
#Simply uses the GPU if available, otherwise uses the CPU.
# A GPU will not be necessary for this task, but it will speed up training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    #Most machine learning models use this size, especially ResNet18, but if a different model is used, this may need to be changed.
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    transform=transform
)
val_data = datasets.ImageFolder(
    os.path.join(data_dir, 'val'),
    transform=transform
)
test_data = datasets.ImageFolder(
    os.path.join(data_dir, 'test'),
    transform=transform
)
# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)

def show_batch(loader):
    images, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(images[:8], nrow=4)
    #The two lines are to pervent overfitting like with what we saw in v3
    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0) * 0.5 + 0.5)  # unnormalize
    plt.title([class_names[l.item()] for l in labels[:8]])
    plt.axis('off')
    plt.show()

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_data.classes))
model = model.to(device)

#The current model you want to use, change if you made another model.
#model_path=r"Plant-Wilting-Model_v4.pth"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "Plant-Wilting-Model_v4.pth")
print("Looking for model at:", os.path.abspath(model_path))
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded existing model weights.")
else:
    print("No saved model found. You need to train the model first.")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def extract_datetime_from_filename(filename):
    """Extract datetime from filename in MM-DD-YYYY_HH-MM format."""
    match = re.match(r"(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2})", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        dt_str = f"{date_str} {time_str}"
        return datetime.strptime(dt_str, "%m-%d-%Y %H-%M")
    return None
#O(n)
def classify_images_in_folder(folder_path, excel_path="results.xlsx"):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            label = 0 if predict_image(image_path) == "healthy" else 1
            dt = extract_datetime_from_filename(filename)
            results.append({'filename': filename, 'datetime': dt, 'label': label})
    
    df = pd.DataFrame(results)
    df = df.dropna(subset=['datetime'])
    df = df.sort_values(by='datetime')
    
    # Convert datetime to Excel serial number to match provided data
    df['datetime'] = df['datetime'].apply(
        lambda dt: (dt - datetime(1899, 12, 30)).total_seconds() / 86400
    )
    
    # Optional: Aggregate data by day to reduce clutter (uncomment if desired)
    # df['date'] = df['datetime'].apply(lambda x: int(x))
    # df = df.groupby('date').agg({'label': 'max'}).reset_index()
    # df['datetime'] = df['date'] + 0.5
    # df = df[['datetime', 'label']]
    
    df = df.drop(columns=['filename'])
    
    # Write to Excel using XlsxWriter
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter', datetime_format='mm/dd/yyyy hh:mm')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    # Get XlsxWriter objects
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Format datetime column explicitly
    date_format = workbook.add_format({'num_format': 'mm/dd/yyyy hh:mm'})
    worksheet.set_column('A:A', 15, date_format)
    
    # Create line chart
    chart = workbook.add_chart({'type': 'line'})
    chart.set_title({'name': 'Plant Health Over Time'})
    chart.set_x_axis({
        'name': 'Date/Time',
        'date_axis': True,
        'num_format': 'mm/dd/yyyy',
        'major_unit': 2,  # Ticks every 2 days
        'minor_unit': 0.5  # Minor ticks every 12 hours
    })
    chart.set_y_axis({
        'name': 'Health Status (0=Healthy, 1=Wilted)',
        'min': 0,
        'max': 1,
        'major_unit': 1,
        'major_gridlines': {'visible': False}
    })
    
    # Add data series
    max_row = len(df) + 1  # Account for header
    chart.add_series({
        'values': f'=Sheet1!$B$2:$B${max_row}',
        'categories': f'=Sheet1!$A$2:$A${max_row}',
        'line': {'color': "#158D39", 'width': 2},
        'marker': {
            'type': 'circle',
            'size': 5,
            'fill': {'color': "#00FF95"},
            'border': {'color': 'black'}
        }
    })
    
    # Insert chart
    worksheet.insert_chart('E5', chart, {'x_scale': 2.0, 'y_scale': 1.5})
    
    # Close workbook
    writer.close()
    return df




