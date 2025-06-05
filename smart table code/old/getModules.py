import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = ['plantcv==4.2.1', 'numpy==1.26.4', 'xlsxwriter==3.2.0', 'customtkinter==5.2.2', 'natsort==8.4.0', 'setuptools', 'pandas==2.2.2', 'opencv-python==4.10.0.84']

for package in packages:
    install(package)
