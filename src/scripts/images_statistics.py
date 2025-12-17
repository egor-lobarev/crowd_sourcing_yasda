import numpy as np
from PIL import Image
import os

# Settings
folder = "./src/data/raw/images"  # CHANGE THIS
img_size = 128

# Get all JPG/PNG files
files = [os.path.join(folder, f) for f in os.listdir(folder) 
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Calculate stats
mean_sum = np.zeros(3)
std_sum = np.zeros(3)

for f in files:
    img = Image.open(f).convert('RGB').resize((img_size, img_size))
    arr = np.array(img) / 255.0
    mean_sum += arr.mean(axis=(0, 1))
    std_sum += arr.std(axis=(0, 1))

# Results
mean = mean_sum / len(files)
std = std_sum / len(files)

print(f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
print(f"Std:  [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")