
from ultralytics import YOLO
import os
import yaml
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import osend

new_data_path = r"C:\Users\arnav.shah\Desktop\vehicle_data"
files = os.listdir(new_data_path)
print("Files in folder:", files)

# Load your trained model
model = YOLO(r"C:\Users\arnav.shah\Downloads\runs\detect\train\weights\best.pt")

# Define the path to your new unseen data (make sure to use a raw string or escape backslashes)
new_data_path = r"C:\Users\arnav.shah\Desktop\vehicle_data"

# Run inference on the new data

results = model.predict(
    source=new_data_path,  # Folder containing the new images
    conf=0.3,              # Confidence threshold (adjust if needed)
    save=True,             # Save the output images with predictions
    show=False              # Display the predictions in real-time
)

