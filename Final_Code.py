import os
import yaml
import cv2
import numpy as np
#import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# ============================ 1. Data Ingestion ============================ #
root_dir = r'C:\Users\arnav.shah\Desktop\VehiclesDetectionDataset'
train_images_dir = os.path.join(root_dir, 'train', 'images')
train_labels_dir = os.path.join(root_dir, 'train', 'labels')

# List all files in the images and labels directories
image_files = sorted(os.listdir(train_images_dir))
label_files = sorted(os.listdir(train_labels_dir))

print(f"Total images ingested: {len(image_files)}")
print(f"Total labels ingested: {len(label_files)}")

# ============================ 2. Data Cleanup ============================ #
# Extract basenames (without extension) for comparison
image_basenames = [os.path.splitext(f)[0] for f in image_files]
label_basenames = [os.path.splitext(f)[0] for f in label_files]

# Identify missing labels or images
missing_labels = list(set(image_basenames) - set(label_basenames))
missing_images = list(set(label_basenames) - set(image_basenames))

print(f"Missing labels for these images: {missing_labels}")
print(f"Missing images for these labels: {missing_images}")

# Check file formats in the directories
def check_files_format(image_list, label_list):
    non_jpg_files = [f for f in image_list if not f.lower().endswith('.jpg')]
    non_txt_files = [f for f in label_list if not f.lower().endswith('.txt')]
    
    if non_jpg_files:
        print("Non-JPG files found in images directory:", non_jpg_files)
    else:
        print("All images are in JPG format.")
    
    if non_txt_files:
        print("Non-TXT files found in labels directory:", non_txt_files)
    else:
        print("All labels are in TXT format.")

check_files_format(image_files, label_files)

# ============================ 3. Data Preparation & Original Distribution ============================ #
# Load class names from dataset.yaml
def load_dataset_info():
    with open(os.path.join(root_dir, 'dataset.yaml'), 'r') as file:
        dataset_info = yaml.safe_load(file)
    return dataset_info['names']

class_names = load_dataset_info()
print("Class Names:", class_names)

# Count occurrences in original labels (YOLO format: "class x_center y_center width height")
class_counts = {i: 0 for i in range(len(class_names))}
images_with_class = {i: 0 for i in range(len(class_names))}

for label_file in label_files:
    with open(os.path.join(train_labels_dir, label_file), 'r') as file:
        labels = file.readlines()
        present_classes = set()
        for label in labels:
            cls = int(float(label.split()[0]))  # Handles cases like '0.0'
            class_counts[cls] += 1
            present_classes.add(cls)
        for cls in present_classes:
            images_with_class[cls] += 1

# Plot original image distribution
plt.figure(figsize=(8, 6))
plt.bar([class_names[i] for i in range(len(class_names))], list(images_with_class.values()))
plt.xlabel('Class Name')
plt.ylabel('Number of Images Containing the Class')
plt.title('Original: Class-wise Image Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot original bounding box distribution
plt.figure(figsize=(8, 6))
plt.bar([class_names[i] for i in range(len(class_names))], list(class_counts.values()))
plt.xlabel('Class Name')
plt.ylabel('Number of Bounding Boxes')
plt.title('Original: Bounding Box Count per Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================ 4. Data Augmentation ============================ #
# Directories for augmented data

augmented_images_dir = os.path.join(root_dir, 'train', 'images_augmented')
augmented_labels_dir = os.path.join(root_dir, 'train', 'labels_augmented')
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)
"""
# Underrepresented class indices (assume "Car" (index 2) is overrepresented)
underrepresented_ids = [0, 1, 3, 4]
augmentation_factor = 3  # Number of augmented copies per eligible image

# Augmentation pipeline (with bbox transformation using YOLO format)
augmentation_pipeline = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.GaussianBlur(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.5))

def augment_image_and_labels(image_path, label_path, augmentation_pipeline, aug_factor):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return
    
    bboxes = []
    category_ids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        category_ids.append(cls)
    
    # Only augment if at least one object belongs to an underrepresented class
    if not any(cls in underrepresented_ids for cls in category_ids):
        return
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(aug_factor):
        augmented = augmentation_pipeline(image=image_rgb, bboxes=bboxes, category_ids=category_ids)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_category_ids = augmented['category_ids']
        
        # Save augmented image
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        aug_image_name = f"{basename}_aug_{i+1}.jpg"
        aug_image_path = os.path.join(augmented_images_dir, aug_image_name)
        cv2.imwrite(aug_image_path, aug_image_bgr)
        
        # Save updated labels in YOLO format
        aug_label_name = f"{basename}_aug_{i+1}.txt"
        aug_label_path = os.path.join(augmented_labels_dir, aug_label_name)
        with open(aug_label_path, 'w') as f_out:
            for cls, bbox in zip(aug_category_ids, aug_bboxes):
                bbox_str = " ".join(f"{v:.6f}" for v in bbox)
                f_out.write(f"{cls} {bbox_str}\n")

# Process all training images for augmentation
train_image_files = sorted([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for image_file in tqdm(train_image_files, desc="Augmenting Training Images"):
    image_path = os.path.join(train_images_dir, image_file)
    label_path = os.path.join(train_labels_dir, os.path.splitext(image_file)[0] + '.txt')
    augment_image_and_labels(image_path, label_path, augmentation_pipeline, augmentation_factor)
"""
# ============================ 5. Post-Augmentation Distribution ============================ #
# Combine label files from original and augmented directories
orig_label_files = [os.path.join(train_labels_dir, f) for f in os.listdir(train_labels_dir) if f.lower().endswith('.txt')]
aug_label_files = [os.path.join(augmented_labels_dir, f) for f in os.listdir(augmented_labels_dir) if f.lower().endswith('.txt')]
all_label_files = orig_label_files + aug_label_files

# Combined bounding box counts
combined_class_counts = {i: 0 for i in range(len(class_names))}
for label_file in all_label_files:
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                cls = int(float(line.split()[0]))
                combined_class_counts[cls] += 1

plt.figure(figsize=(8, 6))
plt.bar([class_names[i] for i in range(len(class_names))], list(combined_class_counts.values()))
plt.xlabel("Class")
plt.ylabel("Number of Bounding Boxes")
plt.title("Bounding Box Distribution (Original + Augmented)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Combined image counts (each image is counted once per class)
combined_image_counts = {i: 0 for i in range(len(class_names))}
for label_file in all_label_files:
    with open(label_file, 'r') as f:
        classes_in_image = set()
        for line in f:
            if line.strip():
                cls = int(float(line.split()[0]))
                classes_in_image.add(cls)
    for cls in classes_in_image:
        combined_image_counts[cls] += 1

plt.figure(figsize=(8, 6))
plt.bar([class_names[i] for i in range(len(class_names))], list(combined_image_counts.values()))
plt.xlabel("Class")
plt.ylabel("Number of Images Containing the Class")
plt.title("Image Distribution (Original + Augmented)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================ 6. Model Building & Training ============================ #
model = YOLO('yolov8l.yaml')

# Uncomment the following line to start training:
dataset_yaml = os.path.join(root_dir, 'dataset.yaml')
model.train(data=dataset_yaml, epochs=50,batch=16,workers=8)

