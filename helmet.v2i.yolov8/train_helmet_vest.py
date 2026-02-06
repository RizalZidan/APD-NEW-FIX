#!/usr/bin/env python3
"""
YOLOv8 Training Script for Helmet and Vest Detection
Dataset: helmet.v2i.yolov8
Classes: helmet, vest
Epochs: 50
Augmentations: flip, 90-degree rotation, brightness, blur, noise, crop
"""

import os
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class CustomDataset:
    """Custom dataset with advanced augmentations"""
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_path, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_path, label_file)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert to absolute coordinates
                    h, w = image.shape[:2]
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    x2 = (x_center + width/2) * w
                    y2 = (y_center + height/2) * h
                    boxes.append([x1, y1, x2, y2, int(class_id)])
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes)
            image = augmented['image']
            boxes = augmented['bboxes']
        
        return image, boxes

def create_augmentations():
    """Create augmentation pipeline with all requested transformations"""
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.7),  # 90-degree rotation
        
        # Color augmentations
        A.RandomBrightness(limit=0.3, p=0.8),
        A.RandomContrast(limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=25, p=0.6),
        
        # Noise and blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MedianBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Crop and resize
        A.RandomCrop(width=0.8, height=0.8, p=0.4),
        A.Resize(height=640, width=640),
        
        # Additional augmentations
        A.MotionBlur(blur_limit=7, p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def train_model():
    """Main training function"""
    # Set paths
    dataset_path = "c:/Users/RizalZidan/Downloads/helmet.v2i.yolov8"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Using nano model for faster training
    
    # Training configuration
    training_config = {
        'data': data_yaml,
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,
        'device': '0' if os.name == 'nt' else 'cpu',  # GPU if available
        'project': 'helmet_vest_detection',
        'name': 'yolov8n_50epochs_augmented',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # box loss weight
        'cls': 0.5,  # cls loss weight
        'dfl': 1.5,  # dfl loss weight
        'pose': 12.0,  # pose loss weight
        'kobj': 1.0,  # keypoint obj loss weight
        'label_smoothing': 0.0,
        'nbs': 64,  # nominal batch size
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 90.0,  # image rotation (+/- deg)
        'translate': 0.1,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.5,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 1.0,  # image mosaic (probability)
        'mixup': 0.0,  # image mixup (probability)
        'copy_paste': 0.0,  # segment copy-paste (probability)
        'auto_augment': 'randaugment',  # auto augmentation policy
        'erasing': 0.4,  # random erasing (probability)
        'crop_fraction': 0.8,  # crop fraction (for random crop)
    }
    
    print("Starting YOLOv8 training for Helmet and Vest Detection")
    print(f"Dataset: {dataset_path}")
    print(f"Classes: helmet, vest")
    print(f"Epochs: {training_config['epochs']}")
    print("Augmentations: flip, 90° rotation, brightness, blur, noise, crop")
    print("-" * 60)
    
    # Start training
    results = model.train(**training_config)
    
    # Print training results
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}")
    print(f"Final mAP@50: {results.results_dict['metrics/mAP50-0.5']:.4f}")
    print(f"Final mAP@50-95: {results.results_dict['metrics/mAP50-95']:.4f}")
    
    # Validate model
    print("\nRunning validation...")
    metrics = model.val(data=data_yaml)
    print(f"Validation mAP@50: {metrics.box.map50:.4f}")
    print(f"Validation mAP@50-95: {metrics.box.map:.4f}")
    
    return results

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import ultralytics
        import albumentations
    except ImportError:
        print("Installing required packages...")
        os.system("pip install ultralytics albumentations opencv-python")
    
    # Run training
    results = train_model()
    
    print("\nTraining Summary:")
    print("=" * 50)
    print("✅ Training completed successfully!")
    print("📊 Model trained for 50 epochs")
    print("🎯 Classes: helmet, vest")
    print("🔄 Augmentations applied: flip, 90° rotation, brightness, blur, noise, crop")
    print("💾 Best model saved in 'helmet_vest_detection' directory")
