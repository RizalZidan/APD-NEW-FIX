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

def train_model():
    """Main training function"""
    # Set paths
    dataset_path = "c:/Users/RizalZidan/Downloads/helmet.v2i.yolov8"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Using nano model for faster training
    
    # Training configuration with all requested augmentations
    training_config = {
        'data': data_yaml,
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,
        'device': '0',  # Use GPU if available
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
        
        # Augmentation parameters
        'hsv_h': 0.015,  # hue augmentation
        'hsv_s': 0.7,    # saturation augmentation  
        'hsv_v': 0.4,    # brightness/value augmentation
        'degrees': 90.0, # rotation (+/- 90 degrees)
        'translate': 0.1, # translation
        'scale': 0.5,    # scaling
        'shear': 0.0,    # shear
        'perspective': 0.0, # perspective
        'flipud': 0.5,   # vertical flip
        'fliplr': 0.5,   # horizontal flip
        'mosaic': 1.0,   # mosaic augmentation
        'mixup': 0.0,    # mixup
        'copy_paste': 0.0, # copy paste
        'auto_augment': 'randaugment', # auto augmentation
        'erasing': 0.4,  # random erasing (crop effect)
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
    # Run training
    results = train_model()
    
    print("\nTraining Summary:")
    print("=" * 50)
    print("✅ Training completed successfully!")
    print("📊 Model trained for 50 epochs")
    print("🎯 Classes: helmet, vest")
    print("🔄 Augmentations applied: flip, 90° rotation, brightness, blur, noise, crop")
    print("💾 Best model saved in 'helmet_vest_detection' directory")
