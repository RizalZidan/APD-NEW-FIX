"""
Object Detector Module using YOLOv8
Handles object detection for persons and APD items
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize Object Detector with YOLOv8 model
        
        Args:
            model_path: Path to trained YOLOv8 model
        """
        if model_path is None:
            # Check for violations model first
            violations_model_path = "violations_detection/yolov8n_violations/weights/best.pt"
            if os.path.exists(violations_model_path):
                print(f"🎯 Loading APD Violations model from {violations_model_path}")
                self.model = YOLO(violations_model_path)
                self.use_person_detection = False
                self.class_names = {0: 'no_helmet', 1: 'no_vest'}
            else:
                # Check for original APD model
                apd_model_path = "helmet.v2i.yolov8/helmet_vest_detection/yolov8n_50epochs_augmented/weights/best.pt"
                if os.path.exists(apd_model_path):
                    print(f"🎯 Loading APD model from {apd_model_path}")
                    self.model = YOLO(apd_model_path)
                    self.use_person_detection = False
                    self.class_names = {0: 'helmet', 1: 'vest'}
                else:
                    # Use default YOLOv8 model for person detection
                    print("⚠️  Using default YOLOv8n model for person detection")
                    self.model = YOLO('yolov8n.pt')
                    self.use_person_detection = True
                    self.class_names = {0: 'person'}
        else:
            if os.path.exists(model_path):
                print(f"🎯 Loading model from {model_path}")
                self.model = YOLO(model_path)
                self.use_person_detection = False
                # Auto-detect class names based on model
                if 'violations' in model_path:
                    self.class_names = {0: 'no_helmet', 1: 'no_vest'}
                else:
                    self.class_names = {0: 'helmet', 1: 'vest'}
            else:
                print(f"⚠️  Model not found at {model_path}, using YOLOv8n pretrained")
                self.model = YOLO('yolov8n.pt')
                self.use_person_detection = True
                self.class_names = {0: 'person'}
        
        self.confidence_threshold = 0.5
        
        print("✅ Object Detector initialized")
        print(f"🎯 Detection mode: {'Person Detection' if self.use_person_detection else 'APD Violations Detection'}")
        print(f"📊 Classes: {list(self.class_names.values())}")
    
    def detect_objects(self, frame):
        """
        Detect objects in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections with bbox, class, confidence
        """
        if self.use_person_detection:
            # Detect persons only
            results = self.model(frame, conf=self.confidence_threshold, classes=[0])  # Class 0 is person
        else:
            # Detect helmets and vests
            results = self.model(frame, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': class_name,
                        'confidence': float(confidence),
                        'class_id': class_id
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def detect_persons(self, frame):
        """
        Detect only persons in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of person detections
        """
        results = self.model(frame, conf=self.confidence_threshold, classes=[0])  # Class 0 is person
        
        persons = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    person = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': 'person'
                    }
                    
                    persons.append(person)
        
        return persons
    
    def detect_violations(self, frame):
        """
        Detect APD violations in frame (no_helmet, no_vest)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of violation detections
        """
        if self.use_person_detection:
            # If using person detection, return empty (no violations detected)
            return []
        
        # Detect violations directly
        results = self.model(frame, conf=self.confidence_threshold)
        
        violations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    # Only include violation classes
                    if class_name in ['no_helmet', 'no_vest']:
                        violation = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': class_name,
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'severity': 'medium'  # Default severity
                        }
                        
                        violations.append(violation)
        
        return violations
    
    def set_confidence_threshold(self, threshold):
        """
        Set confidence threshold for detection
        
        Args:
            threshold: Confidence threshold (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 Confidence threshold set to {self.confidence_threshold}")
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'person_detection' if self.use_person_detection else 'apd_detection',
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.class_names,
            'classes_detected': len(self.class_names)
        }
