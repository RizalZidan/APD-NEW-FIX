"""
APD Detector Module using YOLOv8
Simple violations detection for no_helmet and no_vest
"""

import cv2
import numpy as np
from .object_detector import ObjectDetector
from .apd_analyzer import APDAnalyzer
from .violations_detector import ViolationsDetector

class APDDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize APD Detector with YOLOv8 model and APD analyzer
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Confidence threshold for detection
        """
        # Initialize violations detector
        self.violations_detector = ViolationsDetector(confidence_threshold)
        
        # Initialize object detector (fallback)
        self.object_detector = ObjectDetector(model_path)
        self.object_detector.set_confidence_threshold(confidence_threshold)
        
        # Initialize APD analyzer
        self.apd_analyzer = APDAnalyzer(confidence_threshold)
        
        # Store configuration
        self.confidence_threshold = confidence_threshold
        
        print("✅ APD Detector initialized")
        print("🎯 Focus: Detecting persons as APD violations")
        print("📊 Violation types: no_helmet, no_vest")
        print("🎯 Confidence threshold set to", confidence_threshold)
    
    def detect(self, frame):
        """
        Detect APD violations in frame (focus on violations)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of violation detections with bbox, class, confidence, and violation info
        """
        # Detect both head and vest violations separately
        head_violations = self.violations_detector.detect_head_violations(frame)
        vest_violations = self.violations_detector.detect_vest_violations(frame)
        
        # Combine all violations
        all_violations = head_violations + vest_violations
        
        return all_violations
    
    def detect_all_apd(self, frame):
        """
        Detect all APD items (both compliant and violations)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of all APD detections
        """
        return self.violations_detector.detect_all_apd(frame)
    
    def draw_all_apd(self, frame, detections):
        """
        Draw all APD detections on frame
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        return self.violations_detector.draw_all_apd(frame, detections)
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels for violations on frame
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        return self.violations_detector.draw_violations(frame, detections)
    
    def summarize_violations(self, detections):
        """
        Summarize violation detections
        
        Args:
            detections: List of violation detections
            
        Returns:
            Dictionary with violation summary
        """
        summary = {
            'total_violations': len(detections),
            'no_helmet_count': 0,
            'no_vest_count': 0,
            'both_violations': 0
        }
        
        for detection in detections:
            if detection['class'] == 'no_helmet':
                summary['no_helmet_count'] += 1
            elif detection['class'] == 'no_vest':
                summary['no_vest_count'] += 1
        
        return summary
