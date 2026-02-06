"""
Monitoring System Module
Handles real-time monitoring statistics and display
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
import time

class MonitoringSystem:
    def __init__(self):
        """Initialize Monitoring System"""
        self.session_start_time = datetime.now()
        self.total_detections = 0
        self.total_violations = 0
        self.violation_types = {
            'no_helmet': 0,
            'no_vest': 0,
            'helmet_ok': 0,
            'vest_ok': 0
        }
        self.recent_violations = []
        self.active_workers = set()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✅ Monitoring System initialized")
    
    def update_detection(self, detections, recognized_workers=None):
        """
        Update monitoring statistics with new detections
        
        Args:
            detections: List of detection results
            recognized_workers: List of recognized worker IDs
        """
        self.total_detections += len(detections)
        
        # Update violation counts
        for detection in detections:
            violation_type = detection['class']
            if violation_type in self.violation_types:
                self.violation_types[violation_type] += 1
            
            # Track violations
            if violation_type in ['no_helmet', 'no_vest']:
                self.total_violations += 1
                self.recent_violations.append({
                    'type': violation_type,
                    'timestamp': datetime.now(),
                    'confidence': detection['confidence']
                })
                
                # Keep only recent violations (last 10)
                if len(self.recent_violations) > 10:
                    self.recent_violations.pop(0)
        
        # Update active workers
        if recognized_workers:
            self.active_workers.update(recognized_workers)
        
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def display_stats(self, frame):
        """
        Display monitoring statistics on frame
        
        Args:
            frame: Video frame to draw on
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Calculate session duration
        session_duration = datetime.now() - self.session_start_time
        duration_str = str(session_duration).split('.')[0]  # Remove microseconds
        
        # Display header
        cv2.putText(frame, "APD MONITORING SYSTEM", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display session info (without FPS to keep view clean)
        cv2.putText(frame, f"Session: {duration_str}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display detection stats
        cv2.putText(frame, "DETECTION STATISTICS:", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Total Detections: {self.total_detections}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Active Workers: {len(self.active_workers)}", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display violation stats
        cv2.putText(frame, "VIOLATION COUNTS:", (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Total Violations: {self.total_violations}", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"No Helmet: {self.violation_types['no_helmet']}", (20, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"No Vest: {self.violation_types['no_vest']}", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display recent violations
        if self.recent_violations:
            recent_y = h - 100
            cv2.putText(frame, "RECENT VIOLATIONS:", (20, recent_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            
            for i, violation in enumerate(self.recent_violations[-3:]):  # Show last 3
                recent_y += 20
                time_str = violation['timestamp'].strftime('%H:%M:%S')
                text = f"{time_str} - {violation['type']} ({violation['confidence']:.2f})"
                cv2.putText(frame, text, (20, recent_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display status indicator
        status_color = (0, 255, 0) if self.total_violations == 0 else (0, 0, 255)
        status_text = "ALL CLEAR" if self.total_violations == 0 else f"{self.total_violations} VIOLATIONS"
        cv2.putText(frame, status_text, (w - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame
    
    def get_session_summary(self):
        """Get current session summary"""
        session_duration = datetime.now() - self.session_start_time
        
        summary = {
            'session_duration': str(session_duration).split('.')[0],
            'total_detections': self.total_detections,
            'total_violations': self.total_violations,
            'active_workers': len(self.active_workers),
            'violation_types': self.violation_types.copy(),
            'average_fps': self.current_fps,
            'violations_per_hour': 0
        }
        
        # Calculate violations per hour
        if session_duration.total_seconds() > 0:
            hours = session_duration.total_seconds() / 3600
            summary['violations_per_hour'] = self.total_violations / hours if hours > 0 else 0
        
        return summary
    
    def reset_session(self):
        """Reset monitoring session"""
        self.session_start_time = datetime.now()
        self.total_detections = 0
        self.total_violations = 0
        self.violation_types = {
            'no_helmet': 0,
            'no_vest': 0,
            'helmet_ok': 0,
            'vest_ok': 0
        }
        self.recent_violations = []
        self.active_workers = set()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("🔄 Monitoring session reset")
    
    def export_session_data(self, filename=None):
        """
        Export session data to file
        
        Args:
            filename: Output filename (optional)
        """
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_data_{timestamp}.json"
        
        session_data = {
            'session_info': {
                'start_time': self.session_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': str(datetime.now() - self.session_start_time).split('.')[0]
            },
            'statistics': self.get_session_summary(),
            'recent_violations': [
                {
                    'type': v['type'],
                    'timestamp': v['timestamp'].isoformat(),
                    'confidence': v['confidence']
                }
                for v in self.recent_violations
            ],
            'active_workers': list(self.active_workers)
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📄 Session data exported to {filename}")
        return filename
    
    def set_alert_threshold(self, violations_per_minute):
        """
        Set alert threshold for violations
        
        Args:
            violations_per_minute: Alert threshold
        """
        self.alert_threshold = violations_per_minute
        print(f"🚨 Alert threshold set to {violations_per_minute} violations/minute")
    
    def check_alert_conditions(self):
        """
        Check if alert conditions are met
        
        Returns:
            Boolean indicating if alert should be triggered
        """
        if not hasattr(self, 'alert_threshold'):
            return False
        
        # Count violations in last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_violations_count = sum(
            1 for v in self.recent_violations 
            if v['timestamp'] > one_minute_ago
        )
        
        return recent_violations_count >= self.alert_threshold
    
    def get_worker_activity(self, worker_id):
        """
        Get activity summary for specific worker
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Worker activity summary
        """
        worker_violations = [
            v for v in self.recent_violations 
            if hasattr(v, 'worker_id') and v.get('worker_id') == worker_id
        ]
        
        return {
            'worker_id': worker_id,
            'is_active': worker_id in self.active_workers,
            'recent_violations': len(worker_violations),
            'last_violation': worker_violations[-1]['timestamp'] if worker_violations else None
        }
    
    def generate_realtime_report(self):
        """Generate real-time monitoring report"""
        summary = self.get_session_summary()
        
        report = f"""
🔴 REAL-TIME MONITORING REPORT
{'='*40}
Session Duration: {summary['session_duration']}
Current FPS: {summary['average_fps']:.1f}

📊 STATISTICS:
• Total Detections: {summary['total_detections']}
• Total Violations: {summary['total_violations']}
• Active Workers: {summary['active_workers']}
• Violations/Hour: {summary['violations_per_hour']:.1f}

🚨 VIOLATION BREAKDOWN:
• No Helmet: {summary['violation_types']['no_helmet']}
• No Vest: {summary['violation_types']['no_vest']}
• Helmet OK: {summary['violation_types']['helmet_ok']}
• Vest OK: {summary['violation_types']['vest_ok']}

⏰ RECENT ACTIVITY:
"""
        
        for violation in self.recent_violations[-5:]:  # Last 5 violations
            time_str = violation['timestamp'].strftime('%H:%M:%S')
            report += f"• {time_str} - {violation['type']} ({violation['confidence']:.2f})\n"
        
        return report
