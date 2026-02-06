"""
Violation Logger Module
Handles logging and management of APD violations
"""

import cv2
import os
from datetime import datetime
import json
from pathlib import Path

class ViolationLogger:
    def __init__(self, log_dir="logs", violation_images_dir="violations"):
        """
        Initialize Violation Logger
        
        Args:
            log_dir: Directory for log files
            violation_images_dir: Directory for violation images
        """
        self.log_dir = log_dir
        self.violation_images_dir = violation_images_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(violation_images_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, f"violations_{datetime.now().strftime('%Y%m%d')}.json")
        
        print("✅ Violation Logger initialized")
        print(f"📁 Log directory: {log_dir}")
        print(f"📸 Violation images: {violation_images_dir}")
    
    def log_violation(self, person_id, violation_type, confidence, bbox, frame, camera_id=None):
        """
        Log APD violation
        
        Args:
            person_id: Identified worker ID (None if unknown)
            violation_type: Type of violation ('no_helmet', 'no_vest')
            confidence: Detection confidence
            bbox: Bounding box [x1, y1, x2, y2]
            frame: Video frame
            camera_id: Camera identifier
        """
        timestamp = datetime.now()
        
        # Generate unique violation ID
        violation_id = f"VIO_{timestamp.strftime('%Y%m%d_%H%M%S')}_{id(frame)}"
        
        # Save violation image
        image_path = self._save_violation_image(frame, bbox, violation_id, timestamp)
        
        # Create violation record
        violation_record = {
            'violation_id': violation_id,
            'person_id': person_id,
            'violation_type': violation_type,
            'confidence': float(confidence),
            'timestamp': timestamp.isoformat(),
            'bbox': [float(x) for x in bbox],
            'image_path': image_path,
            'camera_id': camera_id,
            'resolved': False,
            'notes': None
        }
        
        # Write to log file
        self._write_to_log(violation_record)
        
        # Also save as individual JSON file for easy access
        individual_log_path = os.path.join(self.log_dir, f"{violation_id}.json")
        with open(individual_log_path, 'w') as f:
            json.dump(violation_record, f, indent=2)
        
        print(f"🚨 Violation logged: {violation_type} by {person_id or 'Unknown'}")
        
        return violation_id
    
    def _save_violation_image(self, frame, bbox, violation_id, timestamp):
        """
        Save violation image with annotation
        
        Args:
            frame: Video frame
            bbox: Bounding box
            violation_id: Unique violation ID
            timestamp: Timestamp
            
        Returns:
            Path to saved image
        """
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add timestamp and violation info
        timestamp_text = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated_frame, timestamp_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add violation ID
        cv2.putText(annotated_frame, violation_id, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Generate filename
        date_str = timestamp.strftime('%Y%m')
        date_dir = os.path.join(self.violation_images_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        filename = f"{violation_id}.jpg"
        image_path = os.path.join(date_dir, filename)
        
        # Save image
        cv2.imwrite(image_path, annotated_frame)
        
        return image_path
    
    def _write_to_log(self, violation_record):
        """Write violation record to log file"""
        try:
            # Read existing logs
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            
            # Add new violation
            logs.append(violation_record)
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error writing to log file: {e}")
    
    def get_violations_by_date(self, date):
        """
        Get violations for specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of violation records
        """
        log_file = os.path.join(self.log_dir, f"violations_{date.replace('-', '')}.json")
        
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                violations = json.load(f)
            
            # Filter by date
            filtered_violations = [
                v for v in violations 
                if v['timestamp'].startswith(date)
            ]
            
            return filtered_violations
        except Exception as e:
            print(f"❌ Error reading violations: {e}")
            return []
    
    def get_violations_by_worker(self, worker_id, start_date=None, end_date=None):
        """
        Get violations for specific worker
        
        Args:
            worker_id: Worker ID
            start_date: Start date (YYYY-MM-DD) optional
            end_date: End date (YYYY-MM-DD) optional
            
        Returns:
            List of violation records
        """
        all_violations = []
        
        # Get all log files in date range
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_str = current_date.strftime('%Y%m%d')
                log_file = os.path.join(self.log_dir, f"violations_{date_str}.json")
                
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            violations = json.load(f)
                            all_violations.extend(violations)
                    except:
                        pass
                
                current_date += datetime.timedelta(days=1)
        else:
            # Get all available log files
            for log_file in Path(self.log_dir).glob("violations_*.json"):
                try:
                    with open(log_file, 'r') as f:
                        violations = json.load(f)
                        all_violations.extend(violations)
                except:
                    pass
        
        # Filter by worker
        worker_violations = [
            v for v in all_violations 
            if v.get('person_id') == worker_id
        ]
        
        return sorted(worker_violations, key=lambda x: x['timestamp'], reverse=True)
    
    def generate_report(self, start_date=None, end_date=None):
        """
        Generate violation report
        
        Args:
            start_date: Start date (YYYY-MM-DD) optional
            end_date: End date (YYYY-MM-DD) optional
            
        Returns:
            Formatted report string
        """
        all_violations = []
        
        # Collect violations from log files
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_str = current_date.strftime('%Y-%m-%d')
                violations = self.get_violations_by_date(date_str)
                all_violations.extend(violations)
                current_date += datetime.timedelta(days=1)
        else:
            # Get today's violations
            today = datetime.now().strftime('%Y-%m-%d')
            all_violations = self.get_violations_by_date(today)
        
        if not all_violations:
            return "📊 No violations found in the specified period."
        
        # Generate statistics
        total_violations = len(all_violations)
        unique_workers = len(set(v.get('person_id', 'Unknown') for v in all_violations))
        helmet_violations = len([v for v in all_violations if v['violation_type'] == 'no_helmet'])
        vest_violations = len([v for v in all_violations if v['violation_type'] == 'no_vest'])
        
        # Group by worker
        worker_stats = {}
        for violation in all_violations:
            worker = violation.get('person_id', 'Unknown')
            if worker not in worker_stats:
                worker_stats[worker] = {
                    'total': 0,
                    'helmet': 0,
                    'vest': 0,
                    'last_violation': None
                }
            
            worker_stats[worker]['total'] += 1
            if violation['violation_type'] == 'no_helmet':
                worker_stats[worker]['helmet'] += 1
            elif violation['violation_type'] == 'no_vest':
                worker_stats[worker]['vest'] += 1
            
            if (worker_stats[worker]['last_violation'] is None or 
                violation['timestamp'] > worker_stats[worker]['last_violation']):
                worker_stats[worker]['last_violation'] = violation['timestamp']
        
        # Format report
        report = f"""
📊 APD VIOLATION REPORT
{'='*50}
Period: {start_date or 'Today'} to {end_date or 'Today'}

📈 SUMMARY:
• Total Violations: {total_violations}
• Unique Workers: {unique_workers}
• Helmet Violations: {helmet_violations} ({helmet_violations/total_violations*100:.1f}%)
• Vest Violations: {vest_violations} ({vest_violations/total_violations*100:.1f}%)

👥 WORKER BREAKDOWN:
"""
        
        # Sort workers by violation count
        sorted_workers = sorted(worker_stats.items(), 
                              key=lambda x: x[1]['total'], reverse=True)
        
        for worker, stats in sorted_workers[:10]:  # Top 10
            last_violation = datetime.fromisoformat(stats['last_violation'])
            report += f"""
• {worker}:
  - Total: {stats['total']} violations
  - Helmet: {stats['helmet']}, Vest: {stats['vest']}
  - Last Violation: {last_violation.strftime('%Y-%m-%d %H:%M')}
"""
        
        return report
    
    def mark_resolved(self, violation_id, notes=None):
        """
        Mark violation as resolved
        
        Args:
            violation_id: Violation ID
            notes: Resolution notes
        """
        # Find and update violation in log files
        for log_file in Path(self.log_dir).glob("violations_*.json"):
            try:
                with open(log_file, 'r') as f:
                    violations = json.load(f)
                
                updated = False
                for violation in violations:
                    if violation['violation_id'] == violation_id:
                        violation['resolved'] = True
                        violation['notes'] = notes
                        updated = True
                        break
                
                if updated:
                    with open(log_file, 'w') as f:
                        json.dump(violations, f, indent=2)
                    
                    # Also update individual file
                    individual_file = os.path.join(self.log_dir, f"{violation_id}.json")
                    if os.path.exists(individual_file):
                        with open(individual_file, 'r') as f:
                            violation_data = json.load(f)
                        violation_data['resolved'] = True
                        violation_data['notes'] = notes
                        with open(individual_file, 'w') as f:
                            json.dump(violation_data, f, indent=2)
                    
                    print(f"✅ Violation {violation_id} marked as resolved")
                    return True
            except:
                continue
        
        print(f"❌ Violation {violation_id} not found")
        return False
    
    def cleanup_old_logs(self, days_to_keep=30):
        """
        Clean up old log files
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        cutoff_date = datetime.now() - datetime.timedelta(days=days_to_keep)
        
        for log_file in Path(self.log_dir).glob("violations_*.json"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace('violations_', '')
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    print(f"🗑️  Deleted old log file: {log_file}")
            except:
                continue
        
        # Also clean up old violation images
        for date_dir in Path(self.violation_images_dir).iterdir():
            if date_dir.is_dir():
                try:
                    date_str = date_dir.name
                    file_date = datetime.strptime(date_str, '%Y%m')
                    
                    if file_date < cutoff_date:
                        import shutil
                        shutil.rmtree(date_dir)
                        print(f"🗑️  Deleted old violation images: {date_dir}")
                except:
                    continue
