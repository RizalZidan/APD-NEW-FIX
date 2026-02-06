"""
Database Manager Module
Handles SQLite database operations using separated table definitions
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from .database_tables import DatabaseTables

class DatabaseManager:
    def __init__(self, db_path="data/apd_monitoring.db"):
        """
        Initialize Database Manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Initialize database tables
        self.db_tables = DatabaseTables(db_path)
        
        # Create indexes for performance
        self.db_tables.create_indexes()
        
        print("✅ Database Manager initialized")
        print(f"📂 Database: {self.db_path}")
    
    def add_worker(self, worker_id, name, department="", phone="", email=""):
        """Add a new worker to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO workers 
                (worker_id, name, department, registration_date, status, phone, email, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (worker_id, name, department, datetime.now().isoformat(), 'active', phone, email, datetime.now().isoformat()))
            
            conn.commit()
            print(f"✅ Worker {worker_id} added/updated")
    
    def get_worker(self, worker_id):
        """Get worker information by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM workers WHERE worker_id = ?', (worker_id,))
            worker = cursor.fetchone()
            
            if worker:
                return {
                    'worker_id': worker[0],
                    'name': worker[1],
                    'department': worker[2],
                    'registration_date': worker[3],
                    'status': worker[4],
                    'phone': worker[5],
                    'email': worker[6],
                    'face_features_path': worker[7]
                }
            return None
    
    def get_all_workers(self):
        """Get all workers from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM workers ORDER BY registration_date DESC')
            workers = cursor.fetchall()
            
            return workers
    
    def add_violation(self, worker_id, violation_type, confidence, bbox, camera_id=""):
        """Add a violation record to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            cursor.execute('''
                INSERT INTO violations 
                (worker_id, violation_type, confidence, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, camera_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (worker_id, violation_type, confidence, datetime.now().isoformat(), 
                  bbox_x1, bbox_y1, bbox_x2, bbox_y2, camera_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_violations(self, start_date=None, end_date=None, worker_id=None, limit=None):
        """Get violations with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT v.*, w.name as worker_name 
                FROM violations v 
                LEFT JOIN workers w ON v.worker_id = w.worker_id
                WHERE 1=1
            '''
            params = []
            
            if start_date:
                query += ' AND v.timestamp >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND v.timestamp <= ?'
                params.append(end_date)
            
            if worker_id:
                query += ' AND v.worker_id = ?'
                params.append(worker_id)
            
            query += ' ORDER BY v.timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            violations = cursor.fetchall()
            
            return violations
    
    def get_violation_statistics(self, start_date=None, end_date=None):
        """Get violation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    COUNT(*) as total_violations,
                    SUM(CASE WHEN violation_type = 'no_helmet' THEN 1 ELSE 0 END) as helmet_violations,
                    SUM(CASE WHEN violation_type = 'no_vest' THEN 1 ELSE 0 END) as vest_violations,
                    SUM(CASE WHEN violation_type = 'both_violations' THEN 1 ELSE 0 END) as both_violations,
                    AVG(confidence) as avg_confidence
                FROM violations
                WHERE 1=1
            '''
            params = []
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date)
            
            cursor.execute(query, params)
            stats = cursor.fetchone()
            
            return {
                'total_violations': stats[0] or 0,
                'helmet_violations': stats[1] or 0,
                'vest_violations': stats[2] or 0,
                'both_violations': stats[3] or 0,
                'avg_confidence': stats[4] or 0.0
            }
    
    def create_monitoring_session(self, session_id, camera_id=""):
        """Create a new monitoring session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monitoring_sessions 
                (session_id, start_time, camera_id)
                VALUES (?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), camera_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_monitoring_session(self, session_id, total_detections=0, total_violations=0, 
                                total_persons=0, compliant_persons=0):
        """Update monitoring session statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE monitoring_sessions 
                SET total_detections = ?, total_violations = ?, 
                    total_persons = ?, compliant_persons = ?,
                    end_time = ?
                WHERE session_id = ?
            ''', (total_detections, total_violations, total_persons, compliant_persons,
                  datetime.now().isoformat(), session_id))
            
            conn.commit()
    
    def log_detection(self, session_id, frame_number, total_persons, total_helmets, 
                      total_vests, compliant_persons, violations_detected, 
                      processing_time, camera_id=""):
        """Log detection details"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detection_logs 
                (session_id, timestamp, frame_number, total_persons, total_helmets, 
                 total_vests, compliant_persons, violations_detected, processing_time, camera_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), frame_number, total_persons, 
                  total_helmets, total_vests, compliant_persons, violations_detected, 
                  processing_time, camera_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_apd_item(self, detection_log_id, item_type, confidence, bbox, person_id="", camera_id=""):
        """Log detected APD item"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            cursor.execute('''
                INSERT INTO apd_items 
                (detection_log_id, item_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                 person_id, timestamp, camera_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (detection_log_id, item_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                  person_id, datetime.now().isoformat(), camera_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_daily_statistics(self, date, stats):
        """Update daily statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_statistics 
                (date, total_workers, active_workers, total_violations, helmet_violations, 
                 vest_violations, both_violations, compliance_rate, total_detections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, stats.get('total_workers', 0), stats.get('active_workers', 0),
                  stats.get('total_violations', 0), stats.get('helmet_violations', 0),
                  stats.get('vest_violations', 0), stats.get('both_violations', 0),
                  stats.get('compliance_rate', 0.0), stats.get('total_detections', 0)))
            
            conn.commit()
    
    def export_data(self, export_dir="exports"):
        """Export database data to CSV files"""
        os.makedirs(export_dir, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Export workers
            cursor.execute('SELECT * FROM workers')
            self._export_to_csv(cursor.fetchall(), os.path.join(export_dir, 'workers.csv'))
            
            # Export violations
            cursor.execute('SELECT * FROM violations')
            self._export_to_csv(cursor.fetchall(), os.path.join(export_dir, 'violations.csv'))
            
            # Export monitoring sessions
            cursor.execute('SELECT * FROM monitoring_sessions')
            self._export_to_csv(cursor.fetchall(), os.path.join(export_dir, 'monitoring_sessions.csv'))
            
            # Export daily statistics
            cursor.execute('SELECT * FROM daily_statistics')
            self._export_to_csv(cursor.fetchall(), os.path.join(export_dir, 'daily_statistics.csv'))
        
        print(f"✅ Data exported to {export_dir}")
        return True
    
    def _export_to_csv(self, data, filename):
        """Export data to CSV file"""
        import csv
        
        if not data:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f'Column_{i}' for i in range(len(data[0]))])  # Header
            writer.writerows(data)
    
    def get_database_info(self):
        """Get database information"""
        return self.db_tables.get_database_stats()
    
    def backup_database(self, backup_path=None):
        """Create database backup"""
        return self.db_tables.backup_database(backup_path)
