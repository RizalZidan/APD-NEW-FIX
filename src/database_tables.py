"""
Database Tables Module
Handles database table definitions and operations
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path

class DatabaseTables:
    def __init__(self, db_path="data/apd_monitoring.db"):
        """
        Initialize Database Tables Manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.create_data_directory()
        self.initialize_tables()
        
        print("✅ Database Tables initialized")
        print(f"📂 Database: {self.db_path}")
    
    def create_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def initialize_tables(self):
        """Initialize all database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create workers table
            self._create_workers_table(cursor)
            
            # Create violations table
            self._create_violations_table(cursor)
            
            # Create monitoring_sessions table
            self._create_monitoring_sessions_table(cursor)
            
            # Create daily_statistics table
            self._create_daily_statistics_table(cursor)
            
            # Create detection_logs table
            self._create_detection_logs_table(cursor)
            
            # Create apd_items table
            self._create_apd_items_table(cursor)
            
            conn.commit()
            print("📊 All database tables initialized")
    
    def _create_workers_table(self, cursor):
        """Create workers table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                registration_date TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                phone TEXT,
                email TEXT,
                face_features_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Workers table created")
    
    def _create_violations_table(self, cursor):
        """Create violations table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT,
                violation_type TEXT NOT NULL,
                confidence REAL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                camera_id TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (worker_id) REFERENCES workers (worker_id)
            )
        ''')
        print("✅ Violations table created")
    
    def _create_monitoring_sessions_table(self, cursor):
        """Create monitoring sessions table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                camera_id TEXT,
                total_detections INTEGER DEFAULT 0,
                total_violations INTEGER DEFAULT 0,
                total_persons INTEGER DEFAULT 0,
                compliant_persons INTEGER DEFAULT 0,
                session_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Monitoring sessions table created")
    
    def _create_daily_statistics_table(self, cursor):
        """Create daily statistics table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_workers INTEGER DEFAULT 0,
                active_workers INTEGER DEFAULT 0,
                total_violations INTEGER DEFAULT 0,
                helmet_violations INTEGER DEFAULT 0,
                vest_violations INTEGER DEFAULT 0,
                both_violations INTEGER DEFAULT 0,
                compliance_rate REAL DEFAULT 0.0,
                total_detections INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Daily statistics table created")
    
    def _create_detection_logs_table(self, cursor):
        """Create detection logs table for detailed tracking"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                frame_number INTEGER,
                total_persons INTEGER DEFAULT 0,
                total_helmets INTEGER DEFAULT 0,
                total_vests INTEGER DEFAULT 0,
                compliant_persons INTEGER DEFAULT 0,
                violations_detected INTEGER DEFAULT 0,
                processing_time REAL,
                camera_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES monitoring_sessions (session_id)
            )
        ''')
        print("✅ Detection logs table created")
    
    def _create_apd_items_table(self, cursor):
        """Create APD items table for tracking detected items"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS apd_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_log_id TEXT,
                item_type TEXT NOT NULL,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                person_id TEXT,
                timestamp TEXT NOT NULL,
                camera_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (detection_log_id) REFERENCES detection_logs (id)
            )
        ''')
        print("✅ APD items table created")
    
    def get_table_info(self):
        """
        Get information about all tables
        
        Returns:
            Dictionary with table information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_info[table_name] = {
                    'columns': [col[1] for col in columns],
                    'column_count': len(columns)
                }
            
            return table_info
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_violations_worker_id ON violations(worker_id)",
                "CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type)",
                "CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status)",
                "CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_detection_logs_session_id ON detection_logs(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_apd_items_type ON apd_items(item_type)",
                "CREATE INDEX IF NOT EXISTS idx_apd_items_timestamp ON apd_items(timestamp)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            print("✅ Database indexes created")
    
    def backup_database(self, backup_path=None):
        """
        Create a backup of the database
        
        Args:
            backup_path: Path for backup file (optional)
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup/apd_monitoring_backup_{timestamp}.db"
        
        # Create backup directory
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Create backup
        with sqlite3.connect(self.db_path) as source:
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)
        
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
    
    def get_database_stats(self):
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Get row counts for each table
            tables = ['workers', 'violations', 'monitoring_sessions', 'daily_statistics', 'detection_logs', 'apd_items']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database file size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            return stats
