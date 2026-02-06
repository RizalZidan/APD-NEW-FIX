#!/usr/bin/env python3
"""
APD Monitoring System - Advanced Version
Features: Login, Multi-Camera, Data Recap
"""

import os
import sys
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, Response, request, redirect, url_for, session, flash
import cv2
import threading
import time
import base64
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.violations_detector import ViolationsDetector

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize database
def init_db():
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'operator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Violations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT DEFAULT 'inactive',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user
    admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, role) 
        VALUES (?, ?, ?)
    ''', ('admin', admin_password, 'admin'))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Global variables
detector = ViolationsDetector(confidence_threshold=0.3)
cameras = {}
camera_threads = {}
camera_stats = {}
global_stats = {
    'total_violations': 0,
    'no_helmet_count': 0,
    'no_vest_count': 0,
    'active_cameras': 0
}
tracked_persons = {}  # {camera_id: {person_id: {last_seen_time, violations}}}
detection_cooldown = 1.0

# Camera monitoring functions
def start_camera_monitoring(camera_id, camera_source):
    """Start monitoring a specific camera with enhanced RTSP support"""
    global cameras, camera_threads, tracked_persons
    
    if camera_id in cameras:
        return False
    
    print(f"🎯 Starting camera {camera_id} with source: {camera_source}")
    
    # Enhanced RTSP connection
    if camera_source.startswith('rtsp://'):
        # Try multiple RTSP connection methods
        cap = None
        
        # Method 1: Direct connection
        print(f"📡 Method 1: Direct RTSP connection...")
        cap = cv2.VideoCapture(camera_source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Direct RTSP connection successful!")
                cameras[camera_id] = cap
            else:
                cap.release()
                cap = None
        
        # Method 2: With TCP transport
        if cap is None:
            print(f"📡 Method 2: RTSP with TCP transport...")
            tcp_url = f"{camera_source}?transport=tcp"
            cap = cv2.VideoCapture(tcp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ TCP RTSP connection successful!")
                    cameras[camera_id] = cap
                else:
                    cap.release()
                    cap = None
        
        # Method 3: With UDP transport
        if cap is None:
            print(f"📡 Method 3: RTSP with UDP transport...")
            udp_url = f"{camera_source}?transport=udp"
            cap = cv2.VideoCapture(udp_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ UDP RTSP connection successful!")
                    cameras[camera_id] = cap
                else:
                    cap.release()
                    cap = None
        
        # Method 4: With extended timeout
        if cap is None:
            print(f"📡 Method 4: RTSP with extended timeout...")
            cap = cv2.VideoCapture(camera_source)
            if cap.isOpened():
                # Set extended timeout
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Extended timeout RTSP connection successful!")
                    cameras[camera_id] = cap
                else:
                    cap.release()
                    cap = None
        
        if cap is None:
            print(f"❌ All RTSP connection methods failed for {camera_source}")
            return False
            
    else:
        # Regular webcam/file connection
        cap = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)
        if not cap.isOpened():
            print(f"❌ Failed to open camera source: {camera_source}")
            return False
        cameras[camera_id] = cap
    
    # Safety: make sure camera handle really exists before starting thread
    if camera_id not in cameras:
        print(f"❌ Camera {camera_id} not initialized correctly, aborting start")
        return False
    
    tracked_persons[camera_id] = {}
    # camera_stats[camera_id] = {'fps': 0.0}
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_camera, args=(camera_id, cameras[camera_id]), daemon=True)
    camera_threads[camera_id] = thread
    thread.start()
    
    # Update camera status in database
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE cameras SET status = ? WHERE id = ?', ('active', camera_id))
    conn.commit()
    conn.close()
    
    print(f"✅ Camera {camera_id} monitoring started successfully!")
    return True

def stop_camera_monitoring(camera_id):
    """Stop monitoring a specific camera"""
    global cameras, camera_threads
    
    if camera_id in cameras:
        cameras[camera_id].release()
        del cameras[camera_id]
    
    if camera_id in camera_threads:
        del camera_threads[camera_id]
    
    if camera_id in tracked_persons:
        del tracked_persons[camera_id]
    
    if camera_id in camera_stats:
        del camera_stats[camera_id]
    
    # Update camera status in database
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE cameras SET status = ? WHERE id = ?', ('inactive', camera_id))
    conn.commit()
    conn.close()
    
    return True

def monitor_camera(camera_id, cap):
    """Monitor a single camera for violations"""
    global tracked_persons, global_stats
    
    frame_count = 0
    start_time = time.time()
    
    while camera_id in cameras and cameras[camera_id].isOpened():
        ret, frame = cameras[camera_id].read()
        if not ret:
            break
        
        frame_count += 1
        detections = detector.detect_violations(frame)
        
        # Process detections with person tracking
        current_time = time.time()
        if camera_id not in tracked_persons:
            tracked_persons[camera_id] = {}
        
        for detection in detections:
            class_name = detection['class']
            
            # Create person ID based on bounding box position
            bbox = detection.get('bbox', [0, 0, 0, 0])
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            person_id = f"{class_name}_{center_x}_{center_y}"
            
            # Check if this person exists
            if person_id not in tracked_persons[camera_id]:
                tracked_persons[camera_id][person_id] = {
                    'last_seen': current_time,
                    'violations': {
                        'no_helmet': False,
                        'no_vest': False
                    }
                }
            
            person_data = tracked_persons[camera_id][person_id]
            
            # Check if this is a new detection (after cooldown)
            if current_time - person_data['last_seen'] > detection_cooldown:
                
                # Update violation status
                if class_name == 'nohelmet':
                    if not person_data['violations']['no_helmet']:
                        person_data['violations']['no_helmet'] = True
                        
                        # Save to database
                        save_violation(camera_id, class_name, detection['confidence'], bbox)
                        
                        # Update global stats
                        global_stats['no_helmet_count'] += 1
                        global_stats['total_violations'] += 1
                
                elif class_name == 'novest':
                    if not person_data['violations']['no_vest']:
                        person_data['violations']['no_vest'] = True
                        
                        # Save to database
                        save_violation(camera_id, class_name, detection['confidence'], bbox)
                        
                        # Update global stats
                        global_stats['no_vest_count'] += 1
                        global_stats['total_violations'] += 1
                
                # Update last seen time
                person_data['last_seen'] = current_time
        
        # Update FPS for this camera every 30 frames
        if frame_count % 30 == 0 and camera_id in camera_stats:
            elapsed = time.time() - start_time
            if elapsed > 0:
                camera_stats[camera_id]['fps'] = round(frame_count / elapsed, 1)
        
        # Clean up old persons (not seen for 30 seconds)
        cleanup_time = current_time - 30
        persons_to_remove = []
        for pid, pdata in tracked_persons[camera_id].items():
            if pdata['last_seen'] < cleanup_time:
                persons_to_remove.append(pid)
        
        for pid in persons_to_remove:
            del tracked_persons[camera_id][pid]
        
        time.sleep(0.03)

def save_violation(camera_id, violation_type, confidence, bbox):
    """Save violation to database"""
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO violations (camera_id, violation_type, confidence, bbox) 
        VALUES (?, ?, ?, ?)
    ''', (camera_id, violation_type, confidence, str(bbox)))
    conn.commit()
    conn.close()

def generate_camera_feed(camera_id):
    """Generate video feed for a specific camera with enhanced RTSP support"""
    global cameras
    
    # IMPORTANT: keep it light.
    # Only stream frames for cameras that are actively started (exist in cameras dict).
    if camera_id not in cameras:
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Camera {camera_id} Inactive", (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(placeholder, "Click START to run", (180, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Stream frames if camera is available
    frame_count = 0
    start_time = time.time()
    
    while camera_id in cameras and cameras[camera_id].isOpened():
        ret, frame = cameras[camera_id].read()
        if ret:
            frame_count += 1
            
            # Draw detections on frame
            try:
                detections = detector.detect_violations(frame)
                frame_with_detections = detector.draw_violations(frame.copy(), detections)
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
                frame_with_detections = frame.copy()
            
            # Encode frame (tanpa overlay info)
            _, buffer = cv2.imencode('.jpg', frame_with_detections)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print(f"⚠️ Failed to read frame from camera {camera_id}")
            break
        
        time.sleep(0.033)  # ~30 FPS
    
    # Generate placeholder frame
    import numpy as np
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, f"Camera {camera_id} Inactive", (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder, f"Feed Disconnected", (160, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    _, buffer = cv2.imencode('.jpg', placeholder)
    frame_bytes = buffer.tobytes()
    
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APD Monitoring - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .login-container {
            background: #111;
            padding: 60px 40px;
            border: 2px solid #333;
            width: 450px;
            text-align: center;
            position: relative;
        }
        .login-container::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #fff, #000, #fff);
            z-index: -1;
        }
        .login-header {
            margin-bottom: 40px;
        }
        .login-header h1 {
            color: #fff;
            font-size: 32px;
            margin-bottom: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-weight: 700;
        }
        .login-header p {
            color: #888;
            font-size: 14px;
            letter-spacing: 1px;
        }
        .form-group {
            margin-bottom: 25px;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 10px;
            color: #fff;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 12px;
        }
        .form-group input {
            width: 100%;
            padding: 15px;
            background: #000;
            border: 2px solid #333;
            color: #fff;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .form-group input:focus {
            outline: none;
            border-color: #fff;
            background: #111;
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
        }
        .form-group input::placeholder {
            color: #555;
        }
        .login-btn {
            width: 100%;
            padding: 18px;
            background: #000;
            color: #fff;
            border: 2px solid #fff;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .login-btn:hover {
            background: #fff;
            color: #000;
            border-color: #fff;
        }
        .error-message {
            background: #000;
            color: #ff0000;
            padding: 12px;
            border: 2px solid #ff0000;
            margin-bottom: 25px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .scan-line {
            position: absolute;
            width: 100%;
            height: 1px;
            background: #fff;
            top: 0;
            left: 0;
            animation: scan 3s linear infinite;
            opacity: 0.1;
        }
        @keyframes scan {
            0% { top: 0; }
            100% { top: 100%; }
        }
        .corner {
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid #fff;
        }
        .corner-tl {
            top: 10px;
            left: 10px;
            border-right: none;
            border-bottom: none;
        }
        .corner-tr {
            top: 10px;
            right: 10px;
            border-left: none;
            border-bottom: none;
        }
        .corner-bl {
            bottom: 10px;
            left: 10px;
            border-right: none;
            border-top: none;
        }
        .corner-br {
            bottom: 10px;
            right: 10px;
            border-left: none;
            border-top: none;
        }
    </style>
</head>
<body>
    <div class="scan-line"></div>
    <div class="login-container">
        <div class="corner corner-tl"></div>
        <div class="corner corner-tr"></div>
        <div class="corner corner-bl"></div>
        <div class="corner corner-br"></div>
        
        <div class="login-header">
            <h1>APD MONITORING</h1>
            <p>System Access Required</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error-message">ERROR: {{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST" action="/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" placeholder="Enter Username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="Enter Password" required>
            </div>
            <button type="submit" class="login-btn">Access System</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APD Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Courier New', monospace;
            background: #000;
            color: #fff;
        }
        
        /* Header */
        .header {
            background: #111;
            padding: 15px 30px;
            border-bottom: 2px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }
        .header h1 {
            color: #fff;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header-camera-form {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-camera-form input,
        .header-camera-form select {
            padding: 6px 8px;
            font-size: 11px;
            border-radius: 3px;
            border: 1px solid #333;
            background: #000;
            color: #fff;
        }
        .header-camera-form button {
            padding: 6px 10px;
            font-size: 11px;
            border-radius: 3px;
        }
        .header-camera-extra {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .user-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .logout-btn {
            background: #000;
            color: #fff;
            padding: 8px 16px;
            border: 2px solid #fff;
            font-family: 'Courier New', monospace;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .logout-btn:hover {
            background: #fff;
            color: #000;
        }
        
        /* Layout with Sidebar Navigation */
        .layout-main {
            display: flex;
            min-height: calc(100vh - 70px);
        }
        .sidebar {
            width: 220px;
            background: #111;
            border-right: 2px solid #333;
            padding-top: 10px;
        }
        .nav-tabs {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .nav-tab {
            padding: 12px 20px;
            cursor: pointer;
            border-left: 3px solid transparent;
            color: #888;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .nav-tab.active {
            color: #fff;
            border-left-color: #fff;
            background: #000;
        }
        .nav-tab:hover {
            color: #fff;
            background: #000;
        }
        .content-area {
            flex: 1;
        }
        
        /* Container */
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
        }
        
        /* Tab Content */
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        
        /* Camera Grid */
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .view-mode-bar {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 10px;
            margin-bottom: 10px;
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .view-mode-bar select {
            background: #000;
            color: #fff;
            border: 1px solid #333;
            padding: 4px 6px;
            font-size: 12px;
        }
        .camera-card {
            background: #111;
            border: 2px solid #333;
            padding: 20px;
            position: relative;
        }
        .camera-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #fff, #000, #fff);
            z-index: -1;
        }
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .camera-title {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .camera-status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff0000;
        }
        .status-dot.active {
            background: #00ff00;
        }
        .status-text {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .video-container {
            position: relative;
            background: #000;
            border: 1px solid #333;
            overflow: hidden;
            aspect-ratio: 16/9;
        }
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .camera-controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .control-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #fff;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: #000;
            color: #fff;
        }
        .control-btn.start {
            border-color: #ffffff;
            color: #ffffff;
        }
        .control-btn.start:hover {
            background: #ffffff;
            color: #000;
        }
        .control-btn.stop {
            border-color: #ff0000;
            color: #ff0000;
        }
        .control-btn.stop:hover {
            background: #ff0000;
            color: #000;
        }
        .control-btn:hover {
            transform: translateY(-2px);
        }
        .camera-info {
            margin-top: 15px;
            padding: 10px;
            background: #000;
            border: 1px solid #333;
        }
        .info-text {
            color: #888;
            font-size: 11px;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        .info-text:last-child {
            margin-bottom: 0;
        }
        }
        .control-btn:hover {
            opacity: 0.8;
        }
        
        /* Statistics */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #111;
            border: 2px solid #333;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #fff, #000, #fff);
            z-index: -1;
        }
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .stat-label {
            color: #888;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Violations Table */
        .violations-table {
            background: #111;
            border: 2px solid #333;
            padding: 20px;
            position: relative;
        }
        .violations-table::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #fff, #000, #fff);
            z-index: -1;
        }
        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .table-title {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .export-btn {
            background: #000;
            color: #fff;
            padding: 8px 16px;
            border: 2px solid #fff;
            font-family: 'Courier New', monospace;
            font-weight: 600;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .export-btn:hover {
            background: #fff;
            color: #000;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
            font-family: 'Courier New', monospace;
        }
        th {
            background: #000;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        td {
            color: #fff;
        }
        .violation-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: #000;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .violation-badge.nohelmet {
            background: #ff0000;
        }
        .violation-badge.novest {
            background: #ffaa00;
        }
        
        /* Daily Stats Chart */
        .stats-section {
            margin-bottom: 30px;
        }
        .date-filter {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: #111;
            border: 1px solid #333;
        }
        .date-filter label {
            color: #fff;
            font-weight: 600;
        }
        .date-filter input {
            padding: 8px;
            background: #000;
            border: 1px solid #333;
            color: #fff;
        }
        .chart-container {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            border-radius: 8px;
        }
        .export-options {
            display: flex;
            gap: 10px;
        }
        .export-options .export-btn {
            padding: 8px 16px;
            font-size: 12px;
        }
        
        /* Add Camera Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
        }
        .modal-content {
            background: #111;
            border: 2px solid #fff;
            margin: 10% auto;
            padding: 30px;
            width: 400px;
            position: relative;
        }
        .modal-content::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #fff, #000, #fff);
            z-index: -1;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #fff;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 12px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            background: #000;
            border: 2px solid #333;
            color: #fff;
            font-family: 'Courier New', monospace;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #fff;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        .btn {
            padding: 10px 20px;
            border: 2px solid #fff;
            font-family: 'Courier New', monospace;
            font-weight: 600;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: #000;
            color: #fff;
        }
        .btn-primary:hover {
            background: #fff;
            color: #000;
        }
        .btn-secondary {
            background: #333;
            color: #fff;
            border-color: #333;
        }
        .btn-secondary:hover {
            background: #555;
        }
        
        /* Scan line effect */
        .scan-line {
            position: fixed;
            width: 100%;
            height: 1px;
            background: #fff;
            top: 0;
            left: 0;
            animation: scan 3s linear infinite;
            opacity: 0.05;
            z-index: 1;
            pointer-events: none;
        }
        @keyframes scan {
            0% { top: 0; }
            100% { top: 100%; }
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #2c3e50;
            font-weight: 600;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-primary {
            background: #3498db;
            color: white;
        }
        .btn-secondary {
            background: #95a5a6;
            color: white;
        }
    </style>
</head>
<body>
    <div class="scan-line"></div>
    <div class="header">
        <div class="header-left">
            <h1>APD MONITORING SYSTEM</h1>
            <!-- Simple Camera CRUD in Header -->
            <div class="header-camera-form">
                <input type="text" id="camera-name" placeholder="Name">
                <select id="camera-source" onchange="handleSourceChange()">
                    <option value="">Source</option>
                    <option value="0">Webcam 0</option>
                    <option value="1">Webcam 1</option>
                    <option value="2">Webcam 2</option>
                    <option value="3">Webcam 3</option>
                    <option value="rtsp://service:cctv@172.19.156.152/live.sdp">CCTV Main</option>
                    <option value="rtsp://service:cctv@172.19.156.152:554/live.sdp">CCTV Main :554</option>
                    <option value="rtsp">RTSP Custom</option>
                    <option value="file">Video File</option>
                </select>
                <div class="header-camera-extra" id="rtsp-group" style="display:none;">
                    <input type="text" id="rtsp-url" placeholder="rtsp://...">
                    <button class="btn btn-secondary" type="button" onclick="testRtsp()">Test</button>
                </div>
                <div class="header-camera-extra" id="file-group" style="display:none;">
                    <input type="text" id="file-path" placeholder="video.mp4">
                </div>
                <button class="btn btn-primary" type="button" onclick="addCamera()">Save</button>
                <button class="btn btn-secondary" type="button" onclick="resetCameraForm()">Reset</button>
            </div>
        </div>
        <div class="user-info">
            <span>USER: {{ session.username }}</span>
            <a href="/logout" class="logout-btn">Logout</a>
        </div>
    </div>
    
    <div class="layout-main">
        <div class="sidebar">
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="showTab('cameras')">Cameras</div>
                <div class="nav-tab" onclick="showTab('statistics')">Statistics</div>
                <div class="nav-tab" onclick="showTab('violations')">Violations</div>
                <div class="nav-tab" onclick="showTab('settings')">Settings</div>
            </div>
        </div>
        <div class="content-area">
            <div class="container">
                <!-- Cameras Tab -->
                <div id="cameras" class="tab-content active">
                    <div class="view-mode-bar">
                        <span>View Mode</span>
                        <select id="view-mode" onchange="changeViewMode()">
                            <option value="1">1 Cam</option>
                            <option value="2">2 Cam</option>
                            <option value="4" selected>4 Cam</option>
                            <option value="8">8 Cam</option>
                            <option value="16">16 Cam</option>
                        </select>
                    </div>
                    <div class="camera-grid" id="camera-grid">
                        <!-- Camera cards will be loaded here -->
                    </div>
                </div>
                
                <!-- Statistics Tab -->
                <div id="statistics" class="tab-content">
                    <!-- Daily Stats Chart -->
                    <div class="stats-section">
                        <div class="table-title">📊 Daily Statistics</div>
                        <div class="date-filter">
                            <label>Date Range:</label>
                            <input type="date" id="start-date" onchange="updateDailyStats()">
                            <span>to</span>
                            <input type="date" id="end-date" onchange="updateDailyStats()">
                            <button class="btn-secondary" onclick="updateDailyStats()">Update</button>
                        </div>
                        <div class="chart-container">
                            <canvas id="daily-chart" width="800" height="400"></canvas>
                        </div>
                    </div>
                    
                    <!-- Summary Stats -->
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="total-violations">0</div>
                            <div class="stat-label">Total Violations</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="no-helmet-count">0</div>
                            <div class="stat-label">No Helmets</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="no-vest-count">0</div>
                            <div class="stat-label">No Vests</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="active-cameras">0</div>
                            <div class="stat-label">Active Cameras</div>
                        </div>
                    </div>
                </div>
                
                <!-- Violations Tab -->
                <div id="violations" class="tab-content">
                    <div class="violations-table">
                        <div class="table-header">
                            <div class="table-title">📋 Violation Log</div>
                            <div class="export-options">
                                <button class="export-btn" onclick="exportData('pdf')">📄 PDF</button>
                                <button class="export-btn" onclick="exportData('excel')">📊 Excel</button>
                                <button class="export-btn" onclick="exportData('csv')">📄 CSV</button>
                            </div>
                        </div>
                        <div class="date-filter">
                            <label>Date Range:</label>
                            <input type="date" id="violation-start-date" onchange="loadViolations()">
                            <span>to</span>
                            <input type="date" id="violation-end-date" onchange="loadViolations()">
                            <button class="btn-secondary" onclick="loadViolations()">Filter</button>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Date</th>
                                    <th>Camera</th>
                                    <th>Violation Type</th>
                                    <th>Person ID</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="violations-tbody">
                                <!-- Violations will be loaded here -->
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Settings Tab -->
                <div id="settings" class="tab-content">
                    <div class="violations-table">
                        <div class="table-title">⚙️ System Settings</div>
                        <div class="form-group">
                            <label>Detection Cooldown (seconds)</label>
                            <input type="number" id="cooldown-setting" value="5" min="1" max="60">
                        </div>
                        <div class="form-group">
                            <label>Confidence Threshold</label>
                            <input type="number" id="confidence-setting" value="0.3" min="0.1" max="1.0" step="0.1">
                        </div>
                        <button class="control-btn start" onclick="saveSettings()">Save Settings</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentTab = 'cameras';
        let editingCameraId = null;
        let currentViewMode = 4;  // 1,2,4,8,16 cams
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
            
            // Load tab-specific data
            if (tabName === 'cameras') {
                loadCameras();
            } else if (tabName === 'statistics') {
                loadStatistics();
            } else if (tabName === 'violations') {
                loadViolations();
            }
        }
        
        function loadCameras() {
            fetch('/api/cameras')
                .then(r => r.json())
                .then(data => {
                    const grid = document.getElementById('camera-grid');
                    grid.innerHTML = '';

                    // Atur jumlah kolom grid berdasarkan view mode
                    let cols = 1;
                    if (currentViewMode >= 2) cols = 2;
                    if (currentViewMode >= 4) cols = 2;  // 2x2 untuk 4 cam
                    if (currentViewMode >= 8) cols = 4;  // 4 per baris untuk 8/16 cam
                    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(250px, 1fr))`;

                    // Pilih kamera yang akan ditampilkan berdasarkan view mode
                    const camerasToShow = data.cameras.slice(0, currentViewMode);
                    
                    camerasToShow.forEach(camera => {
                        const card = createCameraCard(camera);
                        grid.innerHTML += card;
                    });
                });
        }
        
        function createCameraCard(camera) {
            return `
                <div class="camera-card">
                    <div class="camera-header">
                        <div class="camera-title">${camera.name}</div>
                        <div class="camera-status">
                            <div class="status-dot ${camera.status === 'active' ? 'active' : ''}"></div>
                            <div class="status-text">${camera.status.toUpperCase()}</div>
                        </div>
                    </div>
                    <div class="video-container">
                        <img class="video-feed" src="/camera_feed/${camera.id}" alt="${camera.name}">
                    </div>
                    <div class="camera-info">
                        <div class="info-text">Source: ${camera.source}</div>
                        <div class="info-text">ID: CAM-${camera.id}</div>
                    </div>
                    <div class="camera-controls">
                        <button class="control-btn start" onclick="startCamera(${camera.id})">Start</button>
                        <button class="control-btn stop" onclick="stopCamera(${camera.id})">Stop</button>
                        <button class="control-btn" onclick='editCamera(${camera.id}, ${JSON.stringify(camera.name)}, ${JSON.stringify(camera.source)})'>Edit</button>
                        <button class="control-btn stop" onclick="deleteCamera(${camera.id})">Delete</button>
                    </div>
                </div>
            `;
        }
        
        function loadStatistics() {
            fetch('/api/statistics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('total-violations').textContent = data.total_violations;
                    document.getElementById('no-helmet-count').textContent = data.no_helmet_count;
                    document.getElementById('no-vest-count').textContent = data.no_vest_count;
                    document.getElementById('active-cameras').textContent = data.active_cameras;
                    
                    // Load daily stats
                    updateDailyStats();
                });
        }
        
        function updateDailyStats() {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            fetch(`/api/daily_stats?start=${startDate}&end=${endDate}`)
                .then(r => r.json())
                .then(data => {
                    drawChart(data);
                });
        }
        
        function drawChart(data) {
            const canvas = document.getElementById('daily-chart');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Simple bar chart
            const dates = Object.keys(data);
            const helmetCounts = dates.map(date => data[date].no_helmet || 0);
            const vestCounts = dates.map(date => data[date].no_vest || 0);
            
            const maxValue = Math.max(...helmetCounts, ...vestCounts, 1);
            const barWidth = 60;
            const barSpacing = 20;
            const chartHeight = 300;
            const chartStartY = 50;
            
            // Draw axes
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(50, chartStartY);
            ctx.lineTo(50, chartStartY + chartHeight);
            ctx.lineTo(750, chartStartY + chartHeight);
            ctx.stroke();
            
            // Draw bars
            dates.forEach((date, index) => {
                const x = 80 + index * (barWidth * 2 + barSpacing);
                const helmetHeight = (helmetCounts[index] / maxValue) * chartHeight;
                const vestHeight = (vestCounts[index] / maxValue) * chartHeight;
                
                // No Helmet bar
                ctx.fillStyle = '#ff0000';
                ctx.fillRect(x, chartStartY + chartHeight - helmetHeight, barWidth, helmetHeight);
                
                // No Vest bar
                ctx.fillStyle = '#ffaa00';
                ctx.fillRect(x + barWidth, chartStartY + chartHeight - vestHeight, barWidth, vestHeight);
                
                // Date label
                ctx.fillStyle = '#fff';
                ctx.font = '10px Courier New';
                ctx.fillText(date, x, chartStartY + chartHeight + 20);
            });
            
            // Legend
            ctx.fillStyle = '#ff0000';
            ctx.fillRect(600, 20, 15, 15);
            ctx.fillStyle = '#fff';
            ctx.fillText('No Helmet', 620, 32);
            
            ctx.fillStyle = '#ffaa00';
            ctx.fillRect(600, 40, 15, 15);
            ctx.fillText('No Vest', 620, 52);
        }
        
        function loadViolations() {
            const startDate = document.getElementById('violation-start-date').value;
            const endDate = document.getElementById('violation-end-date').value;
            
            let url = '/api/violations';
            if (startDate && endDate) {
                url += `?start=${startDate}&end=${endDate}`;
            }
            
            fetch(url)
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('violations-tbody');
                    tbody.innerHTML = '';
                    
                    data.violations.forEach(violation => {
                        const row = document.createElement('tr');
                        const date = new Date(violation.timestamp);
                        const personId = `PERSON-${Math.floor(Math.random() * 10000)}`;
                        
                        row.innerHTML = `
                            <td>${date.toLocaleTimeString()}</td>
                            <td>${date.toLocaleDateString()}</td>
                            <td>${violation.camera_name || 'Camera ' + violation.camera_id}</td>
                            <td><span class="violation-badge ${violation.violation_type}">${violation.violation_type.replace('no', 'No ').toUpperCase()}</span></td>
                            <td>${personId}</td>
                            <td>${(violation.confidence * 100).toFixed(1)}%</td>
                        `;
                        tbody.appendChild(row);
                    });
                });
        }
        
        function exportData(format) {
            const startDate = document.getElementById('violation-start-date').value;
            const endDate = document.getElementById('violation-end-date').value;
            
            let url = `/api/export?format=${format}`;
            if (startDate && endDate) {
                url += `&start=${startDate}&end=${endDate}`;
            }
            
            window.open(url);
        }
        
        
        function changeViewMode() {
            const select = document.getElementById('view-mode');
            currentViewMode = parseInt(select.value, 10) || 1;
            if (currentTab === 'cameras') {
                loadCameras();
            }
        }
        
        function startCamera(cameraId) {
            fetch(`/api/camera/${cameraId}/start`, {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadCameras();
                    }
                });
        }
        
        function stopCamera(cameraId) {
            fetch(`/api/camera/${cameraId}/stop`, {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadCameras();
                    }
                });
        }
        
        function handleSourceChange() {
            const source = document.getElementById('camera-source').value;
            const rtspGroup = document.getElementById('rtsp-group');
            const fileGroup = document.getElementById('file-group');
            
            // Hide all optional groups
            rtspGroup.style.display = 'none';
            fileGroup.style.display = 'none';
            
            // Show relevant group based on selection
            if (source === 'rtsp') {
                rtspGroup.style.display = 'block';
            } else if (source === 'file') {
                fileGroup.style.display = 'block';
            }
        }
        
        function testRtsp() {
            const url = document.getElementById('rtsp-url').value;
            if (!url) {
                alert('Masukkan RTSP URL terlebih dahulu');
                return;
            }
            
            fetch('/api/test_rtsp', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert(data.message || 'RTSP connection OK');
                    } else {
                        alert(data.message || 'Gagal koneksi ke RTSP stream');
                    }
                })
                .catch(() => {
                    alert('Terjadi error saat mengetes RTSP');
                });
        }
        
        function addCamera() {
            const name = document.getElementById('camera-name').value;
            let source = document.getElementById('camera-source').value;
            
            // Handle different source types
            if (source === 'rtsp') {
                source = document.getElementById('rtsp-url').value;
            } else if (source === 'file') {
                source = document.getElementById('file-path').value;
            }
            
            if (!name || !source) {
                alert('Please fill in all required fields');
                return;
            }
            
            const payload = {name, source};
            let url = '/api/cameras';
            let method = 'POST';
            
            if (editingCameraId !== null) {
                url = `/api/camera/${editingCameraId}`;
                method = 'PUT';
            }
            
            fetch(url, {
                method: method,
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        editingCameraId = null;
                        resetCameraForm();
                        loadCameras();
                    } else {
                        alert(data.error || 'Failed to save camera');
                    }
                });
        }
        
        function resetCameraForm() {
            editingCameraId = null;
            document.getElementById('camera-name').value = '';
            document.getElementById('camera-source').value = '';
            document.getElementById('rtsp-url').value = '';
            document.getElementById('file-path').value = '';
            handleSourceChange();
        }
        
        function editCamera(id, name, source) {
            editingCameraId = id;
            document.getElementById('camera-name').value = name;
            
            const sourceSelect = document.getElementById('camera-source');
            const options = Array.from(sourceSelect.options).map(o => o.value);
            
            if (options.includes(String(source))) {
                sourceSelect.value = String(source);
                document.getElementById('rtsp-url').value = '';
                document.getElementById('file-path').value = '';
            } else if (String(source).startsWith('rtsp://')) {
                sourceSelect.value = 'rtsp';
                document.getElementById('rtsp-url').value = source;
                document.getElementById('file-path').value = '';
            } else {
                sourceSelect.value = 'file';
                document.getElementById('file-path').value = source;
                document.getElementById('rtsp-url').value = '';
            }
            
            handleSourceChange();
        }
        
        function deleteCamera(id) {
            if (!confirm('Are you sure you want to delete this camera?')) {
                return;
            }
            
            fetch(`/api/camera/${id}`, {method: 'DELETE'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        if (editingCameraId === id) {
                            resetCameraForm();
                        }
                        loadCameras();
                    } else {
                        alert(data.error || 'Failed to delete camera');
                    }
                });
        }
        
        function exportViolations() {
            window.open('/api/violations/export');
        }
        
        function saveSettings() {
            const cooldown = document.getElementById('cooldown-setting').value;
            const confidence = document.getElementById('confidence-setting').value;
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cooldown, confidence})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('Settings saved successfully!');
                    }
                });
        }
        
        // Auto-refresh
        setInterval(() => {
            if (currentTab === 'statistics') {
                loadStatistics();
            } else if (currentTab === 'violations') {
                loadViolations();
            }
        }, 5000);
        
        // Initialize
        loadCameras();
        loadStatistics();
    </script>
</body>
</html>
"""

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('apd_monitoring.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and user[2] == hashlib.sha256(password.encode()).hexdigest():
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_TEMPLATE)

# API routes
@app.route('/api/cameras')
def get_cameras():
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM cameras ORDER BY id')
    db_cameras = cursor.fetchall()
    conn.close()
    
    camera_list = []
    for cam in db_cameras:
        # Check if camera is actually running in the global cameras dict
        is_active = cam[0] in cameras
        fps = camera_stats.get(cam[0], {}).get('fps', 0.0)
        camera_list.append({
            'id': cam[0],
            'name': cam[1],
            'source': cam[2],
            'status': 'active' if is_active else 'inactive',
            'created_at': cam[4],
            'fps': fps
        })
    
    return jsonify({'cameras': camera_list})

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    data = request.get_json()
    name = data.get('name')
    source = data.get('source')
    
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO cameras (name, source) VALUES (?, ?)', (name, source))
    camera_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Keep camera inactive by default (lighter load). User can start manually.
    return jsonify({'success': True, 'camera_id': camera_id})

@app.route('/api/camera/<int:camera_id>/start', methods=['POST'])
def start_camera(camera_id):
    # Get camera info from database
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('SELECT source FROM cameras WHERE id = ?', (camera_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'success': False, 'error': 'Camera not found'})
    
    camera_source = result[0]
    success = start_camera_monitoring(camera_id, camera_source)
    
    if success:
        global_stats['active_cameras'] = len(cameras)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to start camera'})

@app.route('/api/camera/<int:camera_id>/stop', methods=['POST'])
def stop_camera(camera_id):
    success = stop_camera_monitoring(camera_id)
    
    if success:
        global_stats['active_cameras'] = len(cameras)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to stop camera'})

@app.route('/api/test_rtsp', methods=['POST'])
def test_rtsp():
    """Test a given RTSP URL before adding camera"""
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    
    if not url:
        return jsonify({'success': False, 'message': 'RTSP URL is required'}), 400
    
    methods = [
        ('Direct', url),
        ('TCP', f"{url}?transport=tcp"),
        ('UDP', f"{url}?transport=udp")
    ]
    
    for method_name, test_url in methods:
        try:
            cap = cv2.VideoCapture(test_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    cap.release()
                    return jsonify({
                        'success': True,
                        'message': f'{method_name} connection OK ({width}x{height})',
                        'method': method_name,
                        'width': width,
                        'height': height
                    })
            if cap:
                cap.release()
        except Exception as e:
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
    
    return jsonify({
        'success': False,
        'message': 'Failed to connect to RTSP stream with tested methods'
    }), 500

@app.route('/api/camera/<int:camera_id>', methods=['PUT', 'POST', 'DELETE'])
def camera_detail(camera_id):
    """Update or delete a single camera (CRUD support)"""
    global global_stats
    
    if request.method in ['PUT', 'POST']:
        data = request.get_json() or {}
        name = data.get('name')
        source = data.get('source')
        
        if not name or not source:
            return jsonify({'success': False, 'error': 'Name and source are required'}), 400
        
        # Update camera info in database
        conn = sqlite3.connect('apd_monitoring.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE cameras SET name = ?, source = ? WHERE id = ?', (name, source, camera_id))
        conn.commit()
        conn.close()
        
        # If camera is currently running, stop it so user can start again with new config
        if camera_id in cameras:
            stop_camera_monitoring(camera_id)
            global_stats['active_cameras'] = len(cameras)
        
        return jsonify({'success': True})
    
    # DELETE
    success = stop_camera_monitoring(camera_id)
    
    # Remove camera from database
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    conn.commit()
    conn.close()
    
    global_stats['active_cameras'] = len(cameras)
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to delete camera'}), 500

@app.route('/api/daily_stats')
def get_daily_stats():
    """Get daily statistics for chart"""
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    
    if start_date and end_date:
        cursor.execute('''
            SELECT DATE(timestamp) as date, 
                   violation_type, 
                   COUNT(*) as count
            FROM violations 
            WHERE DATE(timestamp) BETWEEN ? AND ?
            GROUP BY DATE(timestamp), violation_type
        ''', (start_date, end_date))
    else:
        # Last 7 days
        cursor.execute('''
            SELECT DATE(timestamp) as date, 
                   violation_type, 
                   COUNT(*) as count
            FROM violations 
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp), violation_type
        ''')
    
    results = cursor.fetchall()
    conn.close()
    
    # Organize data by date
    daily_data = {}
    for date, violation_type, count in results:
        if date not in daily_data:
            daily_data[date] = {'no_helmet': 0, 'no_vest': 0}
        
        if violation_type == 'nohelmet':
            daily_data[date]['no_helmet'] = count
        elif violation_type == 'novest':
            daily_data[date]['no_vest'] = count
    
    return jsonify(daily_data)

@app.route('/api/export')
def export_data():
    """Export violation data in different formats"""
    format_type = request.args.get('format', 'csv')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    
    query = '''
        SELECT v.*, c.name as camera_name
        FROM violations v
        LEFT JOIN cameras c ON v.camera_id = c.id
    '''
    params = []
    
    if start_date and end_date:
        query += ' WHERE DATE(v.timestamp) BETWEEN ? AND ?'
        params.extend([start_date, end_date])
    
    query += ' ORDER BY v.timestamp DESC'
    cursor.execute(query, params)
    violations = cursor.fetchall()
    conn.close()
    
    if format_type == 'csv':
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Time', 'Date', 'Camera', 'Violation Type', 'Person ID', 'Confidence'])
        
        for v in violations:
            writer.writerow([
                v[4],  # timestamp
                v[4][:10],  # date part
                v[6] or f"Camera {v[1]}",  # camera name
                v[2].replace('no', 'No ').title(),  # violation type
                f"PERSON-{v[0]}",  # person ID
                f"{v[3]*100:.1f}%"  # confidence
            ])
        
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=violations.csv'}
        )
        return response
    
    elif format_type == 'excel':
        # Simple Excel format (tab-separated)
        output = "Time\tDate\tCamera\tViolation Type\tPerson ID\tConfidence\n"
        
        for v in violations:
            output += f"{v[4]}\t{v[4][:10]}\t{v[6] or f'Camera {v[1]}'}\t{v[2].replace('no', 'No ').title()}\tPERSON-{v[0]}\t{v[3]*100:.1f}%\n"
        
        response = app.response_class(
            output,
            mimetype='text/tab-separated-values',
            headers={'Content-Disposition': 'attachment; filename=violations.txt'}
        )
        return response
    
    elif format_type == 'pdf':
        # Simple PDF-like text format
        output = "APD VIOLATION REPORT\n"
        output += "=" * 50 + "\n\n"
        
        if start_date and end_date:
            output += f"Period: {start_date} to {end_date}\n\n"
        
        output += f"Total Violations: {len(violations)}\n\n"
        output += "-" * 30 + "\n"
        
        for v in violations:
            output += f"Time: {v[4]}\n"
            output += f"Camera: {v[6] or f'Camera {v[1]}'}\n"
            output += f"Type: {v[2].replace('no', 'No ').title()}\n"
            output += f"Person: PERSON-{v[0]}\n"
            output += f"Confidence: {v[3]*100:.1f}%\n"
            output += "-" * 20 + "\n"
        
        response = app.response_class(
            output,
            mimetype='text/plain',
            headers={'Content-Disposition': 'attachment; filename=violations_report.txt'}
        )
        return response
    
    return jsonify({'error': 'Unsupported format'}), 400

@app.route('/api/statistics')
def get_statistics():
    global global_stats
    
    # Also get violation counts from database for accuracy
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('SELECT violation_type, COUNT(*) FROM violations GROUP BY violation_type')
    violation_counts = dict(cursor.fetchall())
    
    # Get active cameras count
    cursor.execute("SELECT COUNT(*) FROM cameras WHERE status = 'active'")
    active_cameras_db = cursor.fetchone()[0]
    conn.close()
    
    return jsonify({
        'total_violations': global_stats['total_violations'],
        'no_helmet_count': global_stats['no_helmet_count'],
        'no_vest_count': global_stats['no_vest_count'],
        'active_cameras': global_stats['active_cameras']
    })

@app.route('/api/violations')
def get_violations():
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT v.*, c.name as camera_name 
        FROM violations v 
        LEFT JOIN cameras c ON v.camera_id = c.id 
        ORDER BY v.timestamp DESC 
        LIMIT 100
    ''')
    violations = cursor.fetchall()
    conn.close()
    
    violation_list = []
    for v in violations:
        violation_list.append({
            'id': v[0],
            'camera_id': v[1],
            'violation_type': v[2],
            'confidence': v[3],
            'timestamp': v[5],
            'camera_name': v[7] or 'Unknown',
            'processed': bool(v[6])
        })
    
    return jsonify({'violations': violation_list})

@app.route('/api/violations/export')
def export_violations():
    conn = sqlite3.connect('apd_monitoring.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT v.*, c.name as camera_name 
        FROM violations v 
        LEFT JOIN cameras c ON v.camera_id = c.id 
        ORDER BY v.timestamp DESC
    ''')
    violations = cursor.fetchall()
    conn.close()
    
    # Generate CSV
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time', 'Camera', 'Type', 'Confidence', 'Status'])
    
    for v in violations:
        writer.writerow([
            v[5],  # timestamp
            v[7] or 'Unknown',  # camera_name
            v[2],  # violation_type
            f"{v[3]:.2f}",  # confidence
            'Processed' if v[6] else 'Pending'  # processed
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=violations.csv'}
    )

@app.route('/camera_feed/<int:camera_id>')
def camera_feed(camera_id):
    return Response(generate_camera_feed(camera_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Starting Advanced APD Monitoring System...")
    print("🔐 Features: Login, Multi-Camera, Data Recap")
    print("📊 Open http://localhost:5000 in your browser")
    print("👤 Default Login: admin / admin123")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
