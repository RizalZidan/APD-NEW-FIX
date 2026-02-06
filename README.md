# APD Monitoring System
**Implementasi Algoritma YOLOv8 dan Face Recognition Berbasis Cosine Similarity untuk Deteksi dan Monitoring Pelanggar APD**

## 📋 Overview

Sistem ini dirancang untuk mendeteksi dan memonitoring pelanggaran Alat Pelindung Diri (APD) menggunakan kombinasi YOLOv8 untuk deteksi objek (helm dan rompi) dan Face Recognition dengan Cosine Similarity untuk identifikasi pekerja.

## 🎯 Fitur Utama

### Deteksi APD dengan YOLOv8
- Deteksi helm dan rompi secara real-time
- Model yang sudah dilatih dengan dataset khusus
- Confidence threshold yang dapat disesuaikan
- Support untuk multiple camera input

### Face Recognition dengan Cosine Similarity
- Identifikasi pekerja menggunakan wajah
- Cosine similarity untuk matching yang akurat
- Database internal untuk penyimpanan face encoding
- Threshold similarity yang dapat dikonfigurasi

### Sistem Monitoring
- Real-time violation detection
- Logging otomatis pelanggaran
- Statistik monitoring session
- Export data laporan

### Database Internal
- SQLite untuk data pekerja dan pelanggaran
- Tracking history pelanggaran
- Generate laporan per periode
- Export data ke CSV

## 📁 Struktur Proyek

```
APDNYELL/
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── apd_detector.py        # YOLOv8 APD detection
│   ├── face_recognition.py    # Face recognition system
│   ├── database_manager.py    # Database operations
│   ├── violation_logger.py    # Violation logging
│   └── monitoring_system.py   # Real-time monitoring
├── helmet.v2i.yolov8/         # YOLOv8 trained model
│   ├── helmet_vest_detection/
│   │   └── yolov8n_50epochs_augmented/
│   │       └── weights/
│   │           ├── best.pt    # Trained model
│   │           └── last.pt
│   ├── data.yaml              # Dataset configuration
│   └── train_helmet_vest.py   # Training script
├── data/                      # Data storage
│   ├── apd_monitoring.db      # SQLite database
│   └── face_database.pkl      # Face encodings
├── logs/                      # Log files
├── violations/                # Violation images
└── captures/                  # Manual captures
```

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd APDNYELL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Additional Requirements
```bash
# For face recognition on Windows
pip install cmake dlib

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 💻 Penggunaan

### 1. Run Main Application
```bash
python main.py
```

### 2. Menu Options:
1. **Start Real-time Monitoring** - Mulai monitoring dengan webcam atau video file
2. **Register New Worker** - Daftarkan pekerja baru dengan foto wajah
3. **Generate Violation Report** - Buat laporan pelanggaran
4. **List Violations** - Lihat daftar pelanggaran
5. **Exit** - Keluar dari aplikasi

### 3. Register Worker
Siapkan folder dengan foto-foto wajah pekerja:
```
workers/
├── worker_001/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── worker_002/
    ├── photo1.jpg
    └── photo2.jpg
```

## 🔧 Konfigurasi

### Model YOLOv8
Model sudah dilatih dengan dataset helmet dan vest:
- **Classes**: helmet, vest
- **Model**: YOLOv8 Nano
- **Training**: 50 epochs dengan augmentations

### Face Recognition Parameters
- **Similarity Threshold**: 0.6 (default)
- **Face Detection**: HOG method
- **Feature Extraction**: 128-dimensional face encoding

### Database Schema
- **workers**: Data pekerja
- **violations**: Record pelanggaran
- **monitoring_sessions**: Session monitoring
- **daily_statistics**: Statistik harian

## 📊 Laporan dan Statistik

Sistem menyediakan berbagai laporan:
- **Violation Summary**: Total pelanggaran per periode
- **Worker Statistics**: Pelanggaran per pekerja
- **Daily Reports**: Laporan harian otomatis
- **Session Summary**: Statistik per monitoring session

## 🎨 Contoh Output

### Real-time Display
```
🚨 APD MONITORING SYSTEM
Session: 00:45:23
FPS: 28.5
Total Detections: 156
Active Workers: 12
Total Violations: 8
No Helmet: 5
No Vest: 3
```

### Violation Report
```
📊 APD VIOLATION REPORT
==================================================
Period: 2024-01-15 to 2024-01-15

📈 SUMMARY:
• Total Violations: 24
• Unique Workers: 8
• Helmet Violations: 15 (62.5%)
• Vest Violations: 9 (37.5%)

👥 WORKER BREAKDOWN:
• Worker_001:
  - Total: 5 violations
  - Helmet: 3, Vest: 2
  - Last Violation: 2024-01-15 14:23
```

## 🔧 Customization

### Adjust Detection Threshold
```python
# In main.py
apd_detector.set_confidence_threshold(0.7)
face_recognition.set_similarity_threshold(0.8)
```

### Add New APD Classes
1. Update dataset dengan class baru
2. Retrain YOLOv8 model
3. Update class mapping di `apd_detector.py`

### Database Queries
```python
# Get violations by worker
violations = db_manager.get_violations(worker_id="WORKER_001")

# Get statistics
stats = db_manager.get_violation_statistics(
    start_date="2024-01-01", 
    end_date="2024-01-31"
)
```

## 🐛 Troubleshooting

### Common Issues
1. **Model not found**: Pastikan path model benar di `apd_detector.py`
2. **Face recognition error**: Install cmake dan dlib untuk Windows
3. **Database error**: Pastikan folder `data/` ada dan writable
4. **Camera not found**: Check camera index atau path video file

### Performance Tips
- Gunakan GPU untuk YOLOv8 (CUDA)
- Resize input images untuk processing lebih cepat
- Adjust confidence thresholds untuk balance accuracy/speed

## 📚 Referensi

### YOLOv8
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)

### Face Recognition
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

### Dataset
- [Roboflow Helmet Dataset](https://universe.roboflow.com/yusin/helmet-evxi3)

## 📄 Lisensi

MIT License - Lihat file LICENSE untuk detail

## 👥 Kontributor

- [Your Name] - Lead Developer
- [Advisor Name] - Academic Advisor

## 📧 Kontak

Untuk pertanyaan atau support:
- Email: your.email@example.com
- GitHub: https://github.com/yourusername/APDNYELL
