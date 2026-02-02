# AI-Based Driver Identity Verification and Authorization System

A real-time academic prototype for biometric driver authentication using facial recognition, liveness detection, and automated alerting.

## 🎯 Overview

This system demonstrates a complete biometric authentication pipeline suitable for academic research and experimental evaluation. It captures live facial images, verifies identity against enrolled drivers using FaceNet embeddings, performs liveness detection to prevent spoofing, and triggers automated alerts for unauthorized access attempts.

## ✨ Key Features

- **Real-Time Biometric Verification**: Live camera-based facial recognition using FaceNet (128-dimensional embeddings)
- **Liveness Detection**: Eye Aspect Ratio (EAR) based blink detection to prevent photo-based attacks
- **Automated Alerting**: Email notifications with captured images for unauthorized access attempts
- **Performance Logging**: CSV-based logging with detailed metrics (similarity scores, processing times)
- **Threshold Tuning**: Configurable similarity threshold to balance security and usability
- **Privacy-First Design**: Local-only data storage, no cloud transmission
- **Sub-1.5s Latency**: Optimized for real-time performance on standard laptops

## 🏗️ System Architecture

```
┌─────────────────┐
│  Camera Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Face Detection  │ (MTCNN)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Liveness Check  │ (EAR Blink Detection)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FaceNet Embed.  │ (128-dim vector)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similarity      │ (Cosine Similarity)
│ Comparison      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Authori-│ │Unauthor- │
│zed     │ │ized      │
└────────┘ └────┬─────┘
                │
                ▼
         ┌──────────────┐
         │ Email Alert  │
         └──────────────┘
```

## 📋 Requirements

### Hardware
- Webcam or mobile camera (USB or built-in)
- Standard laptop/desktop (CPU-based, no GPU required)
- Minimum 4GB RAM recommended

### Software
- Python 3.9 or higher
- Windows/Linux/macOS

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd DraverIdentityVerification_and_Authorization

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure email alerts (optional):

```bash
# Copy template
copy config\.env.example config\.env

# Edit config\.env with your SMTP credentials
# For Gmail, use an app-specific password:
# https://support.google.com/accounts/answer/185833
```

Edit `config/config.yaml` to adjust system parameters:
- `verification.similarity_threshold`: Adjust between 0.0-1.0 (default: 0.6)
- `camera.device_id`: Change if using external camera (default: 0)
- `camera.resolution_width/height`: Adjust camera resolution

### 3. Test System

```bash
# Run all tests
python scripts/test_system.py --all

# Or test individual components
python scripts/test_system.py --camera
python scripts/test_system.py --face-detection
python scripts/test_system.py --benchmark
```

### 4. Enroll Drivers

```bash
# Enroll a driver (interactive mode with camera preview)
python scripts/enroll_driver.py --name "John Doe" --email "john@example.com"

# Non-interactive mode (auto-capture)
python scripts/enroll_driver.py --name "Jane Smith" --no-interactive

# List enrolled drivers
python scripts/enroll_driver.py --list
```

### 5. Run Verification

```bash
# Start verification with live preview
python scripts/run_verification.py

# Run without video preview (headless)
python scripts/run_verification.py --no-preview

# Disable liveness detection (not recommended)
python scripts/run_verification.py --no-liveness

# Override similarity threshold
python scripts/run_verification.py --threshold 0.7
```

## 📁 Project Structure

```
DraverIdentityVerification_and_Authorization/
├── src/
│   ├── enrollment/          # Driver enrollment module
│   │   ├── camera_capture.py
│   │   ├── face_processor.py
│   │   └── enrollment_manager.py
│   ├── verification/        # Real-time verification engine
│   │   ├── video_stream.py
│   │   ├── liveness_detector.py
│   │   ├── face_matcher.py
│   │   └── verification_engine.py
│   ├── database/            # SQLite database layer
│   │   ├── db_manager.py
│   │   └── models.py
│   ├── alerting/            # Email alerts and logging
│   │   ├── email_service.py
│   │   └── logger.py
│   └── utils/               # Configuration management
│       └── config.py
├── scripts/                 # Command-line scripts
│   ├── enroll_driver.py
│   ├── run_verification.py
│   └── test_system.py
├── config/                  # Configuration files
│   ├── config.yaml
│   └── .env.example
├── data/                    # Data storage
│   ├── database/            # SQLite database
│   ├── logs/                # Performance logs (CSV)
│   └── alerts/              # Captured images
├── requirements.txt
└── README.md
```

## 🔧 Configuration Parameters

### Verification Settings
- `similarity_threshold` (0.6): Minimum similarity for authorization
  - Higher = More strict (fewer false accepts, more false rejects)
  - Lower = More lenient (more false accepts, fewer false rejects)
- `liveness_ear_threshold` (0.25): Eye Aspect Ratio threshold for blink detection
- `liveness_blink_frames` (3): Minimum consecutive frames for valid blink
- `max_processing_time_ms` (1500): Target processing time

### Camera Settings
- `resolution_width` (640): Camera width in pixels
- `resolution_height` (480): Camera height in pixels
- `fps` (30): Target frames per second
- `device_id` (0): Camera device ID (0 = default)

### Logging Settings
- `save_authorized_images` (false): Save images of authorized drivers
- `save_unauthorized_images` (true): Save images of unauthorized attempts
- `max_log_entries` (10000): Maximum log entries before rotation

## 📊 Performance Metrics

The system logs the following metrics for each verification attempt:

- **Timestamp**: Date and time of verification
- **Driver ID/Name**: Matched driver (if any)
- **Similarity Score**: Cosine similarity (0.0-1.0)
- **Authorization Status**: Authorized/Unauthorized
- **Liveness Status**: Passed/Failed
- **Processing Time**: End-to-end latency in milliseconds

Logs are stored in:
- **Database**: `data/database/drivers.db` (SQLite)
- **CSV**: `data/logs/verification_log.csv`

### Viewing Statistics

```python
from src.alerting.logger import PerformanceLogger

logger = PerformanceLogger()
logger.print_statistics()
```

## 🔐 Security & Privacy

### Data Protection
- ✅ All biometric data stored locally (no cloud transmission)
- ✅ SQLite database with file-level permissions
- ✅ Optional image saving (configurable)
- ✅ Email credentials stored in gitignored `.env` file

### Privacy-by-Design
- Biometric embeddings are one-way (cannot reconstruct original face)
- Enrollment requires explicit action (no passive collection)
- Configurable data retention policies
- Option to anonymize driver identifiers

### Ethical Considerations
- ⚠️ Obtain informed consent before enrolling individuals
- ⚠️ Use only for academic/research purposes
- ⚠️ Comply with local biometric data regulations (GDPR, CCPA, etc.)
- ⚠️ Implement appropriate access controls in production

## 🎓 Academic Use

### Suitable For
- Biometric authentication research
- Computer vision coursework
- Security systems prototyping
- Machine learning demonstrations
- Privacy-preserving authentication studies

### Evaluation Metrics
- **False Acceptance Rate (FAR)**: Unauthorized users accepted
- **False Rejection Rate (FRR)**: Authorized users rejected
- **Equal Error Rate (EER)**: Point where FAR = FRR
- **Processing Latency**: Time from capture to decision
- **Liveness Detection Accuracy**: Photo attacks blocked

### Threshold Tuning

Use the benchmark feature to find optimal threshold:

```python
from src.verification.face_matcher import FaceMatcher

matcher = FaceMatcher()

# Collect test embeddings with labels
test_data = [
    (driver_id, embedding, is_genuine),  # is_genuine = True/False
    # ... more test cases
]

results = matcher.benchmark_threshold(test_data)
print(f"Optimal threshold: {results['optimal_threshold']}")
print(f"FAR: {results['optimal_far']:.2%}")
print(f"FRR: {results['optimal_frr']:.2%}")
```

## 🐛 Troubleshooting

### Camera Not Found
```
ERROR: Could not open camera 0
```
**Solution**: Try different device IDs in `config/config.yaml`:
```yaml
camera:
  device_id: 1  # or 2, 3, etc.
```

### Face Detection Fails
```
No face detected
```
**Solutions**:
- Ensure good lighting conditions
- Position face centered in frame
- Move closer to camera (face should be 5-80% of frame)
- Check camera is not blocked

### Slow Performance
```
Processing time: 2500ms
```
**Solutions**:
- Close other applications
- Reduce camera resolution in config
- Disable liveness detection temporarily
- Check CPU usage

### Email Alerts Not Sending
```
WARNING: Email not configured
```
**Solutions**:
- Copy `config/.env.example` to `config/.env`
- Add Gmail credentials with app-specific password
- Verify SMTP settings in `config/config.yaml`

## 📚 Technical Details

### Face Detection
- **Algorithm**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Output**: Bounding box, confidence score, 5 facial landmarks
- **Minimum Confidence**: 0.95 (configurable)

### Facial Recognition
- **Model**: FaceNet (Inception ResNet v1)
- **Embedding Dimension**: 128
- **Distance Metric**: Cosine Similarity
- **Framework**: DeepFace

### Liveness Detection
- **Method**: Eye Aspect Ratio (EAR) based blink detection
- **Formula**: `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
- **Threshold**: 0.25 (eye considered closed if EAR < threshold)
- **Validation**: Minimum 3 consecutive frames

### Database Schema

**Drivers Table**:
```sql
CREATE TABLE drivers (
    driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    biometric_embedding BLOB NOT NULL,
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    email TEXT,
    status TEXT DEFAULT 'active'
);
```

**Verification Logs Table**:
```sql
CREATE TABLE verification_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    driver_id INTEGER,
    driver_name TEXT,
    similarity_score REAL,
    authorized BOOLEAN,
    processing_time_ms REAL,
    image_path TEXT,
    liveness_passed BOOLEAN
);
```

## 🤝 Contributing

This is an academic prototype. For improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for academic and research purposes only. Not intended for commercial deployment without proper security audits and compliance reviews.

## ⚠️ Disclaimer

This system is a **controlled academic prototype** designed for research and educational purposes. It is **NOT** production-ready and should **NOT** be deployed in real-world security-critical applications without:

- Comprehensive security audits
- Legal compliance reviews (GDPR, CCPA, biometric data laws)
- Robust anti-spoofing mechanisms (3D liveness detection)
- Encrypted data storage
- Access control and authentication
- Regular security updates

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Run system tests: `python scripts/test_system.py --all`
3. Review configuration in `config/config.yaml`
4. Check logs in `data/logs/`

## 🙏 Acknowledgments

- **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering"
- **MTCNN**: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- **DeepFace**: Serengil & Ozpinar, "LightFace: A Hybrid Deep Face Recognition Framework"
- **EAR**: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks"

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**Status**: Academic Prototype
