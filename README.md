# AI-Based Driver Identity Verification and Authorization System

A real-time biometric driver authentication system using facial recognition, liveness detection, and automated alerting — built with Python, Flask, and PostgreSQL.

---

## Technology Stack

### Backend
| Technology | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **Flask 3.0** | Web framework & REST API |
| **PostgreSQL** | Production database (relational, ACID-compliant) |
| **psycopg2** | PostgreSQL adapter for Python |
| **TensorFlow / Keras** | Deep learning runtime |
| **DeepFace** | FaceNet embedding generation (128-dim vectors) |
| **MTCNN** | Face detection (Multi-task Cascaded Convolutional Networks) |
| **MediaPipe** | Facial landmark detection for liveness (EAR blink detection) |
| **OpenCV** | Image/video capture, preprocessing, and rendering |
| **NumPy / SciPy** | Numerical computation & cosine similarity |
| **Gunicorn** | Production WSGI server |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5** | Page structure & semantic layout |
| **CSS3** (Vanilla) | Dark-theme UI with glassmorphism, gradients, and micro-animations |
| **JavaScript** (Vanilla ES6+) | Real-time dashboard interactions, enrollment workflow, camera capture |
| **Jinja2** | Server-side templating (Flask) |
| **Font Awesome** | Icon library |
| **Chart.js** | Dashboard analytics charts |

### Infrastructure
| Technology | Purpose |
|---|---|
| **PostgreSQL 15+** | Primary data store (drivers, verification logs, audit logs) |
| **YAML** | Application configuration (`config/config.yaml`) |
| **python-dotenv** | Environment variable management for secrets |
| **SMTP (Gmail)** | Email alerting for unauthorized access |

---

## Key Features

- **Real-Time Biometric Verification** — Live camera-based facial recognition using FaceNet (128-dimensional embeddings)
- **Multi-Sample Enrollment** — 5-frame averaged biometric signature for noise reduction
- **Driver Categories (A–E)** — Categorize drivers by vehicle type (motorcycles, cars, trucks, buses, special vehicles)
- **Liveness Detection** — Eye Aspect Ratio (EAR) blink detection to prevent photo-based spoofing
- **Web Dashboard** — Full-featured admin interface for enrollment, monitoring, and log review
- **Automated Email Alerts** — Notifications with captured images for unauthorized access attempts
- **Performance Logging** — Detailed metrics (similarity scores, processing times, authorization rates)
- **Configurable Thresholds** — Tune similarity threshold to balance security vs. usability
- **Audit Trail** — Complete audit logging of all system events
- **Sub-1.5s Latency** — Optimized for real-time performance on standard hardware

---

## System Architecture

```
┌─────────────────┐
│  Camera Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Face Detection  │  (MTCNN)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Liveness Check  │  (EAR Blink Detection via MediaPipe)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FaceNet Embed.  │  (128-dim vector via DeepFace)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cosine Simil.   │  (Match against PostgreSQL stored embeddings)
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

---

## Requirements

### Hardware
- Webcam (USB or built-in)
- Standard laptop/desktop (CPU-based, no GPU required)
- Minimum 4 GB RAM

### Software
- Python 3.9+
- PostgreSQL 13+ (local or remote)
- Windows / Linux / macOS

---

## Quick Start

### 1. Install Dependencies

```bash
cd DraverIdentityVerification_and_Authorization

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Set Up PostgreSQL

Install PostgreSQL, then create the application database:

```sql
-- In psql or pgAdmin:
CREATE DATABASE draver_db;
```

Update `config/config.yaml` with your credentials:

```yaml
database:
  host: "localhost"
  port: 5432
  name: "draver_db"
  user: "postgres"
  password: "your_password"
```

Or set a single environment variable:

```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/draver_db
```

### 3. (Optional) Migrate Existing SQLite Data

If upgrading from the SQLite version:

```bash
python migrate_to_postgres.py
```

### 4. Run the Application

```bash
python run.py
```

Open your browser to: **http://localhost:5000**

### 5. Configure Email Alerts (Optional)

```bash
copy config\.env.example config\.env
# Edit config\.env with your Gmail SMTP credentials
```

---

## Project Structure

```
DraverIdentityVerification_and_Authorization/
├── src/
│   ├── enrollment/              # Driver enrollment module
│   │   ├── camera_capture.py    # Webcam interface & frame capture
│   │   ├── face_processor.py    # Face detection, alignment, embedding generation
│   │   └── enrollment_manager.py # Enrollment workflow orchestration
│   ├── verification/            # Real-time verification engine
│   │   ├── video_stream.py      # Continuous video stream handler
│   │   ├── liveness_detector.py # EAR-based blink/liveness detection
│   │   ├── face_matcher.py      # Cosine similarity matching
│   │   ├── verification_engine.py # Main verification orchestrator
│   │   └── verification_result_handler.py
│   ├── database/                # PostgreSQL database layer
│   │   ├── db_manager.py        # Connection management & table creation
│   │   ├── models.py            # Driver, VerificationLog, AuditLog dataclasses
│   │   ├── driver_repository.py # Driver CRUD operations
│   │   ├── verification_repository.py # Verification log operations
│   │   └── audit_repository.py  # Audit trail operations
│   ├── dashboard/               # Flask web dashboard
│   │   ├── app.py               # Flask app factory
│   │   ├── routes/
│   │   │   ├── api_routes.py    # REST API endpoints
│   │   │   ├── auth_routes.py   # Login / authentication
│   │   │   └── main_routes.py   # Page routes
│   │   └── templates/
│   │       ├── index.html       # Main dashboard (enrollment, monitoring, logs)
│   │       ├── login.html       # Login page
│   │       ├── driver.html      # Driver detail page
│   │       ├── 404.html         # Error pages
│   │       └── 500.html
│   ├── alerting/                # Alerting & logging
│   │   ├── email_service.py     # SMTP email notifications
│   │   └── logger.py            # CSV performance logger
│   └── utils/                   # Shared utilities
│       ├── config.py            # YAML config loader with env var support
│       └── constants.py         # Application constants
├── scripts/                     # CLI tools
│   ├── enroll_driver.py         # Command-line enrollment
│   ├── run_verification.py      # Command-line verification runner
│   └── test_system.py           # System component tests & benchmarks
├── config/
│   └── config.yaml              # Application configuration
├── data/
│   ├── database/                # Legacy SQLite database (if present)
│   ├── logs/                    # Performance logs (CSV)
│   └── alerts/                  # Captured alert images
├── migrate_to_postgres.py       # SQLite → PostgreSQL migration script
├── run.py                       # Application entry point
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Configuration

All settings live in `config/config.yaml`:

### Verification
| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | 0.6 | Minimum cosine similarity for authorization (0.0–1.0) |
| `liveness_ear_threshold` | 0.25 | EAR threshold for blink detection |
| `liveness_blink_frames` | 3 | Consecutive frames for valid blink |
| `max_processing_time_ms` | 1500 | Target processing latency |

### Camera
| Parameter | Default | Description |
|---|---|---|
| `device_id` | 0 | Camera device index |
| `resolution_width` | 640 | Capture width (px) |
| `resolution_height` | 480 | Capture height (px) |
| `fps` | 30 | Target frame rate |

### Database
| Parameter | Type | Description |
|---|---|---|
| `host` | string | PostgreSQL host |
| `port` | int | PostgreSQL port (default: 5432) |
| `name` | string | Database name |
| `user` | string | Database user |
| `password` | string | Database password |
| `url` | string | Full DSN (overrides individual fields) |

Environment variable `DATABASE_URL` overrides all of the above.

---

## Driver Categories

Drivers are assigned a category during enrollment:

| Category | Vehicle Type |
|---|---|
| **A** | Motorcycles & light vehicles |
| **B** | Passenger cars (standard) |
| **C** | Trucks / heavy goods vehicles |
| **D** | Buses / passenger transport |
| **E** | Articulated / special vehicles |

---

## Database Schema (PostgreSQL)

```sql
CREATE TABLE drivers (
    driver_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    license_number VARCHAR(100),
    category VARCHAR(5) DEFAULT 'A',
    biometric_embedding BYTEA NOT NULL,
    enrollment_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    email VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE verification_logs (
    log_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    driver_id INTEGER REFERENCES drivers(driver_id),
    driver_name VARCHAR(255),
    similarity_score DOUBLE PRECISION,
    authorized BOOLEAN DEFAULT FALSE,
    processing_time_ms DOUBLE PRECISION,
    image_path TEXT,
    liveness_passed BOOLEAN DEFAULT FALSE
);

CREATE TABLE audit_logs (
    audit_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action VARCHAR(255),
    user_email VARCHAR(255),
    details TEXT,
    ip_address VARCHAR(50)
);
```

---

## Performance Metrics

Each verification attempt logs:

- **Timestamp** — Date and time
- **Driver ID / Name** — Matched driver (if any)
- **Similarity Score** — Cosine similarity (0.0–1.0)
- **Authorization Status** — Authorized / Unauthorized
- **Liveness Status** — Passed / Failed
- **Processing Time** — End-to-end latency (ms)

---

## Security & Privacy

- All biometric data stored locally (no cloud transmission by default)
- PostgreSQL with role-based access control
- Biometric embeddings are one-way (cannot reconstruct the original face)
- Email credentials stored in gitignored `.env` file
- Configurable image retention policies
- Complete audit trail of all system events

---

## Technical Details

### Face Detection
- **Algorithm**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Output**: Bounding box, confidence score, 5 facial landmarks
- **Minimum Confidence**: 0.95

### Facial Recognition
- **Model**: FaceNet (Inception ResNet v1)
- **Embedding Dimension**: 128
- **Distance Metric**: Cosine Similarity
- **Framework**: DeepFace

### Liveness Detection
- **Method**: Eye Aspect Ratio (EAR) blink detection via MediaPipe
- **Formula**: `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
- **Threshold**: 0.25 (eye considered closed below this value)

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Camera not found | Try different `device_id` values in `config.yaml` |
| No face detected | Improve lighting, center face, move closer |
| Slow processing | Reduce camera resolution, close other apps |
| Email not sending | Check `config/.env` with Gmail app password |
| PostgreSQL connection refused | Verify PostgreSQL is running: `pg_isready` |

---

## Acknowledgments

- **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering"
- **MTCNN**: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- **DeepFace**: Serengil & Ozpinar, "LightFace: A Hybrid Deep Face Recognition Framework"
- **EAR**: Soukupova & Cech, "Real-Time Eye Blink Detection using Facial Landmarks"

---

**Version**: 2.0.0
**Last Updated**: February 2026
**Status**: Academic Prototype — PostgreSQL Edition
