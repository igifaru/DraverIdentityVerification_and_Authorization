"""
System Constants
Centralized location for all system-wide constants and magic numbers.
"""
from pathlib import Path

# --- File Paths ---
base_dir = Path(__file__).resolve().parent.parent.parent
DB_PATH = base_dir / "data" / "database" / "drivers.db"
LOGS_PATH = base_dir / "data" / "logs" / "verification_log.csv"
CAPTURED_IMAGES_DIR = base_dir / "data" / "alerts" / "captured_images"

# --- System Configuration ---
SYSTEM_ID_DEFAULT = "UNKNOWN-ID"
VEHICLE_PLATE_DEFAULT = "UNKNOWN-PLATE"
OWNER_NAME_DEFAULT = "UNKNOWN-OWNER"

# --- Face Detection ---
FACE_DETECTION_MODEL = 'mtcnn'
FACE_RECOGNITION_MODEL = 'Facenet'
FACE_DETECTION_CONFIDENCE = 0.95
FACE_MARGIN = 0.2  # 20% margin around face
FACE_TARGET_SIZE = (160, 160)

# --- Verification ---
# Similarity threshold for face matching (lower is stricter for cosine distance)
VERIFICATION_THRESHOLD_DEFAULT = 0.4
# Liveness detection (Eye Aspect Ratio)
EAR_THRESHOLD_DEFAULT = 0.25
CONSECUTIVE_FRAMES_EYE_CLOSED = 2

# --- Video Stream ---
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
VIDEO_FPS = 30

# --- UI & Display ---
COLOR_AUTHORIZED = (0, 255, 0)      # Green
COLOR_UNAUTHORIZED = (0, 0, 255)    # Red
COLOR_WARNING = (0, 165, 255)       # Orange
COLOR_INFO = (255, 255, 0)          # Yellow
FONT_SCALE_HEADER = 1.2
FONT_SCALE_TEXT = 0.6
FONT_THICKNESS = 2

# --- Email Alerts ---
ALERT_SUBJECT_TEMPLATE = "SECURITY ALERT: Unauthorized Driver Detected - {plate}"
