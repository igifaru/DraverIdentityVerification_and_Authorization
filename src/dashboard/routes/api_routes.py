"""
API Routes
JSON endpoints consumed by the dashboard front-end.

All endpoints are mounted under the /api prefix (registered in app.py).
Authentication is handled at the main_routes level via @login_required;
API endpoints are expected to be called only from authenticated sessions.
"""

import base64
import threading
import time
from typing import List

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, session

from utils.config import config


api_bp = Blueprint('api', __name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORY_CODES = {'A', 'B', 'C', 'D', 'E'}


def _get_engine():
    """Convenience accessor for the shared VerificationEngine."""
    return current_app.config['VERIFICATION_ENGINE']


def _current_user() -> str:
    """Return the logged-in user's email, or 'UNKNOWN'."""
    return session.get('user_email', 'UNKNOWN')


def _parse_categories(data: dict) -> List[str]:
    """
    Extract and validate the driver category selection from request data.

    Accepts either:
      - ``categories``: list of codes  e.g. ["A", "B", "C"]   (multi-select)
      - ``category``:   single code    e.g. "B"                (legacy / compat)

    Returns a sorted, deduplicated list of valid codes.
    Raises ValueError if the result is empty.
    """
    raw = data.get('categories') or [data.get('category', 'A')]
    valid = sorted({c.strip().upper() for c in raw if c.strip().upper() in VALID_CATEGORY_CODES})
    if not valid:
        raise ValueError("At least one valid category (A–E) must be selected.")
    return valid


def _decode_images(images_data: list) -> List[np.ndarray]:
    """
    Base64-decode a list of JPEG data-URI or plain base64 strings into
    OpenCV BGR arrays. Skips frames that fail to decode rather than aborting.

    Returns a list of decoded images (may be shorter than input on bad frames).
    """
    decoded = []
    for idx, raw in enumerate(images_data):
        try:
            # Strip "data:image/jpeg;base64," prefix if present
            payload = raw.split(',', 1)[1] if ',' in raw else raw
            img_bytes = base64.b64decode(payload)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded.append(img)
                print(f"  [enroll] Sample {idx + 1} decoded – shape={img.shape}")
            else:
                print(f"  [enroll] Sample {idx + 1} produced None after imdecode – skipped")
        except Exception as exc:
            print(f"  [enroll] Sample {idx + 1} decode error: {exc} – skipped")
    return decoded


# ---------------------------------------------------------------------------
# Status & Statistics
# ---------------------------------------------------------------------------

@api_bp.route('/status')
def get_status():
    """Return current system status, today's stats, and the latest result."""
    engine = _get_engine()
    stats = engine.db.get_daily_statistics()

    payload = {
        'system_status': 'active' if engine.is_running else 'stopped',
        'timestamp': time.time(),
        'stats': stats,
        'system_info': {
            'id':           config.system_id,
            'vehicle_plate': config.vehicle_plate,
            'owner_name':   config.owner_name,
        },
    }

    if engine.latest_result:
        r = engine.latest_result
        payload['latest_verification'] = {
            'authorized':      r['authorized'],
            'driver_name':     r['driver_name'],
            'similarity':      float(r['similarity_score']),
            'message':         r.get('status_message', ''),
            'liveness_passed': r['liveness_passed'],
        }

    # Include recent log rows so the dashboard table needs no extra round-trip
    recent = engine.db.get_recent_logs(limit=12)
    payload['recent_logs'] = [log.to_dict() for log in recent]

    return jsonify(payload)


@api_bp.route('/statistics')
def get_statistics():
    """Return aggregate verification statistics."""
    return jsonify(_get_engine().db.get_statistics())


@api_bp.route('/audit')
def get_audit_logs():
    """Return recent system audit log entries."""
    limit = request.args.get('limit', default=50, type=int)
    logs = _get_engine().db.get_audit_logs(limit)
    return jsonify([log.to_dict() for log in logs])


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

@api_bp.route('/enroll/live', methods=['POST'])
def enroll_capture_live():
    """
    Capture a single frame from the live camera stream for enrollment preview.

    The front-end polls this endpoint repeatedly while the user positions their
    face.  We return the *original* (undistorted) frame so the UI can show it;
    face detection happens later during /enroll/save.
    """
    engine = _get_engine()

    # Start camera on-demand if it is not already running
    if not engine.video_stream.is_running:
        print("[enroll/live] Starting camera on demand …")
        engine.start_camera()
        time.sleep(1.0)  # Allow camera sensor to warm up

    with engine._frame_lock:
        frame = engine.latest_frame.copy() if engine.latest_frame is not None \
                else engine.video_stream.read_frame()

    if frame is None:
        return jsonify({'success': False, 'message': 'No video stream available'})

    # Validate that a face is detectable before sending the frame
    preprocessed, status = engine.face_processor.process_for_enrollment(frame)
    if preprocessed is None:
        return jsonify({'error': status})

    # Return the original frame (not the crop) so the UI renders the full view
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'success': True, 'image': image_b64})


@api_bp.route('/enroll/save', methods=['POST'])
def enroll_save_driver():
    """
    Process multi-sample images and enroll a new driver.

    Expected JSON body:
        name        (str)        – driver's legal full name
        driver_id   (str)        – licence number
        categories  (list[str])  – one or more of A, B, C, D, E
        images      (list[str])  – base64-encoded JPEG images (≥1 required)

    Legacy single-image and single-category fields are also accepted for
    backward compatibility.
    """
    data = request.json or {}

    # -- Validate required text fields -----------------------------------
    name           = data.get('name', '').strip()
    license_number = data.get('driver_id', '').strip()
    if not name or not license_number:
        return jsonify({'error': 'Missing required fields: name and driver_id'}), 400

    # -- Parse categories ------------------------------------------------
    try:
        categories = _parse_categories(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    category = ','.join(categories)  # stored as "A,B,C"

    # -- Decode images ---------------------------------------------------
    images_raw = data.get('images') or ([data['image']] if data.get('image') else [])
    if not images_raw:
        return jsonify({'error': 'At least one image is required'}), 400

    images = _decode_images(images_raw)
    if not images:
        return jsonify({'error': 'Failed to decode any of the provided images'}), 400

    # -- Enroll ----------------------------------------------------------
    engine = _get_engine()
    success, message = engine.enroll_new_driver(name, license_number, images, category=category)

    if success:
        engine.db.log_audit(
            "ENROLL_DRIVER",
            _current_user(),
            f"Enrolled driver: {name} (licence: {license_number}, categories: {category})",
            request.remote_addr,
        )
        return jsonify({'success': True, 'message': message})

    return jsonify({'error': message}), 500


# ---------------------------------------------------------------------------
# Verification Engine Control
# ---------------------------------------------------------------------------

@api_bp.route('/system/start', methods=['POST'])
def start_system():
    """Start the verification engine in a background thread."""
    engine = _get_engine()
    if engine.is_running:
        return jsonify({'success': False, 'message': 'System is already running'})

    thread = threading.Thread(
        target=engine.run_continuous_verification,
        kwargs={'show_preview': False},
        daemon=True,
    )
    thread.start()

    engine.db.log_audit(
        "START_ENGINE", _current_user(),
        "Verification engine started via dashboard",
        request.remote_addr,
    )
    return jsonify({'success': True, 'message': 'System started'})


@api_bp.route('/system/stop', methods=['POST'])
def stop_system():
    """Stop the verification engine."""
    engine = _get_engine()
    if not engine.is_running:
        return jsonify({'success': False, 'message': 'System is already stopped'})

    engine.stop()

    engine.db.log_audit(
        "STOP_ENGINE", _current_user(),
        "Verification engine stopped via dashboard",
        request.remote_addr,
    )
    return jsonify({'success': True, 'message': 'System stopping …'})


# ---------------------------------------------------------------------------
# Camera Control
# ---------------------------------------------------------------------------

@api_bp.route('/camera/stop', methods=['POST'])
def stop_camera():
    """
    Release the camera after the enrollment workflow completes.

    The camera is only stopped if the verification engine is not currently
    running (it would re-acquire the camera itself when started).
    """
    engine = _get_engine()
    if not engine.is_running and engine.video_stream.is_running:
        print("[camera/stop] Releasing camera after enrollment …")
        engine.video_stream.stop()
        return jsonify({'success': True, 'message': 'Camera released'})

    return jsonify({'success': False, 'message': 'Camera is in use or already stopped'})
