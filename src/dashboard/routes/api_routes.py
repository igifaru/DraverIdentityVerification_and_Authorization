"""
API Routes
JSON endpoints consumed by the dashboard front-end.

All endpoints are mounted under the /api prefix (registered in app.py).
Authentication is handled at the main_routes level via @login_required;
API endpoints are expected to be called only from authenticated sessions.
"""

import base64
import os
import time
from typing import List

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file, session, redirect, url_for
from functools import wraps

from utils.config import config


api_bp = Blueprint('api', __name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORY_CODES = {'B', 'C', 'D', 'E', 'F'}   # A (motorcycles) excluded — vehicle-only system


def _get_engine():
    """Convenience accessor for the shared VerificationEngine."""
    return current_app.config['VERIFICATION_ENGINE']


def _current_user() -> str:
    """Return the logged-in user's email, or 'UNKNOWN'."""
    return session.get('user_email', 'UNKNOWN')


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function


def _parse_categories(data: dict) -> List[str]:
    """
    Extract and validate the driver category selection from request data.
    """
    raw = data.get('categories') or [data.get('category', 'A')]
    valid = sorted({c.strip().upper() for c in raw if c.strip().upper() in VALID_CATEGORY_CODES})
    if not valid:
        raise ValueError("At least one valid category (A–E) must be selected.")
    return valid


def _decode_images(images_data: list) -> List[np.ndarray]:
    """
    Base64-decode images into OpenCV BGR arrays.
    """
    decoded = []
    for idx, raw in enumerate(images_data):
        try:
            payload = raw.split(',', 1)[1] if ',' in raw else raw
            img_bytes = base64.b64decode(payload)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded.append(img)
            else:
                print(f"  [enroll] Sample {idx + 1} produced None skipping")
        except Exception as exc:
            print(f"  [enroll] Sample {idx + 1} decode error: {exc}")
    return decoded


# ---------------------------------------------------------------------------
# Driver terminal endpoints  (public — no login required)
# ---------------------------------------------------------------------------

@api_bp.route('/driver/detect')
def driver_detect():
    """Fast face-presence check for the driver terminal idle state."""
    engine = _get_engine()
    engine.record_camera_activity(label='driver_detect')
    if not engine.video_stream.is_running:
        engine.start_camera()

    frame = engine.video_stream.read_frame()
    if frame is None:
        return jsonify({'face_present': False, 'confidence': 0.0})

    detection = engine.face_processor.detect_face(frame, min_confidence=0.75)
    if detection is None:
        return jsonify({'face_present': False, 'confidence': 0.0})

    return jsonify({
        'face_present': True,
        'confidence':   round(float(detection['confidence']), 3),
    })


@api_bp.route('/driver/verify', methods=['POST'])
def driver_verify():
    """Full verification pipeline for the driver terminal."""
    engine = _get_engine()
    engine.record_camera_activity(label='driver_verify')
    if not engine.video_stream.is_running:
        engine.start_camera()
        time.sleep(0.5)

    frame = engine.video_stream.read_frame()
    if frame is None:
        return jsonify({'state': 'no_face', 'driver_name': None, 'similarity': 0.0, 'event_id': None})

    success, result = engine.verify_frame(frame, check_liveness=False)
    if not success:
        return jsonify({'state': 'no_face', 'driver_name': None, 'similarity': 0.0, 'event_id': None})
    
    # Update retry count
    if result['authorized']:
        engine.unauthorized_retry_count = 0 
    else:
        engine.unauthorized_retry_count += 1
    
    result['retry_count'] = engine.unauthorized_retry_count
    engine.log_verification(result)

    if not result['authorized']:
        engine._trigger_alert(result)

    state = 'authorized' if result['authorized'] else 'unauthorized'
    return jsonify({
        'state':       state,
        'driver_name': result.get('driver_name'),
        'similarity':  round(float(result.get('similarity_score', 0.0)), 4),
        'event_id':    result.get('log_id'),
    })


# ---------------------------------------------------------------------------
# Status & Statistics
# ---------------------------------------------------------------------------

@api_bp.route('/status')
def get_status():
    engine = _get_engine()
    status = 'stopped'
    if engine.is_running:
        status = 'active' if engine.video_stream.is_running else 'standby'
    
    return jsonify({
        'system_status': status,
        'timestamp': time.time(),
        'system_info': {
            'id':            config.system_id,
            'vehicle_plate': config.vehicle_plate,
            'owner_name':    config.owner_name,
            'location':      config.get('system.location', 'Primary Entrance'),
        },
        'config': {
            'brightness_no_signal': config.brightness_no_signal,
            'brightness_low_light': config.brightness_low_light,
        }
    })


# ---------------------------------------------------------------------------
# Security Alerts & Incidents
# ---------------------------------------------------------------------------

@api_bp.route('/alerts', methods=['GET'])
@login_required
def get_alerts():
    """Return unauthorized attempts with rich metadata for incident module."""
    engine = _get_engine()
    limit = request.args.get('limit', 50, type=int)
    logs = engine.db.get_unauthorized_attempts(limit=limit)
    return jsonify([log.to_dict() for log in logs])


@api_bp.route('/alerts/image/<int:log_id>', methods=['GET'])
@login_required
def get_alert_image(log_id):
    """Serve the captured face image for a specific incident."""
    engine = _get_engine()
    with engine.db._db() as cur:
        cur.execute('SELECT image_path FROM verification_logs WHERE log_id = %s', (log_id,))
        row = cur.fetchone()
        
    if not row or not row['image_path']:
        return jsonify({'error': 'Image not found'}), 404
        
    path = row['image_path']
    if not os.path.exists(path):
        return jsonify({'error': 'File missing on disk'}), 404
        
    return send_file(path, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------

@api_bp.route('/audit', methods=['GET', 'DELETE'])
@login_required
def audit_logs():
    engine = _get_engine()
    if request.method == 'DELETE':
        count = engine.db.clear_audit_logs()
        engine.db.log_audit('CLEAR_AUDIT_LOGS', _current_user(), f'Cleared {count} entries', request.remote_addr)
        return jsonify({'success': True, 'message': f'Cleared {count} entries'})

    limit  = request.args.get('limit', 100, type=int)
    action = request.args.get('action', default=None, type=str)
    logs = engine.db.get_audit_logs(limit, action_filter=action)
    return jsonify([log.to_dict() for log in logs])


@api_bp.route('/audit/<int:audit_id>', methods=['DELETE'])
@login_required
def delete_audit_log(audit_id: int):
    engine = _get_engine()
    if engine.db.delete_audit_log(audit_id):
        engine.db.log_audit('DELETE_AUDIT_LOG', _current_user(), f'Deleted entry id={audit_id}', request.remote_addr)
        return jsonify({'success': True, 'message': 'Deleted'})
    return jsonify({'error': 'Not found'}), 404


# ---------------------------------------------------------------------------
# Driver Management
# ---------------------------------------------------------------------------

@api_bp.route('/drivers')
@login_required
def list_drivers():
    engine = _get_engine()
    drivers = engine.db.get_all_drivers(active_only=False)
    return jsonify([d.to_dict() for d in drivers])


@api_bp.route('/drivers/<int:driver_id>', methods=['DELETE'])
@login_required
def delete_driver(driver_id: int):
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver: return jsonify({'error': 'Not found'}), 404

    if engine.db.delete_driver(driver_id):
        engine.db.log_audit('DELETE_DRIVER', _current_user(), f'Soft-deleted: {driver.name}', request.remote_addr)
        return jsonify({'success': True, 'message': 'Removed'})
    return jsonify({'error': 'Failed'}), 500


@api_bp.route('/drivers/<int:driver_id>', methods=['PUT'])
@login_required
def update_driver(driver_id: int):
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver: return jsonify({'error': 'Not found'}), 404

    data = request.json or {}
    new_name = data.get('name', '').strip()
    new_license = data.get('license_number', '').strip()
    new_status  = data.get('status', '').strip().lower()

    try:
        new_categories = _parse_categories(data) if ('categories' in data or 'category' in data) else None
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    parts, params = [], []
    if new_name:
        parts.append('name = %s'); params.append(new_name)
    if new_license:
        parts.append('license_number = %s'); params.append(new_license)
    if new_categories is not None:
        cat_str = ','.join(new_categories)
        parts.append('category = %s'); params.append(cat_str)
    if new_status in ('active', 'inactive'):
        parts.append('status = %s'); params.append(new_status)

    if not parts: return jsonify({'error': 'No fields to update'}), 400

    params.append(driver_id)
    sql = f"UPDATE drivers SET {', '.join(parts)} WHERE driver_id = %s"
    with engine.db.driver_repo._db() as cur:
        cur.execute(sql, params)

    engine.db.log_audit('UPDATE_DRIVER', _current_user(), f'Updated driver id={driver_id}', request.remote_addr)
    return jsonify({'success': True, 'driver': engine.db.get_driver(driver_id).to_dict()})


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

@api_bp.route('/enroll/live', methods=['POST'])
@login_required
def enroll_capture_live():
    engine = _get_engine()
    engine.record_camera_activity(label='enroll_capture')
    if not engine.video_stream.is_running:
        engine.start_camera()
        time.sleep(3.0)

    for attempt in range(8):
        frame = engine.video_stream.read_frame()
        if frame is None:
            with engine._frame_lock:
                frame = engine.latest_frame.copy() if engine.latest_frame is not None else None
        
        if frame is None: continue

        preprocessed, status = engine.face_processor.process_for_enrollment(frame)
        if preprocessed is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            return jsonify({'success': True, 'image': base64.b64encode(buffer).decode('utf-8')})
        time.sleep(0.8)

    return jsonify({'error': 'No face detected'})


@api_bp.route('/enroll/save', methods=['POST'])
@login_required
def enroll_save_driver():
    data = request.json or {}
    name = data.get('name', '').strip()
    license_number = data.get('driver_id', '').strip()
    if not name or not license_number:
        return jsonify({'error': 'Missing name/ID'}), 400

    engine = _get_engine()
    if engine.db.get_driver_by_name(name): return jsonify({'error': 'Name exists'}), 409
    
    try:
        categories = _parse_categories(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    images_raw = data.get('images') or ([data['image']] if data.get('image') else [])
    images = _decode_images(images_raw)
    if not images: return jsonify({'error': 'No valid images'}), 400

    success, message = engine.enroll_new_driver(name, license_number, images, category=','.join(categories))
    if success:
        engine.db.log_audit("ENROLL_DRIVER", _current_user(), f"Enrolled: {name}", request.remote_addr)
        return jsonify({'success': True, 'message': message})
    return jsonify({'error': message}), 500


@api_bp.route('/driver-photo/<int:driver_id>')
@login_required
def serve_driver_photo(driver_id: int):
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver or not driver.photo_path or not os.path.isfile(driver.photo_path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(driver.photo_path, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# Camera Control
# ---------------------------------------------------------------------------

@api_bp.route('/camera/stop', methods=['POST'])
@login_required
def stop_camera():
    engine = _get_engine()
    if engine.video_stream.is_running:
        engine.video_stream.stop()
        return jsonify({'success': True})
    return jsonify({'success': True})
