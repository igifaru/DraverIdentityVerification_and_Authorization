"""
Driver API Routes  (port 5001)
Public endpoints for the kiosk — detect, verify, camera, GPS, enrollment.
"""

import base64
import time
from typing import List

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from utils.config import config


driver_api_bp = Blueprint('api', __name__)

VALID_CATEGORY_CODES = {'B', 'C', 'D', 'E', 'F'}


def _get_engine():
    return current_app.config['VERIFICATION_ENGINE']


def _parse_categories(data: dict) -> List[str]:
    raw   = data.get('categories') or [data.get('category', 'A')]
    valid = sorted({c.strip().upper() for c in raw if c.strip().upper() in VALID_CATEGORY_CODES})
    if not valid:
        raise ValueError("At least one valid category (B–F) must be selected.")
    return valid


def _decode_images(images_data: list) -> List[np.ndarray]:
    decoded = []
    for raw in images_data:
        try:
            payload   = raw.split(',', 1)[1] if ',' in raw else raw
            arr       = np.frombuffer(base64.b64decode(payload), np.uint8)
            img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded.append(img)
        except Exception:
            pass
    return decoded


# ── Detection & Verification ──────────────────────────────────────────────────

@driver_api_bp.route('/driver/detect')
def driver_detect():
    engine = _get_engine()
    engine.record_camera_activity(label='driver_detect')
    if not engine.video_stream.is_running:
        return jsonify({'face_present': False, 'confidence': 0.0, 'camera_off': True})
    frame = engine.get_latest_raw_frame()
    if frame is None:
        return jsonify({'face_present': False, 'confidence': 0.0})
    detection = engine.face_processor.detect_face(frame, min_confidence=0.75)
    if detection is None:
        return jsonify({'face_present': False, 'confidence': 0.0})
    return jsonify({'face_present': True, 'confidence': round(float(detection['confidence']), 3)})


@driver_api_bp.route('/driver/verify', methods=['POST'])
def driver_verify():
    engine = _get_engine()
    engine.record_camera_activity(label='driver_verify')
    if not engine.video_stream.is_running:
        engine.start_camera()
        time.sleep(0.8)
    frame = engine.get_latest_raw_frame()
    if frame is None:
        return jsonify({'state': 'no_face', 'driver_name': None, 'similarity': 0.0, 'event_id': None})
    success, result = engine.verify_frame(frame, check_liveness=False)
    if not success:
        return jsonify({'state': 'no_face', 'driver_name': None, 'similarity': 0.0, 'event_id': None})
    if result['authorized']:
        engine.unauthorized_retry_count = 0
    else:
        engine.unauthorized_retry_count += 1
        result['retry_count'] = engine.unauthorized_retry_count
        log_id, is_new_incident = engine.log_verification(result)
        if is_new_incident:
            engine._trigger_alert(result)
    state = 'authorized' if result['authorized'] else 'unauthorized'
    return jsonify({
        'state':          state,
        'driver_name':    result.get('driver_name'),
        'similarity':     round(float(result.get('similarity_score', 0.0)), 4),
        'event_id':       result.get('log_id'),
        'status_message': result.get('status_message', ''),
    })


# ── Camera Control ────────────────────────────────────────────────────────────

@driver_api_bp.route('/driver/camera/start', methods=['POST'])
def start_camera_public():
    engine = _get_engine()
    engine.record_camera_activity(label='public_start_request')
    if not engine.video_stream.is_running:
        success = engine.start_camera()
        return jsonify({'success': success})
    return jsonify({'success': True})


@driver_api_bp.route('/driver/camera/stop', methods=['POST'])
def stop_camera_public():
    engine = _get_engine()
    if engine.video_stream.is_running:
        engine.video_stream.stop()
    return jsonify({'success': True})


# ── Status ────────────────────────────────────────────────────────────────────

@driver_api_bp.route('/status')
def get_status():
    engine = _get_engine()
    status = 'stopped'
    if engine and engine.is_running:
        status = 'active' if engine.video_stream.is_running else 'standby'
    return jsonify({
        'system_status': status,
        'timestamp':     time.time(),
        'system_info': {
            'id':            config.system_id,
            'vehicle_plate': config.vehicle_plate,
            'owner_name':    config.owner_name,
            'location':      config.get('system.location', 'Primary Entrance'),
        },
        'config': {
            'brightness_no_signal': config.brightness_no_signal,
            'brightness_low_light': config.brightness_low_light,
        },
    })


# ── GPS Telemetry ─────────────────────────────────────────────────────────────

@driver_api_bp.route('/location/update', methods=['POST'])
def location_update():
    data       = request.json or {}
    gps_state  = data.get('state', 'unknown')
    lat        = data.get('lat') or 0.0
    lon        = data.get('lon') or 0.0
    distance_m = data.get('distance_m', 0.0)
    print(f"[location] {gps_state} | lat={lat}, lon={lon} | dist={distance_m}m")
    engine = _get_engine()
    if engine:
        engine.db.log_audit(
            action=f'GPS_{gps_state.upper()}',
            user_email='SYSTEM',
            details=f'lat={lat}, lon={lon}, distance_m={distance_m}',
            ip_address=request.remote_addr,
        )
    return jsonify({'received': True, 'state': gps_state})


# ── Enrollment ────────────────────────────────────────────────────────────────

@driver_api_bp.route('/enroll/live', methods=['POST'])
def enroll_capture_live():
    engine = _get_engine()
    engine.record_camera_activity(label='enroll_capture')
    if not engine.video_stream.is_running:
        engine.start_camera()
        time.sleep(3.0)
    for _ in range(12):
        frame = engine.get_latest_raw_frame()
        if frame is None:
            with engine._frame_lock:
                frame = engine.latest_frame.copy() if engine.latest_frame is not None else None
        if frame is None:
            time.sleep(0.8)
            continue
        preprocessed, status = engine.face_processor.process_for_enrollment(frame)
        if preprocessed is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            return jsonify({'success': True, 'image': base64.b64encode(buffer).decode('utf-8')})
        time.sleep(0.8)
    return jsonify({'error': 'No face detected after 12 attempts'})


@driver_api_bp.route('/enroll/save', methods=['POST'])
def enroll_save_driver():
    data           = request.json or {}
    name           = data.get('name', '').strip()
    license_number = data.get('driver_id', '').strip()
    dob            = data.get('dob', '').strip()
    gender         = data.get('gender', '').strip()
    expiry_date    = data.get('expiry_date', '').strip()
    issue_place    = data.get('issue_place', '').strip()

    if not name or not license_number:
        return jsonify({'error': 'Missing name/ID'}), 400
    if not license_number.isdigit():
        return jsonify({'error': 'License number must contain only digits'}), 400

    engine = _get_engine()
    if engine.db.get_driver_by_name(name):
        return jsonify({'error': 'Name already exists'}), 409

    try:
        categories = _parse_categories(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    images_raw = data.get('images') or ([data['image']] if data.get('image') else [])
    images     = _decode_images(images_raw)
    if not images:
        return jsonify({'error': 'No valid images'}), 400

    success, message = engine.enroll_new_driver(
        name, license_number, images,
        category=','.join(categories),
        dob=dob or None, gender=gender or None,
        expiry_date=expiry_date or None, issue_place=issue_place or None,
    )
    if success:
        engine.db.log_audit('ENROLL_DRIVER', 'SYSTEM',
                            f'Enrolled: {name}', request.remote_addr)
        return jsonify({'success': True, 'message': message})
    return jsonify({'error': message}), 500