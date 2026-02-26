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
from flask import Blueprint, current_app, jsonify, request, send_file, session

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
    """Return current system status for the dashboard engine chip."""
    engine = _get_engine()
    return jsonify({
        'system_status': 'active' if engine.is_running else 'stopped',
        'timestamp': time.time(),
        'system_info': {
            'id':            config.system_id,
            'vehicle_plate': config.vehicle_plate,
            'owner_name':    config.owner_name,
        },
    })


@api_bp.route('/alerts')
def get_alerts():
    """
    Return unauthorized verification events that occurred after `since` (Unix timestamp).
    The dashboard polls this endpoint to drive toast notifications.

    Query params:
        since (float, default 0) — return only events newer than this Unix timestamp.
        limit (int,  default 20) — cap the number of returned events.
    """
    since = request.args.get('since', default=0.0, type=float)
    limit = request.args.get('limit',  default=20,  type=int)

    engine  = _get_engine()
    recent  = engine.db.get_recent_logs(limit=100)   # fetch enough to filter

    alerts = []
    for log in recent:
        if log.authorized:
            continue                                  # only unauthorized
        if log.timestamp is None:
            continue
        # Convert aware/naive datetime to Unix float for comparison
        import calendar
        ts = calendar.timegm(log.timestamp.timetuple()) + log.timestamp.microsecond / 1e6
        if ts > since:
            alerts.append({
                'log_id':        log.log_id,
                'timestamp':     log.timestamp.isoformat(),
                'unix_ts':       ts,
                'driver_name':   log.driver_name or 'Unknown',
                'similarity':    round(float(log.similarity_score or 0), 4),
                'liveness':      bool(log.liveness_passed),
                'image_url':     f'/api/alert-image/{log.log_id}' if log.image_path else None,
                'vehicle_plate': config.vehicle_plate,
                'owner_name':    config.owner_name,
            })

    # Newest first; respect limit
    alerts.sort(key=lambda a: a['unix_ts'], reverse=True)
    return jsonify(alerts[:limit])


@api_bp.route('/alert-image/<int:log_id>')
def serve_alert_image(log_id: int):
    """
    Stream the saved JPEG for an unauthorized verification event.
    The image is stored at the path recorded in verification_logs.image_path.
    Returns 404 if no image was saved for this log entry.
    """
    engine = _get_engine()
    logs   = engine.db.get_recent_logs(limit=200)
    log    = next((l for l in logs if l.log_id == log_id), None)

    if not log or not log.image_path:
        return jsonify({'error': 'Image not found'}), 404

    path = log.image_path
    if not os.path.isfile(path):
        return jsonify({'error': 'Image file missing on disk'}), 404

    return send_file(path, mimetype='image/jpeg')


@api_bp.route('/audit', methods=['GET', 'DELETE'])
def audit_logs():
    """GET: return recent audit log entries. DELETE: clear all entries."""
    engine = _get_engine()

    if request.method == 'DELETE':
        count = engine.db.clear_audit_logs()
        engine.db.log_audit(
            'CLEAR_AUDIT_LOGS', _current_user(),
            f'Cleared {count} audit log entries',
            request.remote_addr,
        )
        return jsonify({'success': True, 'message': f'Cleared {count} log entries'})

    # GET
    limit  = request.args.get('limit',  default=100, type=int)
    action = request.args.get('action', default=None, type=str)
    logs = engine.db.get_audit_logs(limit, action_filter=action or None)
    return jsonify([log.to_dict() for log in logs])


@api_bp.route('/audit/<int:audit_id>', methods=['DELETE'])
def delete_audit_log(audit_id: int):
    """Permanently delete a single audit log entry."""
    engine = _get_engine()
    deleted = engine.db.delete_audit_log(audit_id)
    if deleted:
        engine.db.log_audit(
            'DELETE_AUDIT_LOG', _current_user(),
            f'Deleted audit log entry id={audit_id}',
            request.remote_addr,
        )
        return jsonify({'success': True, 'message': f'Log entry {audit_id} deleted'})
    return jsonify({'error': 'Audit log entry not found'}), 404


# ---------------------------------------------------------------------------
# Driver Management
# ---------------------------------------------------------------------------

@api_bp.route('/drivers')
def list_drivers():
    """Return all enrolled drivers (active and inactive)."""
    engine = _get_engine()
    drivers = engine.db.get_all_drivers(active_only=False)
    return jsonify([d.to_dict() for d in drivers])


@api_bp.route('/drivers/<int:driver_id>', methods=['DELETE'])
def delete_driver(driver_id: int):
    """Soft-delete a driver (sets status = 'inactive')."""
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver:
        return jsonify({'error': 'Driver not found'}), 404

    success = engine.db.delete_driver(driver_id)
    if success:
        engine.db.log_audit(
            'DELETE_DRIVER', _current_user(),
            f'Soft-deleted driver: {driver.name} (id={driver_id})',
            request.remote_addr,
        )
        return jsonify({'success': True, 'message': f'{driver.name} removed'})
    return jsonify({'error': 'Delete failed'}), 500


@api_bp.route('/drivers/<int:driver_id>', methods=['PUT'])
def update_driver(driver_id: int):
    """
    Update a driver's editable fields: name, license_number, category, status.
    Only fields present in the request body are changed.
    """
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver:
        return jsonify({'error': 'Driver not found'}), 404

    data = request.json or {}
    changes = {}

    new_name    = data.get('name', '').strip()
    new_license = data.get('license_number', '').strip()
    new_status  = data.get('status', '').strip().lower()

    # Categories: accept list or single string
    try:
        new_categories = _parse_categories(data) if ('categories' in data or 'category' in data) else None
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Build dynamic UPDATE
    parts, params = [], []
    if new_name:
        parts.append('name = %s');          params.append(new_name);          changes['name'] = new_name
    if new_license:
        parts.append('license_number = %s'); params.append(new_license);       changes['license_number'] = new_license
    if new_categories is not None:
        cat_str = ','.join(new_categories)
        parts.append('category = %s');       params.append(cat_str);            changes['category'] = cat_str
    if new_status in ('active', 'inactive'):
        parts.append('status = %s');         params.append(new_status);         changes['status'] = new_status

    if not parts:
        return jsonify({'error': 'No valid fields to update'}), 400

    params.append(driver_id)
    sql = f"UPDATE drivers SET {', '.join(parts)} WHERE driver_id = %s"

    with engine.db.driver_repo._db() as cur:
        cur.execute(sql, params)

    engine.db.log_audit(
        'UPDATE_DRIVER', _current_user(),
        f'Updated driver id={driver_id}: {changes}',
        request.remote_addr,
    )
    updated = engine.db.get_driver(driver_id)
    return jsonify({'success': True, 'driver': updated.to_dict()})



# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

@api_bp.route('/enroll/live', methods=['POST'])
def enroll_capture_live():
    """
    Capture a single frame from the live camera stream for enrollment preview.
    Retries up to 4 times (600 ms apart) if no face is detected, to handle
    camera warm-up lag and momentary detection misses.
    """
    engine = _get_engine()

    # Start camera on-demand if it is not already running
    if not engine.video_stream.is_running:
        print("[enroll/live] Starting camera on demand …")
        engine.start_camera()
        time.sleep(2.5)  # Allow camera sensor to fully warm up

    MAX_TRIES = 5
    last_error = 'No face detected'

    for attempt in range(MAX_TRIES):
        # Always read a FRESH raw frame directly from the video stream.
        # engine.latest_frame may have verification overlays (banners, text)
        # baked into it by the result handler, which would confuse face detection
        # and show "No face detected" text on the enrollment preview.
        frame = engine.video_stream.read_frame()
        if frame is None:
            with engine._frame_lock:
                frame = engine.latest_frame.copy() if engine.latest_frame is not None else None

        if frame is None:
            return jsonify({'success': False, 'error': 'No video stream available'})

        # Validate that a face is detectable
        preprocessed, status = engine.face_processor.process_for_enrollment(frame)
        if preprocessed is not None:
            # Face found — return the original frame for preview
            _, buffer = cv2.imencode('.jpg', frame)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'image': image_b64})

        last_error = status
        print(f"[enroll/live] Attempt {attempt + 1}/{MAX_TRIES}: {status} — retrying…")
        time.sleep(0.6)  # Wait 600 ms before next frame

    return jsonify({'error': last_error})


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

    # -- Duplicate check ---------------------------------------------------
    engine = _get_engine()
    if engine.db.get_driver_by_name(name):
        return jsonify({'error': f'A driver named "{name}" is already enrolled.'}), 409
    if engine.db.get_driver_by_license(license_number):
        return jsonify({'error': f'Licence number "{license_number}" is already enrolled.'}), 409

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


@api_bp.route('/driver-photo/<int:driver_id>')
def serve_driver_photo(driver_id: int):
    """
    Serve the enrollment portrait photo for a driver.
    Returns 404 if no photo was saved during enrollment.
    """
    engine = _get_engine()
    driver = engine.db.get_driver(driver_id)
    if not driver or not driver.photo_path:
        return jsonify({'error': 'Photo not found'}), 404

    if not os.path.isfile(driver.photo_path):
        return jsonify({'error': 'Photo file missing on disk'}), 404

    return send_file(driver.photo_path, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# Camera Control
# ---------------------------------------------------------------------------

@api_bp.route('/camera/stop', methods=['POST'])
def stop_camera():
    """
    Release the camera.

    Called after enrollment capture completes so the camera is turned OFF.
    The verification engine continues to run in standby and will process
    frames again if/when the camera is restarted.
    """
    engine = _get_engine()

    if engine.video_stream.is_running:
        engine.video_stream.stop()
        print("[camera/stop] Camera released")
        return jsonify({'success': True, 'message': 'Camera released'})

    return jsonify({'success': True, 'message': 'Camera already stopped'})
