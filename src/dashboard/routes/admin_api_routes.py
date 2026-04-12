"""
Admin API Routes  (port 5000)
All login-protected JSON endpoints. No camera — DB only.
"""

import os
import time
from typing import List

import requests as http_requests
from flask import Blueprint, current_app, jsonify, request, send_file, session
from functools import wraps

from database.db_manager import DatabaseManager
from utils.config import config


admin_api_bp = Blueprint('api', __name__)

VALID_CATEGORY_CODES = {'B', 'C', 'D', 'E', 'F'}


def _db() -> DatabaseManager:
    return DatabaseManager()


def _driver_url() -> str:
    return current_app.config.get('DRIVER_APP_URL', 'http://localhost:5001')


def _current_user() -> str:
    return session.get('user_email', 'UNKNOWN')


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


def _parse_categories(data: dict) -> List[str]:
    raw = data.get('categories') or [data.get('category', 'A')]
    valid = sorted({c.strip().upper() for c in raw if c.strip().upper() in VALID_CATEGORY_CODES})
    if not valid:
        raise ValueError("At least one valid category (B–F) must be selected.")
    return valid


# ── Status ───────────────────────────────────────────────────────────────────

@admin_api_bp.route('/status')
def get_status():
    try:
        resp = http_requests.get(f"{_driver_url()}/api/status", timeout=2)
        data = resp.json()
    except Exception:
        data = {'system_status': 'offline', 'timestamp': time.time()}
    data['stats'] = _db().get_statistics()
    return jsonify(data)


# ── Alerts ───────────────────────────────────────────────────────────────────

@admin_api_bp.route('/alerts', methods=['GET', 'DELETE'])
@login_required
def get_incidents():
    db = _db()
    if request.method == 'DELETE':
        count = db.clear_all_verification_logs()
        db.log_audit('CLEAR_ALERTS_LOGS', _current_user(),
                     f'Cleared {count} alerts', request.remote_addr)
        return jsonify({'success': True, 'deleted': count})
    limit = request.args.get('limit', 100, type=int)
    logs  = db.get_recent_logs(limit=limit)
    return jsonify([log.to_dict() for log in logs])


@admin_api_bp.route('/alerts/image/<int:log_id>')
@login_required
def get_alert_image(log_id):
    db = _db()
    with db._db() as cur:
        cur.execute('SELECT image_path FROM verification_logs WHERE log_id = %s', (log_id,))
        row = cur.fetchone()
    if not row or not row['image_path']:
        return jsonify({'error': 'Image not found'}), 404
    path = row['image_path']
    if not os.path.exists(path):
        return jsonify({'error': 'File missing on disk'}), 404
    return send_file(path, mimetype='image/jpeg')


# ── Audit Logs ────────────────────────────────────────────────────────────────

@admin_api_bp.route('/audit', methods=['GET', 'DELETE'])
@login_required
def audit_logs():
    db = _db()
    if request.method == 'DELETE':
        count = db.clear_audit_logs()
        db.log_audit('CLEAR_AUDIT_LOGS', _current_user(),
                     f'Cleared {count} entries', request.remote_addr)
        return jsonify({'success': True, 'message': f'Cleared {count} entries'})
    limit  = request.args.get('limit', 100, type=int)
    action = request.args.get('action', default=None, type=str)
    logs   = db.get_audit_logs(limit, action_filter=action)
    return jsonify([log.to_dict() for log in logs])


@admin_api_bp.route('/audit/<int:audit_id>', methods=['DELETE'])
@login_required
def delete_audit_log(audit_id: int):
    db = _db()
    if db.delete_audit_log(audit_id):
        db.log_audit('DELETE_AUDIT_LOG', _current_user(),
                     f'Deleted entry id={audit_id}', request.remote_addr)
        return jsonify({'success': True, 'message': 'Deleted'})
    return jsonify({'error': 'Not found'}), 404


# ── Driver Management ─────────────────────────────────────────────────────────

@admin_api_bp.route('/drivers')
@login_required
def list_drivers():
    return jsonify([d.to_dict() for d in _db().get_all_drivers(active_only=False)])


@admin_api_bp.route('/drivers/<int:driver_id>', methods=['DELETE'])
@login_required
def delete_driver(driver_id: int):
    db = _db()
    driver = db.get_driver(driver_id)
    if not driver:
        return jsonify({'error': 'Not found'}), 404
    if db.delete_driver(driver_id):
        db.log_audit('DELETE_DRIVER', _current_user(),
                     f'Soft-deleted: {driver.name}', request.remote_addr)
        return jsonify({'success': True, 'message': 'Removed'})
    return jsonify({'error': 'Failed'}), 500


@admin_api_bp.route('/drivers/<int:driver_id>', methods=['PUT'])
@login_required
def update_driver(driver_id: int):
    db = _db()
    driver = db.get_driver(driver_id)
    if not driver:
        return jsonify({'error': 'Not found'}), 404

    data        = request.json or {}
    new_name    = data.get('name', '').strip()
    new_license = data.get('license_number', '').strip()
    new_status  = data.get('status', '').strip().lower()
    new_dob     = data.get('dob', '').strip()
    new_gender  = data.get('gender', '').strip()
    new_expiry  = data.get('expiry_date', '').strip()
    new_place   = data.get('issue_place', '').strip()

    if new_license and not new_license.isdigit():
        return jsonify({'error': 'License number must contain only digits'}), 400

    try:
        new_categories = _parse_categories(data) if ('categories' in data or 'category' in data) else None
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    parts, params = [], []
    if new_name:    parts.append('name = %s');           params.append(new_name)
    if new_license: parts.append('license_number = %s'); params.append(new_license)
    if new_categories:
        parts.append('category = %s'); params.append(','.join(new_categories))
    if new_status in ('active', 'inactive'):
        parts.append('status = %s'); params.append(new_status)
    if new_dob:    parts.append('dob = %s');          params.append(new_dob)
    if new_gender: parts.append('gender = %s');       params.append(new_gender)
    if new_expiry: parts.append('expiry_date = %s');  params.append(new_expiry)
    if new_place:  parts.append('issue_place = %s');  params.append(new_place)

    if not parts:
        return jsonify({'error': 'No fields to update'}), 400

    params.append(driver_id)
    sql = f"UPDATE drivers SET {', '.join(parts)} WHERE driver_id = %s"
    with db.driver_repo._db() as cur:
        cur.execute(sql, params)

    db.log_audit('UPDATE_DRIVER', _current_user(),
                 f'Updated driver id={driver_id}', request.remote_addr)
    return jsonify({'success': True, 'driver': db.get_driver(driver_id).to_dict()})


# ── Enrollment ────────────────────────────────────────────────────────────────

@admin_api_bp.route('/enroll/live', methods=['POST'])
@login_required
def enroll_capture_live():
    """Proxy live-capture request to driver app (camera lives there)."""
    try:
        resp = http_requests.post(f"{_driver_url()}/api/enroll/live", timeout=15)
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({'error': f'Driver app unreachable: {exc}'}), 503


@admin_api_bp.route('/enroll/save', methods=['POST'])
@login_required
def enroll_save_driver():
    """Proxy enrollment save to driver app (DeepFace models live there)."""
    data = request.json or {}
    try:
        resp = http_requests.post(
            f"{_driver_url()}/api/enroll/save",
            json=data, timeout=30,
        )
        result = resp.json()
        if result.get('success'):
            _db().log_audit('ENROLL_DRIVER', _current_user(),
                            f"Enrolled: {data.get('name', '')}", request.remote_addr)
        return jsonify(result), resp.status_code
    except Exception as exc:
        return jsonify({'error': f'Driver app unreachable: {exc}'}), 503


@admin_api_bp.route('/driver-photo/<int:driver_id>')
@login_required
def serve_driver_photo(driver_id: int):
    db = _db()
    driver = db.get_driver(driver_id)
    if not driver or not driver.photo_path or not os.path.isfile(driver.photo_path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(driver.photo_path, mimetype='image/jpeg')


# ── Camera Control (proxied) ──────────────────────────────────────────────────

@admin_api_bp.route('/camera/stop', methods=['POST'])
@login_required
def stop_camera():
    try:
        resp = http_requests.post(f"{_driver_url()}/api/driver/camera/stop", timeout=3)
        return jsonify(resp.json()), resp.status_code
    except Exception:
        return jsonify({'success': False, 'error': 'Driver app unreachable'}), 503