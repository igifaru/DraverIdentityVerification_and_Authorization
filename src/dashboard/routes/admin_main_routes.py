"""
Admin Main Routes  (port 5000)
"""

import os
import time

import cv2
import numpy as np
import requests as http_requests
from flask import (
    Blueprint, Response, current_app,
    jsonify, render_template, request, send_file, stream_with_context,
)

from dashboard.routes.auth_routes import login_required
from utils.config import config


admin_main_bp = Blueprint('main', __name__)


@admin_main_bp.route('/')
@login_required
def index():
    return render_template('index.html',
        system_id=config.system_id,
        vehicle_plate=config.vehicle_plate,
        owner_name=config.owner_name,
    )


@admin_main_bp.route('/video_feed')
@login_required
def video_feed():
    """Proxy the MJPEG stream from the driver app (port 5001)."""
    driver_url = current_app.config.get('DRIVER_APP_URL', 'http://localhost:5001')

    def _proxy_stream():
        try:
            with http_requests.get(
                f"{driver_url}/video_feed",
                stream=True, timeout=5
            ) as r:
                for chunk in r.iter_content(chunk_size=4096):
                    yield chunk
        except Exception:
            frame = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(frame, "Driver Terminal Offline", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')

    return Response(
        stream_with_context(_proxy_stream()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@admin_main_bp.route('/api/logs/image/<path:image_path>')
@login_required
def get_log_image(image_path):
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return "Image not found", 404


@admin_main_bp.route('/api/status/driver')
def get_driver_status_api():
    driver_url = current_app.config.get('DRIVER_APP_URL', 'http://localhost:5001')
    try:
        resp = http_requests.get(f"{driver_url}/api/status/driver", timeout=2)
        return jsonify(resp.json())
    except Exception:
        return jsonify({'status': 'offline', 'message': 'Driver terminal unreachable'})