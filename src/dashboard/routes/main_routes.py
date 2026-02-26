"""
Main Routes
Page routes and the MJPEG video streaming endpoint.

Routes:
  GET  /              – Dashboard (requires login)
  GET  /driver        – Driver feedback screen (public)
  GET  /video_feed    – MJPEG stream
  GET  /api/status/driver       – Simplified status for driver screen
  GET  /api/logs/image/<path>   – Serve captured alert images
"""

import os
import time

import cv2
import numpy as np
from flask import (
    Blueprint, Response, current_app,
    jsonify, render_template, request, send_file,
)

from dashboard.routes.auth_routes import login_required
from utils.config import config


main_bp = Blueprint('main', __name__)

# How long to sleep between yielded frames (~30 fps)
_FRAME_INTERVAL_S = 1 / 30


# ---------------------------------------------------------------------------
# Video streaming helper
# ---------------------------------------------------------------------------

def _generate_mjpeg(engine):
    """
    Generator that yields MJPEG boundary frames indefinitely.

    Frame source priority:
      1. engine.latest_frame  – when the verification loop is running
      2. video_stream.read_frame() – when the camera is on but engine is idle
      3. A grey placeholder with "Camera Standby" text

    The camera is never started here; that is controlled by the enrollment
    workflow and the Start System button.
    """
    while True:
        if engine.is_running:
            with engine._frame_lock:
                frame = engine.latest_frame
        elif engine.video_stream.is_running:
            frame = engine.video_stream.read_frame()
        else:
            frame = None

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(
                frame, "Camera Standby", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2,
            )

        _, buffer = cv2.imencode('.jpg', frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
        time.sleep(_FRAME_INTERVAL_S)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@main_bp.route('/')
@login_required
def index():
    """Render the main admin dashboard."""
    engine = current_app.config['VERIFICATION_ENGINE']
    return render_template(
        'index.html',
        system_id=config.system_id,
        vehicle_plate=config.vehicle_plate,
        owner_name=config.owner_name,
    )


@main_bp.route('/driver')
def driver_screen():
    """Render the public driver feedback screen."""
    return render_template('driver.html', vehicle_plate=config.vehicle_plate)


# ---------------------------------------------------------------------------
# Streaming & media
# ---------------------------------------------------------------------------

@main_bp.route('/video_feed')
def video_feed():
    """MJPEG streaming endpoint consumed by the dashboard <img> tag."""
    engine = current_app.config['VERIFICATION_ENGINE']
    return Response(
        _generate_mjpeg(engine),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@main_bp.route('/api/logs/image/<path:image_path>')
@login_required
def get_log_image(image_path):
    """Serve a captured alert image by its filesystem path."""
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return "Image not found", 404


# ---------------------------------------------------------------------------
# Driver screen API
# ---------------------------------------------------------------------------

@main_bp.route('/api/status/driver')
def get_driver_status_api():
    """Return simplified verification status for the driver feedback screen."""
    engine = current_app.config['VERIFICATION_ENGINE']
    return jsonify(engine.get_driver_status())
