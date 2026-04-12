"""
Driver Main Routes  (port 5001)
Page, MJPEG stream, and status for the driver kiosk.
"""

import time

import cv2
import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template

from utils.config import config


driver_main_bp = Blueprint('main', __name__)

_FRAME_INTERVAL_S = 1 / 30


def _generate_mjpeg(engine):
    while True:
        if engine and engine.is_running:
            with engine._frame_lock:
                frame = engine.latest_frame
        elif engine and engine.video_stream.is_running:
            frame = engine.video_stream.read_frame()
        else:
            frame = None

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(frame, "Camera Standby", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(_FRAME_INTERVAL_S)


@driver_main_bp.route('/driver')
def driver_screen():
    return render_template('driver.html',
        vehicle_plate=config.vehicle_plate,
        system_id=config.system_id,
    )


@driver_main_bp.route('/video_feed')
def video_feed():
    engine = current_app.config['VERIFICATION_ENGINE']
    return Response(
        _generate_mjpeg(engine),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@driver_main_bp.route('/api/status/driver')
def get_driver_status_api():
    engine = current_app.config['VERIFICATION_ENGINE']
    if engine is None:
        return jsonify({'status': 'offline'})
    return jsonify(engine.get_driver_status())