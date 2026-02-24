"""
Main Routes
Handles main dashboard pages and video streaming
"""
from flask import Blueprint, render_template, Response, request, jsonify, current_app, send_file
import cv2
import time
from dashboard.routes.auth_routes import login_required

import numpy as np

main_bp = Blueprint('main', __name__)

def generate_video_feed(engine):
    """Generate video feed for streaming"""
    
    # Camera will be started on-demand by enrollment workflow
    # No longer auto-starting here for better UX control

    while True:
        # If engine is running background loop, use its frame
        if engine.is_running:
            with engine._frame_lock:
                frame = engine.latest_frame
        else:
            # Check if camera is running before reading
            if engine.video_stream.is_running:
                frame = engine.video_stream.read_frame()
            else:
                frame = None

        if frame is None:
            # Create placeholder frame if camera is off
            frame = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(frame, "Camera Standby", (220, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # Limit to ~30 FPS

@main_bp.route('/')
@login_required
def index():
    """Serve dashboard HTML"""
    from utils.config import config
    
    engine = current_app.config['VERIFICATION_ENGINE']
    stats = engine.db.get_daily_statistics()
    
    return render_template('index.html', 
                          stats=stats,
                          system_id=config.system_id,
                          vehicle_plate=config.vehicle_plate,
                          owner_name=config.owner_name)

@main_bp.route('/driver')
def driver_screen():
    """Serve driver feedback screen HTML"""
    from utils.config import config
    return render_template('driver.html', vehicle_plate=config.vehicle_plate)

@main_bp.route('/video_feed')
def video_feed():
    """Video streaming route"""
    engine = current_app.config['VERIFICATION_ENGINE']
    return Response(generate_video_feed(engine),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/api/status/driver')
def get_driver_status_api():
    """Get simplified status for driver feedback screen"""
    engine = current_app.config['VERIFICATION_ENGINE']
    status = engine.get_driver_status()
    return jsonify(status)

@main_bp.route('/api/logs/image/<path:image_path>')
@login_required
def get_log_image(image_path):
    """Serve captured verification images"""
    import os
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return "Image not found", 404

# Enrollment Routes handled in api_routes.py now
