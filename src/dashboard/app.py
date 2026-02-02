"""
Flask Dashboard Application
Web-based interface for monitoring driver verification system
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_cors import CORS
import cv2
import json
import secrets
import base64
import io
from functools import wraps
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from verification.verification_engine import VerificationEngine
from database.db_manager import DatabaseManager
from utils.config import config
from alerting.email_service import EmailService


app = Flask(__name__)
# Set secret key for session management
app.secret_key = secrets.token_hex(16)
CORS(app)

import threading
import time

# Global verification engine
engine = None
engine_lock = threading.Lock()
db = DatabaseManager()

# Admin credentials (hardcoded for prototype as per request)
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "admin123"


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_engine():
    """Get or create verification engine instance"""
    global engine
    with engine_lock:
        if engine is None:
            engine = VerificationEngine()
            # Start verification loop in background thread
            t = threading.Thread(
                target=engine.run_continuous_verification,
                kwargs={'show_preview': False, 'enable_liveness': True}
            )
            t.daemon = True
            t.start()
            print("✓ Background verification thread started")
                
    return engine


def generate_video_feed():
    """Generate video feed for streaming"""
    engine = get_engine()
    
    while True:
        # Get latest processed frame
        with engine._frame_lock:
            if engine.latest_frame is None:
                frame = None
            else:
                frame = engine.latest_frame.copy()
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit streaming FPS to reducing load
        time.sleep(0.03)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['user'] = email
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Serve dashboard HTML"""
    return render_template('index.html')


@app.route('/driver')
def driver_screen():
    """Serve driver feedback screen HTML"""
    return render_template('driver.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route (public for driver/admin use)"""
    return Response(
        generate_video_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/status')
@login_required
def get_status():
    """Get current verification status"""
    engine = get_engine()
    
    if engine is None:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    # Get database statistics
    stats = db.get_statistics()
    
    # Get recent logs
    recent_logs = db.get_recent_logs(limit=10)
    
    logs_data = []
    for log in recent_logs:
        logs_data.append({
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'driver_name': log.driver_name or 'UNKNOWN',
            'similarity_score': f"{log.similarity_score:.3f}",
            'authorized': log.authorized,
            'processing_time_ms': f"{log.processing_time_ms:.0f}"
        })
    
    # Get daily statistics
    daily_stats = db.get_daily_statistics()
    
    return jsonify({
        'status': 'running' if engine.is_running else 'stopped',
        'threshold': engine.face_matcher.get_threshold(),
        'stats': stats,
        'daily_stats': daily_stats,
        'recent_logs': logs_data
    })


@app.route('/api/logs/image/<path:image_path>')
@login_required
def get_log_image(image_path):
    """Serve captured verification images"""
    # Security: Ensure we only serve from the alert image directory
    base_dir = Path(config.get('logging.alert_image_path', 'data/alerts/captured_images/')).absolute()
    image_full_path = Path(image_path).absolute()
    
    if not str(image_full_path).startswith(str(base_dir)):
        return jsonify({'error': 'Unauthorized path'}), 403
        
    if not image_full_path.exists():
        return jsonify({'error': 'Image not found'}), 404
        
    return Response(open(image_full_path, 'rb').read(), mimetype='image/jpeg')


@app.route('/api/driver/status')
def get_driver_status():
    """Get simplified status for driver feedback screen"""
    engine = get_engine()
    
    if engine is None:
        return jsonify({
            'state': 'error',
            'status_display': 'SYSTEM ERROR',
            'instruction': 'Please contact technician'
        })
    
    return jsonify(engine.get_driver_status())


@app.route('/api/enroll/live', methods=['POST'])
@login_required
def enroll_capture_live():
    """Capture current frame from live stream for enrollment preview"""
    engine = get_engine()
    with engine._frame_lock:
        if engine.latest_frame is None:
            return jsonify({'error': 'Camera not ready'}), 400
        frame = engine.latest_frame.copy()
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': img_base64})


@app.route('/api/enroll/upload', methods=['POST'])
@login_required
def enroll_upload_image():
    """Handle uploaded image for enrollment preview"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return jsonify({'image': img_base64})


@app.route('/api/enroll/save', methods=['POST'])
@login_required
def enroll_save_driver():
    """Process biometrics and save new driver"""
    data = request.json
    name = data.get('name')
    driver_id = data.get('driver_id')
    img_base64 = data.get('image')
    
    if not name or not driver_id or not img_base64:
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Decode image
    try:
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]
        img_data = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
    engine = get_engine()
    success, message = engine.enroll_new_driver(name, driver_id, image)
    
    if success:
        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 400


@app.route('/api/drivers')
@login_required
def get_drivers():
    """Get list of enrolled drivers"""
    drivers = db.get_all_drivers()
    
    drivers_data = []
    for driver in drivers:
        drivers_data.append({
            'driver_id': driver.driver_id,
            'name': driver.name,
            'email': driver.email or 'N/A',
            'enrollment_date': driver.enrollment_date.strftime('%Y-%m-%d %H:%M:%S'),
            'status': driver.status
        })
    
    return jsonify({'drivers': drivers_data})


@app.route('/api/statistics')
@login_required
def get_statistics():
    """Get detailed statistics"""
    stats = db.get_statistics()
    
    # Get unauthorized attempts
    unauthorized = db.get_unauthorized_attempts(limit=20)
    
    unauthorized_data = []
    for log in unauthorized:
        unauthorized_data.append({
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'similarity_score': f"{log.similarity_score:.3f}",
            'image_path': log.image_path
        })
    
    return jsonify({
        'stats': stats,
        'unauthorized_attempts': unauthorized_data
    })


def main():
    """Run Flask application"""
    print("\n" + "="*60)
    print("DRIVER VERIFICATION DASHBOARD")
    print("="*60)
    print(f"Starting Flask server...")
    print(f"Dashboard URL: http://localhost:5000")
    print(f"Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Check if drivers are enrolled
    stats = db.get_statistics()
    if stats['total_drivers'] == 0:
        print("WARNING: No drivers enrolled!")
        print("Enroll drivers using: python scripts/enroll_driver.py")
        print()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=config.debug_mode, threaded=True)
    except Exception as e:
        print(f"CRITICAL: Flask application crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if engine:
            engine.stop()


if __name__ == '__main__':
    main()
