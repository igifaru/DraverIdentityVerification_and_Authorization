"""
API Routes
Handles JSON API endpoints for the dashboard
"""
from flask import Blueprint, jsonify, current_app, request, session
import time

api_bp = Blueprint('api', __name__)

@api_bp.route('/status')
def get_status():
    """Get current verification status"""
    from utils.config import config
    
    engine = current_app.config['VERIFICATION_ENGINE']
    
    # Get latest stats
    stats = engine.db.get_daily_statistics()
    
    # Get active status
    status = {
        'system_status': 'active' if engine.is_running else 'stopped',
        'timestamp': time.time(),
        'stats': stats,
        'system_info': {
            'id': config.system_id,
            'vehicle_plate': config.vehicle_plate,
            'owner_name': config.owner_name
        }
    }
    
    # Get latest verification result if available
    if engine.latest_result:
        result = engine.latest_result
        status['latest_verification'] = {
            'authorized': result['authorized'],
            'driver_name': result['driver_name'],
            'similarity': float(result['similarity_score']),
            'message': result.get('status_message', ''),
            'liveness_passed': result['liveness_passed']
        }
    
    return jsonify(status)

@api_bp.route('/statistics')
def get_statistics():
    """Get detailed statistics"""
    engine = current_app.config['VERIFICATION_ENGINE']
    stats = engine.db.get_statistics()
    return jsonify(stats)

@api_bp.route('/audit')
def get_audit_logs():
    """Get recent system audit logs"""
    engine = current_app.config['VERIFICATION_ENGINE']
    limit = request.args.get('limit', default=50, type=int)
    logs = engine.db.get_audit_logs(limit)
    return jsonify([log.to_dict() for log in logs])

# Enrollment APIs
@api_bp.route('/enroll/live', methods=['POST'])
def enroll_capture_live():
    """Capture current frame from live stream for enrollment preview"""
    engine = current_app.config['VERIFICATION_ENGINE']
    
    # Start camera if not running (on-demand for enrollment)
    if not engine.video_stream.is_running:
        print("Starting camera for enrollment capture...")
        engine.start_camera()
        import time
        time.sleep(1.0) # Wait for camera warm-up
    
    with engine._frame_lock:
        if engine.latest_frame is None:
            # Fallback: try to read a frame directly
            frame = engine.video_stream.read_frame()
        else:
            frame = engine.latest_frame.copy()
    
    if frame is None:
        return jsonify({'success': False, 'message': 'No video stream available'})
        
    # Validate that a face is present (for preview feedback)
    # but return the ORIGINAL frame, not the preprocessed crop
    preprocessed, status = engine.face_processor.process_for_enrollment(frame)
    if preprocessed is None:
        return jsonify({'error': status})
        
    # Convert ORIGINAL frame to base64 for enrollment
    # The enrollment process will handle face detection and preprocessing
    import base64
    import cv2
    _, buffer = cv2.imencode('.jpg', frame)  # Changed from 'preprocessed' to 'frame'
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'image': img_str
    })

@api_bp.route('/enroll/save', methods=['POST'])
def enroll_save_driver():
    """Process biometrics and save new driver (supports multi-sample)"""
    data = request.json
    name = data.get('name')
    license_number = data.get('driver_id')

    # Accept an array of categories (multi-select) or fall back to single value
    raw_categories = data.get('categories', None)
    if raw_categories is None:
        # backward-compat: single category field
        single = data.get('category', 'A')
        raw_categories = [single]

    valid_codes = {'A', 'B', 'C', 'D', 'E'}
    selected = sorted(set(c.strip().upper() for c in raw_categories if c.strip().upper() in valid_codes))
    if not selected:
        return jsonify({'error': 'At least one valid category (A-E) must be selected.'}), 400

    category = ','.join(selected)  # e.g. "A,B,C"

    images_data = data.get('images', [])
    if not images_data and data.get('image'):
        images_data = [data.get('image')]

    if not all([name, license_number, images_data]):
        return jsonify({'error': 'Missing required fields: name, driver_id, and images array'}), 400

    decoded_images = []
    import base64
    import numpy as np
    import cv2
    
    for idx, image_data in enumerate(images_data):
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                print(f"Decoded sample {idx+1}: shape={image.shape}")
                decoded_images.append(image)
        except Exception as e:
            print(f"Error decoding sample {idx}: {e}")
            continue
            
    if not decoded_images:
        return jsonify({'error': 'Failed to decode any images'}), 400
    
    engine = current_app.config['VERIFICATION_ENGINE']
    success, message = engine.enroll_new_driver(name, license_number, decoded_images,
                                                category=category)
    
    if success:
        # Log audit event
        email = session.get('user_email', 'UNKNOWN')
        engine.db.log_audit("ENROLL_DRIVER", email, f"Enrolled driver: {name} (License: {license_number})", request.remote_addr)
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 500

@api_bp.route('/system/start', methods=['POST'])
def start_system():
    """Start the verification engine"""
    engine = current_app.config['VERIFICATION_ENGINE']
    if engine.is_running:
        return jsonify({'success': False, 'message': 'System already running'})
    
    import threading
    thread = threading.Thread(
        target=engine.run_continuous_verification,
        kwargs={'show_preview': False},
        daemon=True
    )
    thread.start()
    
    # Log audit event
    email = session.get('user_email', 'UNKNOWN')
    engine.db.log_audit("START_ENGINE", email, "Verification engine started manually", request.remote_addr)
    
    return jsonify({'success': True, 'message': 'System started'})

@api_bp.route('/system/stop', methods=['POST'])
def stop_system():
    """Stop the verification engine"""
    engine = current_app.config['VERIFICATION_ENGINE']
    if not engine.is_running:
        return jsonify({'success': False, 'message': 'System already stopped'})
    
    engine.stop()
    
    # Log audit event
    email = session.get('user_email', 'UNKNOWN')
    engine.db.log_audit("STOP_ENGINE", email, "Verification engine stopped manually", request.remote_addr)
    
    return jsonify({'success': True, 'message': 'System stopping...'})

@api_bp.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the camera (for enrollment workflow)"""
    engine = current_app.config['VERIFICATION_ENGINE']
    
    # Only stop camera if verification engine is not running
    if not engine.is_running and engine.video_stream.is_running:
        print("Stopping camera after enrollment...")
        engine.video_stream.stop()
        return jsonify({'success': True, 'message': 'Camera stopped'})
    
    return jsonify({'success': False, 'message': 'Camera in use or already stopped'})
