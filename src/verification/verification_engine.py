"""
Verification Engine Module
Main orchestrator for real-time driver verification
"""

import cv2
import numpy as np
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from deepface import DeepFace

from verification.video_stream import VideoStream
from verification.liveness_detector import LivenessDetector
from verification.face_matcher import FaceMatcher
from enrollment.face_processor import FaceProcessor
from database.db_manager import DatabaseManager
from database.models import VerificationLog
from utils.config import config
from alerting.email_service import EmailService


class VerificationEngine:
    """Main verification engine for real-time driver authentication"""
    
    def __init__(self):
        """Initialize verification engine"""
        print("\n" + "="*60)
        print("INITIALIZING VERIFICATION ENGINE")
        print("="*60 + "\n")
        
        self.video_stream = VideoStream()
        self.face_processor = FaceProcessor()
        self.liveness_detector = LivenessDetector()
        self.face_matcher = FaceMatcher()
        self.db = DatabaseManager()
        
        self.is_running = False
        self.last_verification_time = 0
        self.verification_cooldown = 2.0  # Seconds between verifications
        
        # Web streaming support
        self._frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_result = None  # Store last verification result
        
        # Email Alerting
        self.email_service = EmailService()
        self.alert_recipients = config.get('email.alert_recipients', [])
        
        print("✓ Verification engine initialized\n")
    

    def save_verification_image(self, frame: np.ndarray, authorized: bool, driver_name: str = None) -> Optional[str]:
        """
        Save verification attempt image
        
        Args:
            frame: Frame to save
            authorized: Whether verification was authorized
            driver_name: Driver name (if authorized)
            
        Returns:
            Path to saved image or None
        """
        # Only save unauthorized images by default (privacy)
        if authorized and not config.get('logging.save_authorized_images', False):
            return None
        
        if not authorized and not config.get('logging.save_unauthorized_images', True):
            return None
        
        try:
            # Create directory if needed
            image_dir = Path(config.alert_image_path)
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "authorized" if authorized else "unauthorized"
            name_part = f"_{driver_name}" if driver_name else ""
            filename = f"{timestamp}_{status}{name_part}.jpg"
            
            filepath = image_dir / filename
            
            # Save image
            cv2.imwrite(str(filepath), frame)
            
            return str(filepath)
            
        except Exception as e:
            print(f"ERROR: Failed to save image: {e}")
            return None
    
    def verify_frame(self, frame: np.ndarray, check_liveness: bool = True) -> Tuple[bool, dict]:
        """
        Verify a single frame
        
        Args:
            frame: Input frame
            check_liveness: Whether to perform liveness detection
            
        Returns:
            Tuple of (success, result_dict)
        """
        start_time = time.time()
        
        result = {
            'authorized': False,
            'driver_id': None,
            'driver_name': None,
            'similarity_score': 0.0,
            'liveness_passed': False,
            'processing_time_ms': 0.0,
            'status_message': '',
            'image_path': None
        }
        
        # Detect face
        detection = self.face_processor.detect_face(frame)
        
        if detection is None:
            result['status_message'] = "No face detected"
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return False, result
        
        # Check liveness if enabled
        if check_liveness:
            is_live, liveness_status, liveness_confidence = self.liveness_detector.check_liveness(frame)
            result['liveness_passed'] = is_live
            
            if not is_live:
                result['status_message'] = f"Liveness check: {liveness_status}"
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                return False, result
        else:
            result['liveness_passed'] = True
        
        # Process face
        preprocessed_face, status = self.face_processor.process_for_enrollment(frame)
        
        if preprocessed_face is None:
            result['status_message'] = f"Face processing failed: {status}"
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return False, result
        
        # Generate embedding
        embedding = self.face_processor.generate_embedding(preprocessed_face)
        
        if embedding is None:
            result['status_message'] = "Failed to generate embedding"
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return False, result
        
        # Match against enrolled drivers
        is_authorized, driver_id, driver_name, similarity = self.face_matcher.verify_identity(embedding)
        
        result['authorized'] = is_authorized
        result['driver_id'] = driver_id
        result['driver_name'] = driver_name
        result['similarity_score'] = similarity
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        
        if is_authorized:
            result['status_message'] = f"AUTHORIZED: {driver_name} ({similarity:.3f})"
        else:
            if driver_name:
                result['status_message'] = f"UNAUTHORIZED: Best match '{driver_name}' ({similarity:.3f})"
            else:
                result['status_message'] = "UNAUTHORIZED: No enrolled drivers"
        
        # Save image
        result['image_path'] = self.save_verification_image(frame, is_authorized, driver_name)
        
        # Trigger email alert for unauthorized if enabled
        if not is_authorized and self.email_service.is_configured:
            result['email_sent'] = False # Will be updated in _trigger_alert
            
        return True, result
    
    def log_verification(self, result: dict):
        """
        Log verification attempt to database
        
        Args:
            result: Verification result dictionary
        """
        log = VerificationLog(
            driver_id=result['driver_id'],
            driver_name=result['driver_name'],
            similarity_score=result['similarity_score'],
            authorized=result['authorized'],
            processing_time_ms=result['processing_time_ms'],
            image_path=result['image_path'],
            liveness_passed=result['liveness_passed']
        )
        
        self.db.log_verification(log)
    
    def run_continuous_verification(self, show_preview: bool = True, enable_liveness: bool = True):
        """
        Run continuous verification loop
        
        Args:
            show_preview: Whether to show live video preview
            enable_liveness: Whether to enable liveness detection
        """
        print("\n" + "="*60)
        print("STARTING CONTINUOUS VERIFICATION")
        print("="*60)
        print(f"Liveness Detection: {'ENABLED' if enable_liveness else 'DISABLED'}")
        print(f"Video Preview: {'ENABLED' if show_preview else 'DISABLED'}")
        print(f"Similarity Threshold: {self.face_matcher.get_threshold()}")
        print("Press 'q' or ESC to quit")
        print("="*60 + "\n")
        
        # Start video stream
        if not self.video_stream.start():
            print("ERROR: Failed to start video stream")
            return
        
        self.is_running = True
        frame_count = 0
        verification_count = 0
        
        try:
            while self.is_running:
                try:
                    # Read frame
                    frame = self.video_stream.read_frame()
                    
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    display_frame = frame.copy()
                    
                    # Check if enough time has passed since last verification
                    current_time = time.time()
                    can_verify = (current_time - self.last_verification_time) >= self.verification_cooldown
                    
                    if can_verify:
                        # Check brightness
                        brightness = self.video_stream.get_brightness(frame)
                        is_too_dark = brightness < 30
                        
                        # Perform verification
                        success, result = self.verify_frame(frame, check_liveness=enable_liveness)
                        
                        if is_too_dark:
                            result['status_message'] = "LOW LIGHT: Please improve lighting"
                        
                        # Special message if no drivers enrolled
                        stats = self.db.get_statistics()
                        if stats['total_drivers'] == 0:
                            result['status_message'] = "ENROLL DRIVERS: Use Dashboard Registration"
                        
                        if success:
                            verification_count += 1
                            self.last_verification_time = current_time
                            
                            # Log verification
                            self.log_verification(result)
                            
                            # Print result if not running in background
                            if show_preview:
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Verification #{verification_count}")
                                print(f"  Status: {result['status_message']}")
                                print(f"  Processing Time: {result['processing_time_ms']:.2f} ms")
                            
                            # Trigger alert if unauthorized and brightness is okay
                            if not result['authorized'] and not is_too_dark:
                                self._trigger_alert(result)
                        
                        # Draw result on frame
                        display_frame = self._draw_verification_result(display_frame, result)
                    else:
                        # Show cooldown status
                        remaining = self.verification_cooldown - (current_time - self.last_verification_time)
                        cv2.putText(display_frame, f"Cooldown: {remaining:.1f}s", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Update latest frame and result for web streaming
                    with self._frame_lock:
                        self.latest_frame = display_frame.copy()
                        if can_verify:
                            self.latest_result = result.copy()
                    
                    # Show preview (desktop window)
                    if show_preview:
                        cv2.imshow("Driver Verification System", display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            break
                except Exception as e:
                    print(f"ERROR in verification loop iteration: {e}")
                    traceback.print_exc()
                    time.sleep(1) # Wait before retry
                
        except KeyboardInterrupt:
            print("\nVerification interrupted by user")
        finally:
            self.stop()
            print(f"\nVerification Session Summary: {verification_count} verifications performed")
    
    def _draw_verification_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw verification result on frame"""
        annotated = frame.copy()
        
        if result['authorized']:
            status_color = (0, 255, 0)
            status_text = "AUTHORIZED"
        elif "No face" in result.get('status_message', ''):
            status_color = (128, 128, 128)
            status_text = "SCANNING..."
        else:
            status_color = (0, 0, 255)
            status_text = "UNAUTHORIZED"
            
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), status_color, -1)
        cv2.putText(annotated, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        if result.get('status_message'):
            cv2.putText(annotated, result['status_message'], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated
    
    def _trigger_alert(self, result: dict):
        """Trigger alert for unauthorized access"""
        print("\nALERT: UNAUTHORIZED ACCESS ATTEMPT")
        if self.email_service.is_configured:
            success = self.email_service.send_unauthorized_alert(
                recipients=self.alert_recipients,
                similarity_score=result['similarity_score'],
                best_match_name=result.get('driver_name'),
                image_path=result['image_path']
            )
            result['email_sent'] = success

    def stop(self):
        """Stop verification engine"""
        self.is_running = False
        self.video_stream.stop()
        cv2.destroyAllWindows()
        print("Verification engine stopped")

    def get_driver_status(self) -> dict:
        """
        Get simplified status for the driver feedback screen
        """
        if not self.is_running:
            return {
                'state': 'ready',
                'status_display': 'SYSTEM READY',
                'instruction': 'Initializing camera...'
            }
            
        result = getattr(self, 'latest_result', None)
        
        state = 'scanning'
        status_display = 'SCANNING'
        instruction = 'Please look at the camera'
        
        if result:
            msg = result.get('status_message', '').upper()
            if "LOW LIGHT" in msg:
                state = 'warning'
                status_display = 'LOW LIGHT'
                instruction = 'Please improve lighting'
            elif "ENROLL" in msg:
                state = 'warning'
                status_display = 'SETUP REQUIRED'
                instruction = 'Contact authority'
            elif result.get('authorized'):
                state = 'authorized'
                status_display = 'ACCESS GRANTED'
                instruction = 'Driver verified successfully'
            elif not result.get('liveness_passed') and result.get('similarity_score', 0) > 0:
                state = 'warning'
                status_display = 'LIVENESS FAILED'
                instruction = 'Please blink naturally'
            elif result.get('similarity_score', 0) > 0:
                state = 'unauthorized'
                status_display = 'ACCESS DENIED'
                instruction = 'Unauthorized driver detected'
                
        return {
            'state': state,
            'status_display': status_display,
            'instruction': instruction
        }

    def enroll_new_driver(self, name: str, driver_id_str: str, image: np.ndarray) -> Tuple[bool, str]:
        """
        Enroll a new driver from a provided image
        """
        preprocessed, status = self.face_processor.process_for_enrollment(image)
        if preprocessed is None:
            return False, f"Enrollment failed: {status}"
            
        embedding = self.face_processor.generate_embedding(preprocessed)
        if embedding is None:
            return False, "Enrollment failed: Could not generate biometric signature"
            
        try:
            self.db.enroll_driver(name, embedding, id_number=driver_id_str)
            self.face_matcher.load_enrolled_drivers()
            return True, f"Successfully enrolled {name}"
        except Exception as e:
            return False, f"Database error: {str(e)}"


if __name__ == "__main__":
    engine = VerificationEngine()
    stats = engine.db.get_statistics()
    if stats['total_drivers'] == 0:
        print("\nWARNING: No drivers enrolled!")
    else:
        print(f"\nFound {stats['total_drivers']} enrolled driver(s)")
        engine.run_continuous_verification(show_preview=True, enable_liveness=True)
