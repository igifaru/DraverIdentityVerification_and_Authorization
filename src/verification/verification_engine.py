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
from typing import Optional, Tuple, Dict
from deepface import DeepFace

from verification.video_stream import VideoStream
from verification.liveness_detector import LivenessDetector
from verification.face_matcher import FaceMatcher
from verification.verification_result_handler import VerificationResultHandler
from enrollment.face_processor import FaceProcessor
from database.db_manager import DatabaseManager
from database.models import VerificationLog
from utils.config import config
from utils.constants import CAPTURED_IMAGES_DIR
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
        
        # Liveness Detection Config
        self.enable_liveness_check = config.get('verification.enable_liveness', True)
        if self.enable_liveness_check:
            self.liveness_detector = LivenessDetector()
        else:
            self.liveness_detector = None
            print("WARNING: Liveness detection DISABLED by configuration")

        self.face_matcher = FaceMatcher()
        self.db = DatabaseManager()
        self.result_handler = VerificationResultHandler()
        
        self.is_running = False
        self.last_verification_time = 0
        
        # Cooldown Config
        self.enable_cooldown = config.get('verification.enable_cooldown', True)
        if self.enable_cooldown:
            self.verification_cooldown = config.get('verification.cooldown_seconds', 2.0)
        else:
            self.verification_cooldown = 0.0
            print("WARNING: Verification cooldown DISABLED by configuration")
        
        # Web streaming support
        self._frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_result = None  # Store last verification result
        
        # Email Alerting
        self.email_service = EmailService()
        self.alert_recipients = config.get('email.alert_recipients', [])
        
        print("[OK] Verification engine initialized\n")
    

    def save_verification_image(self, frame: np.ndarray, authorized: bool, driver_name: str = None) -> Optional[str]:
        """
        Save verification attempt image
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
        """Verify a single frame"""
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
        # Check liveness if enabled
        if check_liveness and self.liveness_detector:
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
        """Log verification attempt to database"""
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
    
    def start_camera(self):
        """Start the video stream if not already running"""
        if not self.video_stream.is_running:
             if self.video_stream.start():
                 print("Camera started successfully")
                 return True
             else:
                 print("ERROR: Failed to start camera")
                 return False
        return True

    def run_continuous_verification(self, show_preview: bool = True, enable_liveness: Optional[bool] = None):
        """Run continuous verification loop"""
        # Resolve configuration
        if enable_liveness is None:
            enable_liveness = self.enable_liveness_check

        print("\n" + "="*60)
        print("STARTING CONTINUOUS VERIFICATION")
        print("="*60)
        print(f"Liveness Detection: {'ENABLED' if enable_liveness else 'DISABLED'}")
        print(f"Cooldown: {'ENABLED' if self.enable_cooldown else 'DISABLED'} ({self.verification_cooldown}s)")
        print(f"Video Preview: {'ENABLED' if show_preview else 'DISABLED'}")
        print(f"Similarity Threshold: {self.face_matcher.get_threshold()}")
        print("Press 'q' or ESC to quit")
        print("="*60 + "\n")
        
        # Start video stream
        if not self.start_camera():
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
                        time.sleep(0.01 if not self.enable_cooldown else 0.1)
                        continue
                    
                    frame_count += 1
                    display_frame = frame.copy()
                    
                    # Check if enough time has passed since last verification
                    current_time = time.time()
                    if self.enable_cooldown:
                        can_verify = (current_time - self.last_verification_time) >= self.verification_cooldown
                    else:
                        can_verify = True
                    
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
                        
                        # Draw result on frame using Handler
                        display_frame = self.result_handler.draw_result(display_frame, result)
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
        """Get simplified status for the driver feedback screen"""
        return self.result_handler.get_driver_status(
            getattr(self, 'latest_result', None), 
            self.is_running
        )

    def enroll_new_driver(self, name: str, driver_id_str: str, images: list) -> Tuple[bool, str]:
        """Enroll a new driver from a list of provided images (multi-sample)"""
        if not images:
            return False, "No images provided for enrollment"
            
        print(f"\n=== ENROLLMENT DEBUG: Processing {len(images)} images for {name} ===")
        embeddings = []
        errors = []
        
        for idx, img in enumerate(images):
            print(f"Sample {idx+1}: Image type={type(img)}, shape={img.shape if hasattr(img, 'shape') else 'N/A'}")
            
            preprocessed, status = self.face_processor.process_for_enrollment(img)
            if preprocessed is not None:
                print(f"Sample {idx+1}: Face detected and preprocessed successfully")
                embedding = self.face_processor.generate_embedding(preprocessed)
                if embedding is not None:
                    embeddings.append(embedding)
                    print(f"Sample {idx+1}: Embedding generated (shape={embedding.shape})")
                else:
                    errors.append(f"Sample {idx+1}: Could not generate biometric signature")
                    print(f"Sample {idx+1}: Failed to generate embedding")
            else:
                errors.append(f"Sample {idx+1}: {status}")
                print(f"Sample {idx+1}: Face detection failed - {status}")
                
        print(f"=== ENROLLMENT SUMMARY: {len(embeddings)}/{len(images)} samples successful ===\n")
        
        if len(embeddings) < 1:
            error_msg = "; ".join(errors[:3])
            return False, f"Enrollment failed: {error_msg}"
            
        # Averaging Logic: Create a robust biometric signature from all valid samples
        # This reduces noise and improves matching consistency
        mean_embedding = np.mean(embeddings, axis=0)
        
        try:
            self.db.enroll_driver(name, mean_embedding, id_number=driver_id_str)
            # FaceMatcher reads from DB directly, no need to reload
            return True, f"Successfully enrolled {name} with {len(embeddings)}/5 biometric samples"
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
