"""
Verification Engine Module
Main orchestrator for real-time driver verification
"""

import cv2
import numpy as np
import time
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
        import threading
        self._frame_lock = threading.Lock()
        self.latest_frame = None
        
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
                # Read frame
                frame = self.video_stream.read_frame()
                
                if frame is None:
                    # Give the camera a moment to recover and avoid tight loop
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
                        result['status_message'] = "ENROLL DRIVERS: Run 'python scripts/enroll_driver.py'"
                    
                    if success:
                        verification_count += 1
                        self.last_verification_time = current_time
                        
                        # Log verification
                        self.log_verification(result)
                        
                        # Print result
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
                
                # Update latest frame for web streaming
                with self._frame_lock:
                    self.latest_frame = display_frame.copy()
                
                # Show preview (desktop window)
                if show_preview:
                    cv2.imshow("Driver Verification System", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\nStopping verification...")
                        break
                
        except KeyboardInterrupt:
            print("\n\nVerification interrupted by user")
        
        finally:
            self.stop()
            
            print(f"\n{'='*60}")
            print("VERIFICATION SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Total Frames: {frame_count}")
            print(f"Total Verifications: {verification_count}")
            print(f"{'='*60}\n")
    
    def _draw_verification_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw verification result on frame"""
        annotated = frame.copy()
        
        # Draw status bar
        if result['authorized']:
            status_color = (0, 255, 0)  # Green
            status_text = "AUTHORIZED"
        elif "No face" in result.get('status_message', ''):
            status_color = (128, 128, 128)  # Gray
            status_text = "SCANNING..."
        else:
            status_color = (0, 0, 255)  # Red
            status_text = "UNAUTHORIZED"
            
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), status_color, -1)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), (255, 255, 255), 2)
        
        # Draw status text
        cv2.putText(annotated, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw status message (more detail)
        if result.get('status_message'):
            cv2.putText(annotated, result['status_message'], (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw similarity score (only if face detected)
        if result['similarity_score'] > 0:
            score_text = f"Similarity: {result['similarity_score']:.3f}"
            cv2.putText(annotated, score_text, (annotated.shape[1] - 250, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw processing time
        time_text = f"{result['processing_time_ms']:.0f} ms"
        cv2.putText(annotated, time_text, (annotated.shape[1] - 250, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def _trigger_alert(self, result: dict):
        """Trigger alert for unauthorized access"""
        print("\n" + "!"*60)
        print("⚠️  UNAUTHORIZED ACCESS ATTEMPT DETECTED")
        print("!"*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        if result['image_path']:
            print(f"Image saved: {result['image_path']}")
        print("!"*60 + "\n")
        
        # Email alert will be sent by the alerting module
        # This is a placeholder for alert triggering
    
    def stop(self):
        """Stop verification engine"""
        self.is_running = False
        self.video_stream.stop()
        cv2.destroyAllWindows()
        print("Verification engine stopped")


if __name__ == "__main__":
    # Test verification engine
    print("Testing verification engine...")
    
    engine = VerificationEngine()
    
    # Check if there are enrolled drivers
    stats = engine.db.get_statistics()
    
    if stats['total_drivers'] == 0:
        print("\nWARNING: No drivers enrolled!")
        print("Please enroll at least one driver before running verification.")
        print("Run: python scripts/enroll_driver.py")
    else:
        print(f"\nFound {stats['total_drivers']} enrolled driver(s)")
        print("Starting verification...")
        
        # Run continuous verification
        engine.run_continuous_verification(show_preview=True, enable_liveness=True)
