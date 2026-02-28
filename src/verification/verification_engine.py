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
        """
        Run the full verification pipeline on a single video frame.

        Pipeline:
          1. Detect + align + crop face  (process_for_enrollment — single MTCNN call)
          2. Liveness check              (optional)
          3. Generate FaceNet embedding
          4. Match against enrolled drivers

        Returns:
            (success, result_dict)
            success=False means no usable face was found (not that access was denied).
        """
        start_time = time.time()

        result = {
            'authorized':         False,
            'driver_id':          None,
            'driver_name':        None,
            'similarity_score':   0.0,
            'liveness_passed':    False,
            'processing_time_ms': 0.0,
            'status_message':     '',
            'image_path':         None,
        }

        # ---- Step 1: detect + align + crop (single MTCNN call, threshold=0.80) ----
        face_crop, detect_status = self.face_processor.process_for_enrollment(
            frame, min_confidence=0.80
        )

        if face_crop is None:
            result['status_message'] = detect_status   # "No face detected", quality msg, etc.
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return False, result

        # ---- Step 2: liveness check ----
        if check_liveness and self.liveness_detector:
            is_live, liveness_status, _ = self.liveness_detector.check_liveness(frame)
            result['liveness_passed'] = is_live
            if not is_live:
                result['status_message'] = f"Liveness check: {liveness_status}"
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                return False, result
        else:
            result['liveness_passed'] = True

        # ---- Step 3: embedding ----
        embedding = self.face_processor.generate_embedding(face_crop)
        if embedding is None:
            result['status_message'] = "Failed to generate embedding"
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return False, result

        # ---- Step 4: identity match ----
        is_authorized, driver_id, driver_name, similarity = \
            self.face_matcher.verify_identity(embedding)

        result['authorized']       = is_authorized
        result['driver_id']        = driver_id
        result['driver_name']      = driver_name
        result['similarity_score'] = similarity
        result['processing_time_ms'] = (time.time() - start_time) * 1000

        if is_authorized:
            result['status_message'] = f"AUTHORIZED: {driver_name} ({similarity:.3f})"
        elif driver_id:
            result['status_message'] = (
                f"UNAUTHORIZED: Best match '{driver_name}' ({similarity:.3f})"
            )
        else:
            result['status_message'] = "UNAUTHORIZED: No confident match"

        # Save image for unauthorized attempts only
        result['image_path'] = self.save_verification_image(
            frame, is_authorized, driver_name
        )

        return True, result

    
    def log_verification(self, result: dict):
        """Log verification attempt to database.

        psycopg2 cannot serialize numpy scalar types (numpy.bool_, numpy.float32,
        numpy.int64, …).  We cast every field to the equivalent native Python type
        before building the VerificationLog to avoid ProgrammingError at insert time.
        """
        log = VerificationLog(
            driver_id=int(result['driver_id']) if result['driver_id'] is not None else None,
            driver_name=result['driver_name'],
            similarity_score=float(result['similarity_score']),
            authorized=bool(result['authorized']),
            processing_time_ms=float(result['processing_time_ms']),
            image_path=result['image_path'],
            liveness_passed=bool(result['liveness_passed']),
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
        """Run continuous verification loop.
        
        The camera is NOT started automatically. The engine idles in
        standby mode until the camera is started externally (e.g. by the
        enrollment workflow). When the camera is on, the engine processes
        frames for face verification; when it is off, the loop sleeps.
        """
        # Resolve configuration
        if enable_liveness is None:
            enable_liveness = self.enable_liveness_check

        print("\n" + "="*60)
        print("VERIFICATION ENGINE READY (camera OFF — standby)")
        print("="*60)
        print(f"Liveness Detection: {'ENABLED' if enable_liveness else 'DISABLED'}")
        print(f"Cooldown: {'ENABLED' if self.enable_cooldown else 'DISABLED'} ({self.verification_cooldown}s)")
        print(f"Similarity Threshold: {self.face_matcher.get_threshold()}")
        print("="*60 + "\n")
        
        # NOTE: camera is NOT started here — it stays OFF until enrollment
        
        self.is_running = True
        frame_count = 0
        verification_count = 0
        
        try:
            while self.is_running:
                try:
                    # If camera is not running, idle in standby
                    if not self.video_stream.is_running:
                        time.sleep(0.5)
                        continue

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
                        brightness = self.face_processor._brightness(frame)

                        # Tier 1: completely dark / no signal
                        # face_processor will also reject this, but set the message early
                        # so the status API reflects it even before verify_frame is called.
                        is_no_signal = brightness < self.face_processor._BRIGHTNESS_NO_SIGNAL
                        is_low_light = (not is_no_signal and
                                        brightness < self.face_processor._BRIGHTNESS_LOW_LIGHT)

                        success, result = self.verify_frame(frame, check_liveness=enable_liveness)

                        # Override status message for camera/light conditions
                        if is_no_signal and not success:
                            result['status_message'] = (
                                "NO SIGNAL: Camera may be blocked or no light source"
                            )
                        elif is_low_light and not success:
                            result['status_message'] = "LOW LIGHT: Please improve lighting"

                        # Notify if no drivers enrolled
                        stats = self.db.get_statistics()
                        if stats['total_drivers'] == 0:
                            result['status_message'] = "ENROLL DRIVERS: Use Dashboard Registration"

                        if success:
                            verification_count += 1
                            self.last_verification_time = current_time
                            self.log_verification(result)

                            if show_preview:
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                                      f"Verification #{verification_count}")
                                print(f"  Status: {result['status_message']}")
                                print(f"  Processing Time: {result['processing_time_ms']:.2f} ms")

                            # Alert for every unauthorized result
                            # (no-signal frames never reach here — verify_frame returns False)
                            if not result['authorized']:
                                self._trigger_alert(result)

                        # Draw result on frame
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
                            result['result_timestamp'] = time.time()   # lets the dashboard detect new events
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
        if self.video_stream.is_running:
            self.video_stream.stop()
        cv2.destroyAllWindows()
        print("Verification engine stopped")
    
    def get_driver_status(self) -> dict:
        """Get simplified status for the driver feedback screen"""
        return self.result_handler.get_driver_status(
            getattr(self, 'latest_result', None), 
            self.is_running
        )

    def enroll_new_driver(self, name: str, license_number: str, images: list,
                          category: str = 'A') -> Tuple[bool, str]:
        """Enroll a new driver from a list of provided images (multi-sample).

        Pipeline per image:
          1. Detect + align + crop face (process_for_enrollment → raw BGR crop)
          2. Generate L2-normalised 128-d embedding (FaceNet via DeepFace)
          3. Validate: no NaN/Inf, non-zero norm
          4. Average valid embeddings and re-normalise the mean
          5. Persist to DB
        """
        if not images:
            return False, "No images provided for enrollment"

        print(f"\n=== ENROLLMENT: Processing {len(images)} images for {name} ===")
        embeddings = []
        errors = []

        for idx, img in enumerate(images):
            print(f"Sample {idx+1}/{len(images)}: shape={getattr(img, 'shape', 'N/A')}")

            face_crop, status = self.face_processor.process_for_enrollment(img)
            if face_crop is None:
                errors.append(f"Sample {idx+1}: {status}")
                print(f"  -> Skipped: {status}")
                continue

            embedding = self.face_processor.generate_embedding(face_crop)
            if embedding is None:
                errors.append(f"Sample {idx+1}: embedding generation failed")
                print(f"  -> Skipped: embedding generation failed")
                continue

            # Validate embedding quality
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                errors.append(f"Sample {idx+1}: NaN/Inf in embedding")
                print(f"  -> Skipped: NaN/Inf in embedding")
                continue

            norm = np.linalg.norm(embedding)
            if norm < 1e-6:
                errors.append(f"Sample {idx+1}: near-zero norm embedding")
                print(f"  -> Skipped: near-zero norm embedding")
                continue

            embeddings.append(embedding)
            print(f"  -> OK (dim={embedding.shape[0]}, norm={norm:.4f})")

        print(f"=== {len(embeddings)}/{len(images)} samples valid ===\n")

        if not embeddings:
            return False, "Enrollment failed: " + "; ".join(errors[:3])

        # Average the unit-sphere embeddings then re-normalise so the stored
        # vector is itself a unit vector (critical for correct cosine similarity)
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        mean_norm = np.linalg.norm(mean_emb)
        if mean_norm < 1e-8:
            return False, "Enrollment failed: mean embedding is degenerate"
        final_embedding = (mean_emb / mean_norm).astype(np.float32)
        print(f"[enroll] Final embedding: dim={final_embedding.shape[0]}, norm={np.linalg.norm(final_embedding):.6f}")

        # ---- Save enrollment portrait photo to disk ----
        photo_path = None
        try:
            from pathlib import Path
            photo_dir = Path(config.get('logging.alert_image_path', 'data/alert_images')).parent / 'enrollment_photos'
            photo_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name).strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_name}.jpg"
            filepath = photo_dir / filename
            cv2.imwrite(str(filepath), images[0])
            photo_path = str(filepath)
            print(f"[enroll] Portrait saved: {photo_path}")
        except Exception as exc:
            print(f"[enroll] Warning: could not save portrait photo: {exc}")

        try:
            self.db.enroll_driver(name, final_embedding,
                                  license_number=license_number,
                                  category=category,
                                  photo_path=photo_path)
            return True, (f"Successfully enrolled {name} (Category {category}) "
                          f"with {len(embeddings)}/{len(images)} biometric samples")
        except Exception as exc:
            return False, f"Database error: {exc}"




if __name__ == "__main__":
    engine = VerificationEngine()
    stats = engine.db.get_statistics()
    if stats['total_drivers'] == 0:
        print("\nWARNING: No drivers enrolled!")
    else:
        print(f"\nFound {stats['total_drivers']} enrolled driver(s)")
        engine.run_continuous_verification(show_preview=True, enable_liveness=True)
