"""
Enrollment Manager Module
Orchestrates the complete enrollment process
"""

import cv2
import numpy as np
from deepface import DeepFace
from typing import Optional, Tuple
from enrollment.camera_capture import CameraCapture
from enrollment.face_processor import FaceProcessor
from database.db_manager import DatabaseManager
from utils.config import config


class EnrollmentManager:
    """Manages the complete driver enrollment process"""
    
    def __init__(self):
        """Initialize enrollment manager"""
        self.camera = CameraCapture()
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
        print("Enrollment manager initialized")
    

    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding quality
        
        Args:
            embedding: FaceNet embedding
            
        Returns:
            True if valid, False otherwise
        """
        # Check dimensionality
        if embedding.shape[0] != 128:
            print(f"ERROR: Invalid embedding dimension: {embedding.shape[0]}")
            return False
        
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print("ERROR: Embedding contains NaN or Inf values")
            return False
        
        # Check embedding magnitude (should be normalized)
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1 or magnitude > 100:
            print(f"WARNING: Unusual embedding magnitude: {magnitude}")
        
        return True
    
    def enroll_driver(self, name: str, email: str = None, interactive: bool = True) -> Tuple[bool, str]:
        """
        Enroll a new driver
        
        Args:
            name: Driver's name
            email: Optional email address
            interactive: If True, use interactive camera preview
            
        Returns:
            Tuple of (success, message)
        """
        print(f"\n{'='*60}")
        print(f"ENROLLING DRIVER: {name}")
        print(f"{'='*60}\n")
        
        # Check if driver already exists
        if self.db.driver_exists(name):
            return False, f"Driver '{name}' is already enrolled"
        
        # Initialize camera
        if not self.camera.initialize():
            return False, "Failed to initialize camera"
        
        try:
            # Capture image(s)
            if interactive:
                frame = self.camera.capture_with_preview(
                    f"Enrollment: {name} - Press SPACE to capture, ESC to cancel"
                )
                
                if frame is None:
                    return False, "Enrollment cancelled"
                
                frames = [frame]
            else:
                # Capture multiple frames automatically
                num_frames = config.get('system.enrollment_image_count', 5)
                print(f"Capturing {num_frames} frames...")
                frames = self.camera.capture_multiple_frames(num_frames)
                
                if not frames:
                    return False, "Failed to capture frames"
            
            # Select best frame
            best_frame, quality = self.camera.select_best_frame(frames)
            print(f"Selected frame with quality: {quality:.2f}")
            
            if quality < 0.4:
                return False, f"Frame quality too low: {quality:.2f}"
            
            # Process face
            print("Processing face...")
            preprocessed_face, status = self.face_processor.process_for_enrollment(best_frame)
            
            if preprocessed_face is None:
                return False, f"Face processing failed: {status}"
            
            print(f"Face processing: {status}")
            
            # Generate embedding
            print("Generating biometric embedding...")
            embedding = self.face_processor.generate_embedding(preprocessed_face)
            
            if embedding is None:
                return False, "Failed to generate embedding"
            
            # Validate embedding
            if not self.validate_embedding(embedding):
                return False, "Invalid embedding generated"
            
            print(f"Embedding generated: {embedding.shape}")
            
            # Store in database
            print("Storing in database...")
            driver_id = self.db.enroll_driver(name, embedding, email)
            
            print(f"\n{'='*60}")
            print(f"✓ ENROLLMENT SUCCESSFUL")
            print(f"  Driver ID: {driver_id}")
            print(f"  Name: {name}")
            print(f"  Email: {email or 'N/A'}")
            print(f"  Embedding dimension: {embedding.shape[0]}")
            print(f"{'='*60}\n")
            
            # Show success preview
            if interactive:
                self._show_enrollment_success(best_frame, name, driver_id)
            
            return True, f"Driver '{name}' enrolled successfully (ID: {driver_id})"
            
        except Exception as e:
            return False, f"Enrollment error: {str(e)}"
        
        finally:
            self.camera.release()
    
    def _show_enrollment_success(self, image: np.ndarray, name: str, driver_id: int, duration_ms: int = 3000):
        """
        Show enrollment success message
        
        Args:
            image: Enrolled image
            name: Driver name
            driver_id: Driver ID
            duration_ms: Display duration in milliseconds
        """
        display = image.copy()
        
        # Add success overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 100), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        # Add text
        cv2.putText(display, "ENROLLMENT SUCCESSFUL", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, f"Name: {name} | ID: {driver_id}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enrollment Success", display)
        cv2.waitKey(duration_ms)
        cv2.destroyAllWindows()
    
    def list_enrolled_drivers(self):
        """List all enrolled drivers"""
        drivers = self.db.get_all_drivers()
        
        if not drivers:
            print("No drivers enrolled yet")
            return
        
        print(f"\n{'='*80}")
        print(f"ENROLLED DRIVERS ({len(drivers)})")
        print(f"{'='*80}")
        print(f"{'ID':<6} {'Name':<30} {'Email':<30} {'Enrolled':<20}")
        print(f"{'-'*80}")
        
        for driver in drivers:
            driver_id = driver.driver_id
            name = driver.name
            email = driver.email or 'N/A'
            enrolled = driver.enrollment_date.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{driver_id:<6} {name:<30} {email:<30} {enrolled:<20}")
        
        print(f"{'='*80}\n")
    
    def test_camera(self):
        """Test camera functionality"""
        print("Testing camera...")
        
        if not self.camera.initialize():
            print("ERROR: Failed to initialize camera")
            return
        
        print("Camera initialized successfully")
        print("Press ESC to exit")
        
        while True:
            frame = self.camera.capture_frame()
            
            if frame is None:
                break
            
            # Show quality
            quality = self.camera.assess_frame_quality(frame)
            cv2.putText(frame, f"Quality: {quality:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect face
            detection = self.face_processor.detect_face(frame)
            if detection:
                frame = self.face_processor.draw_detection(frame, detection)
            
            cv2.imshow("Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        self.camera.release()
        print("Camera test completed")


if __name__ == "__main__":
    # Test enrollment manager
    print("Testing enrollment manager...")
    
    manager = EnrollmentManager()
    
    # List enrolled drivers
    manager.list_enrolled_drivers()
    
    # Test camera
    print("\nStarting camera test...")
    manager.test_camera()
