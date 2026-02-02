"""
Camera Capture Module
Handles camera initialization and high-quality image capture for enrollment
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from utils.config import config


class CameraCapture:
    """Manages camera operations for enrollment"""
    
    def __init__(self, device_id: int = None, resolution: Tuple[int, int] = None):
        """
        Initialize camera capture
        
        Args:
            device_id: Camera device ID (uses config if not provided)
            resolution: Camera resolution (width, height) (uses config if not provided)
        """
        self.device_id = device_id if device_id is not None else config.camera_device_id
        self.resolution = resolution or config.camera_resolution
        self.cap = None
    
    def initialize(self) -> bool:
        """
        Initialize camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
            
            # Warm up camera
            for _ in range(5):
                self.cap.read()
            
            print(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {config.camera_fps} FPS")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            print("ERROR: Camera not initialized")
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("ERROR: Failed to capture frame")
            return None
        
        return frame
    
    def capture_multiple_frames(self, count: int = 5, delay_ms: int = 200) -> list:
        """
        Capture multiple frames with delay
        
        Args:
            count: Number of frames to capture
            delay_ms: Delay between captures in milliseconds
            
        Returns:
            List of captured frames
        """
        frames = []
        
        for i in range(count):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
                if i < count - 1:  # Don't wait after last frame
                    cv2.waitKey(delay_ms)
        
        return frames
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess frame quality based on brightness and sharpness
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (mean intensity)
        brightness = np.mean(gray) / 255.0
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize sharpness (typical range: 0-1000)
        sharpness_normalized = min(sharpness / 1000.0, 1.0)
        
        # Brightness should be around 0.4-0.7 for optimal quality
        brightness_score = 1.0 - abs(brightness - 0.55) / 0.55
        brightness_score = max(0, brightness_score)
        
        # Combined quality score (weighted average)
        quality = 0.4 * brightness_score + 0.6 * sharpness_normalized
        
        return quality
    
    def select_best_frame(self, frames: list) -> Tuple[Optional[np.ndarray], float]:
        """
        Select the best quality frame from a list
        
        Args:
            frames: List of frames
            
        Returns:
            Tuple of (best_frame, quality_score)
        """
        if not frames:
            return None, 0.0
        
        best_frame = None
        best_quality = 0.0
        
        for frame in frames:
            quality = self.assess_frame_quality(frame)
            if quality > best_quality:
                best_quality = quality
                best_frame = frame
        
        return best_frame, best_quality
    
    def capture_with_preview(self, window_name: str = "Enrollment - Press SPACE to capture, ESC to cancel") -> Optional[np.ndarray]:
        """
        Capture frame with live preview
        
        Args:
            window_name: Window title
            
        Returns:
            Captured frame or None if cancelled
        """
        if self.cap is None or not self.cap.isOpened():
            print("ERROR: Camera not initialized")
            return None
        
        print(f"\n{window_name}")
        print("Press SPACE to capture, ESC to cancel")
        
        captured_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            
            # Add instructions
            cv2.putText(display_frame, "SPACE: Capture | ESC: Cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show quality indicator
            quality = self.assess_frame_quality(frame)
            quality_text = f"Quality: {quality:.2f}"
            quality_color = (0, 255, 0) if quality > 0.6 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
            cv2.putText(display_frame, quality_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key
                captured_frame = frame.copy()
                print(f"Frame captured! Quality: {quality:.2f}")
                break
            elif key == 27:  # ESC key
                print("Capture cancelled")
                break
        
        cv2.destroyAllWindows()
        return captured_frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


if __name__ == "__main__":
    # Test camera capture
    print("Testing camera capture...")
    
    with CameraCapture() as camera:
        if camera.cap is not None:
            # Capture with preview
            frame = camera.capture_with_preview()
            
            if frame is not None:
                print(f"Captured frame shape: {frame.shape}")
                quality = camera.assess_frame_quality(frame)
                print(f"Frame quality: {quality:.2f}")
