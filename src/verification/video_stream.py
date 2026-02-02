"""
Video Stream Module
Handles real-time video streaming for verification
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque
from utils.config import config


class VideoStream:
    """Manages real-time video stream for verification"""
    
    def __init__(self, device_id: int = None, resolution: Tuple[int, int] = None, fps: int = None):
        """
        Initialize video stream
        
        Args:
            device_id: Camera device ID (uses config if not provided)
            resolution: Camera resolution (width, height) (uses config if not provided)
            fps: Target FPS (uses config if not provided)
        """
        self.device_id = device_id if device_id is not None else config.camera_device_id
        self.resolution = resolution or config.camera_resolution
        self.fps = fps or config.camera_fps
        
        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=5)  # Buffer last 5 frames
        
        print(f"Video stream configured: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")
    
    def start(self) -> bool:
        """
        Start video stream
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use CAP_DSHOW on Windows for better compatibility if index is 0
            if sys.platform.startswith('win') and self.device_id == 0:
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Enable auto-focus and auto-exposure for better quality
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            # Warm up camera (discard first few frames)
            for _ in range(10):
                self.cap.read()
            
            self.is_running = True
            print(f"✓ Video stream started on camera {self.device_id}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start video stream: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the video stream
        
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.is_running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            # Reduce log spam for failed frames
            if not hasattr(self, '_fail_count'):
                self._fail_count = 0
            self._fail_count += 1
            if self._fail_count % 30 == 0:
                print(f"WARNING: Failed to read frame (count: {self._fail_count})")
            return None
        
        self._fail_count = 0
        
        # Add to buffer
        self.frame_buffer.append(frame.copy())
        
        return frame
    
    def get_buffered_frame(self, index: int = -1) -> Optional[np.ndarray]:
        """
        Get frame from buffer
        
        Args:
            index: Buffer index (-1 for most recent)
            
        Returns:
            Buffered frame or None
        """
        if not self.frame_buffer:
            return None
        
        try:
            return self.frame_buffer[index]
        except IndexError:
            return None
    
    def get_average_frame(self, num_frames: int = 3) -> Optional[np.ndarray]:
        """
        Get average of last N frames (reduces noise)
        
        Args:
            num_frames: Number of frames to average
            
        Returns:
            Averaged frame or None
        """
        if len(self.frame_buffer) < num_frames:
            return self.get_buffered_frame()
        
        frames = list(self.frame_buffer)[-num_frames:]
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        
        return avg_frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better face detection
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def get_brightness(self, frame: Optional[np.ndarray] = None) -> float:
        """
        Get average brightness of a frame
        
        Returns:
            Brightness value (0-255)
        """
        if frame is None:
            frame = self.get_buffered_frame()
            
        if frame is None:
            return 0.0
            
        # Convert to LAB and use L channel for brightness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        return float(np.mean(l))

    def get_fps(self) -> float:
        """
        Get actual FPS of the video stream
        
        Returns:
            Current FPS
        """
        if self.cap is None:
            return 0.0
        
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get actual resolution of the video stream
        
        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
    
    def stop(self):
        """Stop video stream"""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                print(f"WARNING: Exception during camera release: {e}")
            finally:
                self.cap = None
                self.is_running = False
                self.frame_buffer.clear()
                print("Video stream stopped")
    
    def restart(self) -> bool:
        """
        Restart video stream
        
        Returns:
            True if successful, False otherwise
        """
        self.stop()
        return self.start()
    
    def is_healthy(self) -> bool:
        """
        Check if video stream is healthy
        
        Returns:
            True if stream is running and can read frames
        """
        if not self.is_running or self.cap is None:
            return False
        
        # Try to read a frame
        ret, _ = self.cap.read()
        
        return ret
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


if __name__ == "__main__":
    # Test video stream
    print("Testing video stream...")
    
    with VideoStream() as stream:
        if stream.is_running:
            print(f"Stream resolution: {stream.get_resolution()}")
            print(f"Stream FPS: {stream.get_fps()}")
            print("\nPress ESC to exit")
            
            frame_count = 0
            
            while True:
                frame = stream.read_frame()
                
                if frame is None:
                    break
                
                frame_count += 1
                
                # Show frame info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {stream.get_fps():.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Buffer: {len(stream.frame_buffer)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Video Stream Test", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            
            cv2.destroyAllWindows()
            print(f"\nTotal frames processed: {frame_count}")
