"""
Liveness Detector Module
Implements Eye Aspect Ratio (EAR) based blink detection using MediaPipe Face Mesh
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from collections import deque
from typing import Optional, Tuple, List
from utils.config import config


class LivenessDetector:
    """Detects liveness using blink detection (EAR method) with MediaPipe"""
    
    def __init__(self, ear_threshold: float = None, blink_frames: int = None):
        """
        Initialize liveness detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold
            blink_frames: Minimum consecutive frames for blink
        """
        self.ear_threshold = ear_threshold or config.liveness_ear_threshold
        self.blink_frames = blink_frames or config.liveness_blink_frames
        
        # Initialize MediaPipe Face Mesh
        print("Initializing MediaPipe Face Mesh for liveness detection...")
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.is_available = True
        except (ImportError, AttributeError) as e:
            print(f"WARNING: MediaPipe liveness detection unavailable: {e}")
            self.is_available = False
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # Blink detection state
        self.frame_counter = 0
        self.total_blinks = 0
        self.ear_history = deque(maxlen=30)  # Keep last 30 EAR values
        self.blink_detected = False
        self.last_landmarks = None
        
        # MediaPipe Landmark Indices
        # Left Eye (Upper, Lower, Inner, Outer)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Right Eye
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        print(f"Liveness detector initialized (EAR threshold: {self.ear_threshold})")
    
    def calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        # Vertical distances
        v1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_landmarks_from_mesh(self, image: np.ndarray, landmarks) -> Tuple[Optional[List], Optional[List]]:
        """Extract eye landmarks from MediaPipe results"""
        h, w = image.shape[:2]
        
        try:
            # Get landmarks for first face
            face_landmarks = landmarks[0].landmark
            
            # Extract Left Eye
            left_eye = []
            for idx in self.LEFT_EYE:
                point = face_landmarks[idx]
                left_eye.append((int(point.x * w), int(point.y * h)))
            
            # Extract Right Eye
            right_eye = []
            for idx in self.RIGHT_EYE:
                point = face_landmarks[idx]
                right_eye.append((int(point.x * w), int(point.y * h)))
                
            return left_eye, right_eye
            
        except Exception as e:
            print(f"Landmark extraction error: {e}")
            return None, None

    def check_liveness(self, image: np.ndarray, timeout_frames: int = 90) -> Tuple[bool, str, float]:
        """
        Check liveness over multiple frames
        
        Args:
            image: Verification frame (full image)
            timeout_frames: Maximum frames to wait for blink
            
        Returns:
            Tuple of (is_live, status_message, confidence)
        """
        # Process image with MediaPipe
        if not self.is_available:
            return True, "Liveness check skipped (Lib missing)", 1.0

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        avg_ear = 0.0
        current_ear = 0.0
        
        if results.multi_face_landmarks:
            self.last_landmarks = results.multi_face_landmarks
            
            left_eye, right_eye = self.get_landmarks_from_mesh(image, results.multi_face_landmarks)
            
            if left_eye and right_eye:
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                
                current_ear = (left_ear + right_ear) / 2.0
                self.ear_history.append(current_ear)
                
                # Blink logic
                if current_ear < self.ear_threshold:
                    self.frame_counter += 1
                else:
                    if self.frame_counter >= self.blink_frames:
                        self.total_blinks += 1
                        self.blink_detected = True
                        print(f"BLINK DETECTED! (EAR: {current_ear:.3f})")
                    self.frame_counter = 0
            
            avg_ear = np.mean(list(self.ear_history)) if self.ear_history else current_ear
        else:
            # No face detected by MediaPipe
            avg_ear = 0.0
        
        # Calculate confidence
        confidence = 0.0
        if len(self.ear_history) > 10:
            ear_variance = np.var(list(self.ear_history))
            confidence = min(ear_variance * 500, 1.0)
            
        if self.blink_detected:
            return True, f"LIVE - Blink detected ({self.total_blinks})", confidence
        
        if len(self.ear_history) > timeout_frames:
            return False, f"FAILED - No blink in {timeout_frames} frames", confidence
            
        return False, f"Waiting for blink... (EAR: {avg_ear:.3f})", confidence

    def draw_status(self, image: np.ndarray) -> np.ndarray:
        """Draw liveness status and landmarks"""
        annotated = image.copy()
        
        if self.last_landmarks:
            left_eye, right_eye = self.get_landmarks_from_mesh(image, self.last_landmarks)
            
            if left_eye and right_eye:
                for point in left_eye + right_eye:
                    cv2.circle(annotated, point, 1, (0, 255, 255), -1)
        
        # Draw status text
        current_ear = self.ear_history[-1] if self.ear_history else 0.0
        
        cv2.putText(annotated, f"EAR: {current_ear:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(annotated, f"Blinks: {self.total_blinks}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        status_text = "LIVE" if self.blink_detected else "Checking..."
        color = (0, 255, 0) if self.blink_detected else (0, 165, 255)
        cv2.putText(annotated, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
        return annotated

    def reset(self):
        self.frame_counter = 0
        self.total_blinks = 0
        self.ear_history.clear()
        self.blink_detected = False
