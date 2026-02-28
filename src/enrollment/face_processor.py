"""
Face Processor Module
Handles face detection, preprocessing, and quality validation using MTCNN.

Robustness features:
  - CLAHE enhancement for low-light frames and glasses contrast
  - Auto-retry detection with enhanced image if raw frame fails
  - Relaxed aspect ratio validation for drivers wearing glasses
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from typing import Optional, Tuple, List
from utils.config import config


class FaceProcessor:
    """Processes facial images for enrollment and verification"""

    def __init__(self):
        """Initialize face processor with MTCNN detector"""
        print("Initializing MTCNN face detector...")
        self.detector = MTCNN()
        self.target_size = (160, 160)  # FaceNet input size
        print("Face processor initialized")


    # ------------------------------------------------------------------
    # Image enhancement
    # ------------------------------------------------------------------

    @staticmethod
    def enhance_frame(image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
        Lightness channel (LAB colorspace) to produce a perceptually balanced
        image that helps MTCNN under two conditions:

          Low light  : boosts perceived brightness without blowing out highlights
          Glasses    : improves local contrast around eye / frame regions so
                       MTCNN landmarks are more reliably located

        Args:
            image: BGR uint8 image from the camera.

        Returns:
            Enhanced BGR uint8 image (same shape).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Adaptive equalisation — clip_limit=2.0, tile 8×8 is a good balance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)

        enhanced_lab = cv2.merge([l_eq, a_ch, b_ch])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    @staticmethod
    def _brightness(image: np.ndarray) -> float:
        """Return mean luminance of a BGR image (0–255)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    # Brightness thresholds
    # Below NO_SIGNAL  : completely dark / camera blocked  → reject, don't enhance
    # Below LOW_LIGHT  : dim but workable                  → enhance, then detect
    # Above LOW_LIGHT  : normal conditions                 → detect raw, enhance only as fallback
    _BRIGHTNESS_NO_SIGNAL  = 8     # CLAHE cannot manufacture data that does not exist
    _BRIGHTNESS_LOW_LIGHT  = 40    # usable dim light — CLAHE gives meaningful boost

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_face(
        self,
        image: np.ndarray,
        min_confidence: float = None,
        pre_enhanced: bool = False,
    ) -> Optional[dict]:
        """
        Detect the highest-confidence face in *image* using MTCNN.

        3-tier strategy based on image brightness:

          Tier 1  (brightness < 8)   : No-signal — reject immediately.
                                        CLAHE cannot manufacture pixel data
                                        from a completely black frame.
          Tier 2  (8 ≤ brightness < 40): Low-light — enhance first, then detect.
                                        CLAHE gives a meaningful boost here.
          Tier 3  (brightness ≥ 40)  : Normal — try raw first, fall back to
                                        CLAHE-enhanced only if raw fails.

        Args:
            image:          BGR uint8 camera frame.
            min_confidence: Minimum MTCNN confidence (falls back to config).
            pre_enhanced:   If True, treat *image* as already enhanced
                            (skip redundant CLAHE pass).

        Returns:
            Detection dict with 'box', 'confidence', 'keypoints', or None.
        """
        min_confidence = min_confidence or config.get('verification.confidence_threshold', 0.95)

        def _run_mtcnn(img: np.ndarray) -> Optional[dict]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb)
            if not detections:
                return None
            best = max(detections, key=lambda x: x['confidence'])
            return best if best['confidence'] >= min_confidence else None

        brightness = self._brightness(image)

        # ——— Tier 1: completely dark / no signal ———
        if brightness < self._BRIGHTNESS_NO_SIGNAL:
            print(f"  MTCNN: skipped — no signal (brightness={brightness:.1f} < {self._BRIGHTNESS_NO_SIGNAL})")
            return None

        # ——— Tier 2: dim but workable — enhance first ———
        if brightness < self._BRIGHTNESS_LOW_LIGHT:
            img_to_use = image if pre_enhanced else self.enhance_frame(image)
            result = _run_mtcnn(img_to_use)
            if result is not None:
                print(f"  MTCNN: found face (enhanced, dim)  conf={result['confidence']:.3f}  "
                      f"brightness={brightness:.1f}")
            else:
                print(f"  MTCNN: no face  brightness={brightness:.1f} (dim, enhanced)")
            return result

        # ——— Tier 3: normal — try raw, fallback to enhanced only if needed ———
        result = _run_mtcnn(image)
        if result is not None:
            print(f"  MTCNN: found face (raw)  conf={result['confidence']:.3f}")
            return result

        # Fallback: try enhanced (glasses / partial occlusion)
        if not pre_enhanced:
            enhanced = self.enhance_frame(image)
            result = _run_mtcnn(enhanced)
            if result is not None:
                print(f"  MTCNN: found face (enhanced fallback)  conf={result['confidence']:.3f}")
                return result

        print(f"  MTCNN: no face detected  brightness={brightness:.1f}  threshold={min_confidence}")
        return None


    def extract_face(self, image: np.ndarray, detection: dict, margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face region from image with margin
        
        Args:
            image: Input image
            detection: Detection dictionary from detect_face
            margin: Margin around face box (0.2 = 20% on each side)
            
        Returns:
            Extracted face image or None
        """
        x, y, width, height = detection['box']
        
        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + width + margin_x)
        y2 = min(image.shape[0], y + height + margin_y)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        return face
    
    def align_face(self, image: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """
        Align face using eye landmarks
        
        Args:
            image: Input image
            detection: Detection dictionary with keypoints
            
        Returns:
            Aligned face image or None
        """
        keypoints = detection['keypoints']
        
        # Get eye coordinates
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center point between eyes (ensure integers for cv2)
        eye_center = (int((left_eye[0] + right_eye[0]) / 2), 
                      int((left_eye[1] + right_eye[1]) / 2))
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for FaceNet
        
        Args:
            face: Face image
            
        Returns:
            Preprocessed face (160x160, normalized)
        """
        # Resize to target size
        face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [-1, 1]
        face_normalized = (face_resized.astype('float32') - 127.5) / 128.0
        
        return face_normalized
    
    def validate_face_quality(self, image: np.ndarray, detection: dict) -> Tuple[bool, str]:
        """
        Validate face quality for enrollment.
        Returns (is_valid, reason).

        Tolerances are deliberately relaxed around:
          - Aspect ratio: glasses frames can make the face appear wider
          - Centering:    \u00b140% offset (was \u00b130%)
          - Face area:    3% minimum (was 5%)
        """
        # Check confidence
        if detection['confidence'] < 0.75:
            return False, f"Low confidence: {detection['confidence']:.2f}"

        x, y, width, height = detection['box']
        face_area  = width * height
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area

        if face_ratio < 0.03:
            return False, "Face too small \u2014 move closer to the camera"
        if face_ratio > 0.85:
            return False, "Face too close to camera \u2014 move back slightly"
        if width < 60 or height < 60:
            return False, "Face resolution too low \u2014 move closer"

        face_center_x  = x + width  / 2
        face_center_y  = y + height / 2
        offset_x = abs(face_center_x - image.shape[1] / 2) / image.shape[1]
        offset_y = abs(face_center_y - image.shape[0] / 2) / image.shape[0]
        if offset_x > 0.4 or offset_y > 0.4:
            return False, "Face not centered \u2014 move to the middle of the frame"

        # Relaxed aspect ratio: glasses frames can widen apparent face width
        aspect_ratio = width / height
        if aspect_ratio < 0.55 or aspect_ratio > 1.50:
            return False, "Unusual face angle \u2014 face the camera directly"

        return True, "OK"


    
    def process_for_enrollment(self, image: np.ndarray, min_confidence: float = 0.70):
        """
        Complete preprocessing pipeline for face detection and crop extraction.

        Brightness decision tree (applied before any MTCNN call):

          brightness < 8   : Completely dark / camera blocked / lens covered.
                             CLAHE cannot manufacture data that does not exist.
                             Rejected immediately — no MTCNN inference wasted.

          8 ≤ brightness < 40 : Dim but workable (dark room, dusk, shaded).
                             CLAHE is applied first for maximum boost, then
                             detection runs on the enhanced image.

          brightness ≥ 40  : Normal light.
                             Raw detect first (fastest path). CLAHE applied
                             only as a fallback (e.g. for glasses occlusion).

        Used by both the enrollment flow (min_confidence=0.70) and the live
        verification loop (min_confidence=0.80).

        Returns:
            (face_crop, status_message)
            face_crop: uint8 BGR crop ready for generate_embedding(), or None.
        """
        brightness = self._brightness(image)

        # —— Tier 1: completely dark / no signal ——
        if brightness < self._BRIGHTNESS_NO_SIGNAL:
            reason = (
                f"No signal — frame is too dark (brightness={brightness:.1f}). "
                f"Camera may be blocked, lens covered, or no light source present."
            )
            print(f"  process_for_enrollment: {reason}")
            return None, reason

        # —— Tier 2: dim but workable — enhance first ——
        if brightness < self._BRIGHTNESS_LOW_LIGHT:
            working_image = self.enhance_frame(image)
            pre_enhanced  = True
        # —— Tier 3: normal light — use raw, CLAHE is fallback inside detect_face ——
        else:
            working_image = image
            pre_enhanced  = False

        detection = self.detect_face(
            working_image,
            min_confidence=min_confidence,
            pre_enhanced=pre_enhanced,
        )

        if detection is None:
            return None, "No face detected"

        is_valid, reason = self.validate_face_quality(working_image, detection)
        if not is_valid:
            return None, f"Quality check failed: {reason}"

        aligned   = self.align_face(working_image, detection)
        face_crop = self.extract_face(aligned, detection)

        if face_crop is None:
            return None, "Failed to extract face"

        return face_crop, "OK"


    def generate_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a L2-normalised FaceNet-128 embedding from a raw face crop.

        Args:
            face_crop: Raw BGR uint8 face image (any size — will be resized
                        internally by DeepFace to 160x160).

        Returns:
            128-dimensional L2-normalised embedding as float32, or None on failure.
        """
        try:
            from deepface import DeepFace

            # Ensure the image is uint8 in RGB colour space
            if face_crop.dtype != np.uint8:
                # Convert float [0,1] or [-1,1] to uint8 [0,255]
                face_uint8 = np.clip(
                    (face_crop * 127.5 + 127.5) if face_crop.min() < 0
                    else (face_crop * 255),
                    0, 255
                ).astype(np.uint8)
            else:
                face_uint8 = face_crop

            # Convert BGR → RGB (DeepFace expects RGB)
            face_rgb = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2RGB)

            # Resize to FaceNet input size
            face_resized = cv2.resize(face_rgb, self.target_size, interpolation=cv2.INTER_AREA)

            print(f"  Generating embedding: shape={face_resized.shape}, "
                  f"dtype={face_resized.dtype}, "
                  f"range=[{face_resized.min()},{face_resized.max()}]")

            result = DeepFace.represent(
                img_path=face_resized,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip',
            )

            raw = np.array(
                result[0]['embedding'] if isinstance(result, list) else result['embedding'],
                dtype=np.float32
            )

            # L2-normalise so that cosine similarity == dot product
            norm = np.linalg.norm(raw)
            if norm < 1e-8:
                print("  ERROR: Near-zero norm embedding — face may be blank")
                return None
            embedding = raw / norm

            print(f"  Embedding OK: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            return embedding

        except Exception as exc:
            import traceback
            print(f"ERROR: Failed to generate embedding: {exc}")
            print(traceback.format_exc())
            return None


    def draw_detection(self, image: np.ndarray, detection: dict) -> np.ndarray:
        """
        Draw detection box and landmarks on image
        
        Args:
            image: Input image
            detection: Detection dictionary
            
        Returns:
            Image with annotations
        """
        annotated = image.copy()
        
        # Draw bounding box
        x, y, width, height = detection['box']
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Draw confidence
        confidence_text = f"{detection['confidence']:.2f}"
        cv2.putText(annotated, confidence_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw keypoints
        keypoints = detection['keypoints']
        for name, point in keypoints.items():
            cv2.circle(annotated, point, 3, (0, 0, 255), -1)
        
        return annotated


if __name__ == "__main__":
    # Test face processor
    print("Testing face processor...")
    
    processor = FaceProcessor()
    print("Face processor ready")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("Press SPACE to test detection, ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and draw
            detection = processor.detect_face(frame)
            
            if detection:
                annotated = processor.draw_detection(frame, detection)
                is_valid, reason = processor.validate_face_quality(frame, detection)
                
                status_text = f"Valid: {is_valid} - {reason}"
                cv2.putText(annotated, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Face Detection Test", annotated)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Face Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
