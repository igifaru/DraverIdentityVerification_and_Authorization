"""
Verification Result Handler
Handles visual formatting and display of verification results on video frames
"""
import cv2
import numpy as np
from typing import Dict, Tuple
from utils.config import config
from utils.constants import (
    COLOR_AUTHORIZED, 
    COLOR_UNAUTHORIZED, 
    COLOR_WARNING, 
    FONT_SCALE_HEADER, 
    FONT_THICKNESS
)

class VerificationResultHandler:
    """Manages visual presentation of verification results"""
    
    @staticmethod
    def draw_result(frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw verification result overlay on frame
        
        Args:
            frame: Input video frame
            result: Verification result dictionary
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Determine status color and text
        if result['authorized']:
            status_color = COLOR_AUTHORIZED
            status_text = "AUTHORIZED"
        elif "No face" in result.get('status_message', ''):
            status_color = (128, 128, 128) # Gray
            status_text = "SCANNING..."
        elif "LOW LIGHT" in result.get('status_message', ''):
            status_color = COLOR_WARNING
            status_text = "LOW LIGHT"
        else:
            status_color = COLOR_UNAUTHORIZED
            status_text = "UNAUTHORIZED"
            
        # Draw status banner
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), status_color, -1)
        
        # Draw main status text
        cv2.putText(
            annotated, 
            status_text, 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            FONT_SCALE_HEADER, 
            (255, 255, 255), 
            3
        )
        
        # Draw secondary message
        if result.get('status_message'):
            cv2.putText(
                annotated, 
                result['status_message'], 
                (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
        
        return annotated

    @staticmethod
    def get_driver_status(result: Dict, is_running: bool) -> Dict:
        """
        Return simplified verification status for the driver terminal.

        States returned:
          scanning     – engine running, no face or still processing
          authorized   – driver recognised (silent on terminal)
          unauthorized – access denied  (terminal shows alert overlay)
          warning      – low-light or liveness failure
          ready        – engine not yet started
        """
        if not is_running:
            return {
                'state':       'ready',
                'status_display': 'SYSTEM READY',
                'instruction': 'Initializing camera...',
                'event_id':    None,
                'meta':        '',
            }

        state         = 'scanning'
        status_display = 'SCANNING'
        instruction   = 'Please look at the camera'
        event_id      = None
        meta          = ''

        if result:
            event_id = result.get('log_id')          # unique per verification
            msg      = result.get('status_message', '').upper()

            if 'LOW LIGHT' in msg:
                state         = 'warning'
                status_display = 'LOW LIGHT'
                instruction   = 'Please improve lighting'
            elif 'ENROLL' in msg:
                state         = 'warning'
                status_display = 'SETUP REQUIRED'
                instruction   = 'Contact authority'
            elif result.get('authorized'):
                state         = 'authorized'
                status_display = 'ACCESS GRANTED'
                instruction   = 'Driver verified successfully'
            elif not result.get('liveness_passed') and result.get('similarity_score', 0) > 0:
                state         = 'unauthorized'
                status_display = 'ACCESS DENIED'
                instruction   = 'Liveness check failed'
                score         = result.get('similarity_score', 0)
                meta          = f'Similarity: {score:.1%}'
            elif result.get('similarity_score', 0) > 0:
                state         = 'unauthorized'
                status_display = 'ACCESS DENIED'
                instruction   = 'Unauthorized driver detected'
                score         = result.get('similarity_score', 0)
                meta          = f'Similarity: {score:.1%}'

        return {
            'state':          state,
            'status_display': status_display,
            'instruction':    instruction,
            'event_id':       event_id,
            'meta':           meta,
        }
