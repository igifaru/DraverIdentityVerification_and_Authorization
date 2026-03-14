"""
Verification Result Handler
Handles visual formatting and display of verification results on video frames
"""
import cv2
import numpy as np
from typing import Dict, Tuple
from utils.config import config
# --- Visual Constants ---
COLOR_AUTHORIZED = (0, 255, 0)      # Green
COLOR_UNAUTHORIZED = (0, 0, 255)    # Red
COLOR_WARNING = (0, 165, 255)       # Orange
FONT_SCALE_HEADER = 1.2
FONT_THICKNESS = 2

class VerificationResultHandler:
    """Manages visual presentation of verification results"""
    
    @staticmethod
    def draw_result(frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw verification result overlay on frame.
        NOTE: We now skip drawing heavy banners/text here because the 
        web UI handles status display with better aesthetics and accessibility.
        """
        annotated = frame.copy()
        
        # We only draw a subtle status indicator if debugging or specifically requested.
        # Otherwise, the clean frame is preferred for the web view to avoid mirroring issues.
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
