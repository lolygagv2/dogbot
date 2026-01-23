#!/usr/bin/env python3
"""
services/media/photo_capture.py - Photo capture with HUD overlay

Captures photos from the camera stream with optional HUD overlay showing:
- Dog bounding boxes with names
- Mission/training state
- Treat dispense indicator
"""

import os
import cv2
import base64
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# Photo storage configuration
PHOTO_DIR = Path("/home/morgan/dogbot/photos")
MAX_PHOTOS = 100
JPEG_QUALITY = 85


class PhotoCaptureService:
    """Photo capture service with HUD overlay"""

    def __init__(self):
        self.logger = logging.getLogger('PhotoCapture')
        self._lock = threading.Lock()

        # Treat indicator state (flashes briefly when treat dispensed)
        self._treat_indicator_until = 0

        # Ensure photo directory exists
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)

        self.logger.info("PhotoCaptureService initialized")

    def capture_photo(
        self,
        with_hud: bool = True,
        detector=None,
        ai_controller=None,
        mission_engine=None,
        mode_fsm=None
    ) -> Dict[str, Any]:
        """
        Capture a photo with optional HUD overlay.

        Args:
            with_hud: Whether to draw HUD overlay
            detector: DetectorService instance (for frame access)
            ai_controller: AIController instance (for dog detection data)
            mission_engine: MissionEngine instance (for mission state)
            mode_fsm: ModeFSM instance (for current mode)

        Returns:
            Dict with photo data, base64 encoded image, and metadata
        """
        try:
            # Get services if not provided
            if detector is None:
                from services.perception.detector import get_detector_service
                detector = get_detector_service()

            # Get frame from detector
            frame = detector.get_last_frame()
            if frame is None:
                return {
                    "success": False,
                    "error": "No camera frame available"
                }

            # Convert from RGB (Picamera2) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Apply HUD overlay if requested
            if with_hud:
                frame_bgr = self._draw_hud(
                    frame_bgr,
                    detector=detector,
                    ai_controller=ai_controller,
                    mission_engine=mission_engine,
                    mode_fsm=mode_fsm
                )

            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            success, jpeg_data = cv2.imencode('.jpg', frame_bgr, encode_params)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to encode JPEG"
                }

            # Base64 encode
            b64_data = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')

            # Generate filename and save locally
            timestamp = datetime.now()
            filename = f"wimz_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = PHOTO_DIR / filename

            # Save locally
            with open(filepath, 'wb') as f:
                f.write(jpeg_data.tobytes())

            # Cleanup old photos
            self._cleanup_old_photos()

            self.logger.info(f"Photo captured: {filename} (HUD={with_hud})")

            return {
                "success": True,
                "data": b64_data,
                "timestamp": timestamp.isoformat() + "Z",
                "filename": filename,
                "filepath": str(filepath),
                "with_hud": with_hud,
                "resolution": f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                "size_bytes": len(jpeg_data)
            }

        except Exception as e:
            self.logger.error(f"Photo capture error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _draw_hud(
        self,
        frame: np.ndarray,
        detector=None,
        ai_controller=None,
        mission_engine=None,
        mode_fsm=None
    ) -> np.ndarray:
        """Draw HUD overlay on frame"""
        try:
            height, width = frame.shape[:2]

            # Get AI controller if not provided
            if ai_controller is None and detector is not None:
                ai_controller = getattr(detector, 'ai', None)

            # 1. Draw dog bounding boxes and labels
            if ai_controller is not None:
                frame = self._draw_dog_boxes(frame, ai_controller)

            # 2. Draw mission state (bottom-left)
            mission_text = self._get_mission_state_text(mission_engine, mode_fsm)
            if mission_text:
                frame = self._draw_text_with_background(
                    frame, mission_text,
                    (10, height - 20),
                    font_scale=0.5,
                    color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    bg_alpha=0.5
                )

            # 3. Draw treat indicator (top-right) if recently dispensed
            if time.time() < self._treat_indicator_until:
                frame = self._draw_treat_indicator(frame)

            # 4. Timestamp (top-left, subtle)
            timestamp_text = datetime.now().strftime("%H:%M:%S")
            frame = self._draw_text_with_background(
                frame, timestamp_text,
                (10, 25),
                font_scale=0.4,
                color=(200, 200, 200),
                bg_color=(0, 0, 0),
                bg_alpha=0.3
            )

        except Exception as e:
            self.logger.debug(f"HUD overlay error: {e}")

        return frame

    def _draw_dog_boxes(self, frame: np.ndarray, ai_controller) -> np.ndarray:
        """Draw bounding boxes and labels for detected dogs"""
        try:
            # Get tracked dogs from dog_tracker
            if hasattr(ai_controller, 'dog_tracker') and ai_controller.dog_tracker:
                tracked = ai_controller.dog_tracker.get_tracked_dogs()

                for dog_id, dog_data in tracked.items():
                    bbox = dog_data.get('bbox')
                    if not bbox or len(bbox) < 4:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

                    # Draw thin white/light blue bounding box (2px)
                    box_color = (235, 206, 135)  # Light blue (BGR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Draw rounded corners (optional visual enhancement)
                    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                    self._draw_rounded_corners(frame, (x1, y1), (x2, y2), box_color, corner_len)

                    # Dog name label with pill background
                    dog_name = dog_data.get('name', 'Dog')
                    if dog_name:
                        # Capitalize first letter
                        display_name = dog_name.capitalize() if dog_name else "Dog"
                        frame = self._draw_pill_label(
                            frame, display_name,
                            (x1, y1 - 8),
                            font_scale=0.55,
                            color=(255, 255, 255),
                            bg_color=(80, 80, 80),
                            bg_alpha=0.7
                        )

                    # Behavior label if detected
                    behavior = dog_data.get('behavior', '')
                    confidence = dog_data.get('confidence', 0)
                    if behavior and confidence > 0.5:
                        behavior_text = f"{behavior}"
                        frame = self._draw_pill_label(
                            frame, behavior_text,
                            (x1, y2 + 20),
                            font_scale=0.45,
                            color=(0, 255, 100),
                            bg_color=(40, 40, 40),
                            bg_alpha=0.6
                        )

        except Exception as e:
            self.logger.debug(f"Draw dog boxes error: {e}")

        return frame

    def _draw_rounded_corners(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        corner_len: int
    ):
        """Draw rounded corner accents on bounding box"""
        x1, y1 = pt1
        x2, y2 = pt2
        thickness = 3

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def _draw_pill_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        bg_alpha: float = 0.6
    ) -> np.ndarray:
        """Draw text with pill-shaped semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        padding_x, padding_y = 8, 4

        # Background pill rectangle
        x1 = x - padding_x
        y1 = y - text_h - padding_y
        x2 = x + text_w + padding_x
        y2 = y + padding_y

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return frame

    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        bg_alpha: float = 0.5
    ) -> np.ndarray:
        """Draw text with semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        padding = 4

        # Background rectangle
        x1, y1 = x - padding, y - text_h - padding
        x2, y2 = x + text_w + padding, y + padding

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return frame

    def _get_mission_state_text(self, mission_engine=None, mode_fsm=None) -> Optional[str]:
        """Get mission/training state text for HUD"""
        try:
            # Get mission engine if not provided
            if mission_engine is None:
                from orchestrators.mission_engine import get_mission_engine
                mission_engine = get_mission_engine()

            # Get mode FSM if not provided
            if mode_fsm is None:
                from orchestrators.mode_fsm import get_mode_fsm
                mode_fsm = get_mode_fsm()

            # Get current mode
            current_mode = None
            if mode_fsm:
                from core.state import get_state
                state = get_state()
                current_mode = state.get_mode()

            # Get mission status
            if mission_engine:
                status = mission_engine.get_mission_status()

                if status.get('active'):
                    mission_name = status.get('mission_name', 'Mission')
                    stage_info = status.get('stage_info', {})
                    stage_name = stage_info.get('name', '')

                    if stage_name:
                        # Format: "Waiting for sit..." or "Quiet Training Active"
                        if 'wait' in stage_name.lower():
                            return f"Waiting for {stage_name.split('_')[-1]}..."
                        else:
                            return f"{stage_name}"
                    else:
                        return f"{mission_name} Active"

            # Fallback to mode-based text
            if current_mode:
                mode_text = {
                    'silent_guardian': "Quiet Training Active",
                    'coach': "Training Mode",
                    'manual': "Manual Control",
                    'idle': None  # Don't show for idle
                }
                mode_str = current_mode.value if hasattr(current_mode, 'value') else str(current_mode)
                return mode_text.get(mode_str.lower())

        except Exception as e:
            self.logger.debug(f"Get mission state error: {e}")

        return None

    def _draw_treat_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw treat indicator icon in top-right corner"""
        try:
            height, width = frame.shape[:2]

            # Position in top-right
            icon_x = width - 50
            icon_y = 30

            # Draw bone emoji text (fallback to text if emoji not supported)
            text = "TREAT!"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Pulsing effect based on time
            pulse = abs(int(time.time() * 10) % 10 - 5) / 5.0
            color = (0, int(200 + 55 * pulse), int(100 + 100 * pulse))  # Green pulse

            cv2.putText(frame, text, (icon_x - 40, icon_y), font, 0.6, color, 2, cv2.LINE_AA)

        except Exception as e:
            self.logger.debug(f"Draw treat indicator error: {e}")

        return frame

    def trigger_treat_indicator(self, duration: float = 2.0):
        """Trigger treat indicator to show for specified duration"""
        self._treat_indicator_until = time.time() + duration
        self.logger.debug(f"Treat indicator triggered for {duration}s")

    def _cleanup_old_photos(self):
        """Delete oldest photos if over MAX_PHOTOS limit"""
        try:
            photos = sorted(PHOTO_DIR.glob("wimz_*.jpg"), key=lambda p: p.stat().st_mtime)

            if len(photos) > MAX_PHOTOS:
                to_delete = photos[:len(photos) - MAX_PHOTOS]
                for photo in to_delete:
                    photo.unlink()
                    self.logger.debug(f"Deleted old photo: {photo.name}")

        except Exception as e:
            self.logger.warning(f"Photo cleanup error: {e}")

    def get_recent_photos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent photos"""
        try:
            photos = sorted(PHOTO_DIR.glob("wimz_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

            result = []
            for photo in photos[:limit]:
                stat = photo.stat()
                result.append({
                    "filename": photo.name,
                    "filepath": str(photo),
                    "size_bytes": stat.st_size,
                    "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            return result

        except Exception as e:
            self.logger.error(f"Get recent photos error: {e}")
            return []


# Global singleton instance
_photo_capture_service: Optional[PhotoCaptureService] = None


def get_photo_capture_service() -> PhotoCaptureService:
    """Get the global PhotoCaptureService instance"""
    global _photo_capture_service
    if _photo_capture_service is None:
        _photo_capture_service = PhotoCaptureService()
    return _photo_capture_service
