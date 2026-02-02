#!/usr/bin/env python3
"""
Custom VideoStreamTrack for WIM-Z WebRTC streaming
Reads frames from DetectorService and converts to WebRTC format
"""

import asyncio
import cv2
import numpy as np
import time
import logging
from typing import Optional, TYPE_CHECKING

from aiortc import VideoStreamTrack
from av import VideoFrame

if TYPE_CHECKING:
    from services.perception.detector import DetectorService

# Skeleton connections for pose visualization (matching video_recorder.py)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Front legs
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Back legs
    (5, 11), (6, 12),  # Body spine
]


class WIMZVideoTrack(VideoStreamTrack):
    """
    Custom video track that captures frames from DetectorService

    Features:
    - Reads frames via get_last_frame() (thread-safe)
    - Optionally adds AI detection overlays
    - Targets specified FPS with frame timing
    - Converts to VideoFrame for aiortc
    """

    kind = "video"

    def __init__(
        self,
        detector: 'DetectorService',
        fps: int = 15,
        enable_overlay: bool = True
    ):
        super().__init__()
        self.logger = logging.getLogger('WIMZVideoTrack')
        self.detector = detector
        self.target_fps = fps
        self.enable_overlay = enable_overlay

        # Frame timing
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Track state
        self._running = True
        self._paused = False  # Pause during camera reconfig to prevent stale frame reads

        self.logger.info(f"WIMZVideoTrack initialized at {fps} FPS, overlay={enable_overlay}")

    # BUILD 36: Maximum frame age before considering it stale (500ms)
    MAX_FRAME_AGE_SEC = 0.5

    async def recv(self) -> VideoFrame:
        """Receive next video frame for WebRTC transmission"""
        pts, time_base = await self.next_timestamp()

        # Rate limiting - wait for next frame interval
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_interval:
            await asyncio.sleep(self.frame_interval - elapsed)
        self.last_frame_time = time.time()

        # BUILD 36: Get frame WITH timestamp to check freshness
        # This fixes Issue 7 from Build 35 - video lag of 10-30 seconds
        frame = None
        frame_age = float('inf')
        if not self._paused:
            frame, frame_timestamp = self.detector.get_last_frame_with_timestamp()
            if frame is not None and frame_timestamp > 0:
                frame_age = time.time() - frame_timestamp
                # Skip stale frames to prevent video lag
                if frame_age > self.MAX_FRAME_AGE_SEC:
                    if self.frame_count % 50 == 0:  # Log periodically
                        self.logger.warning(f"📹 Skipping stale frame (age={frame_age:.2f}s > {self.MAX_FRAME_AGE_SEC}s)")
                    frame = None  # Force black frame with "Buffering" message

        # Log frame status periodically (every 100 frames)
        if self.frame_count % 100 == 0:
            if frame is not None:
                self.logger.info(f"📹 Video frame {self.frame_count}: {frame.shape}, age={frame_age:.3f}s")
            else:
                self.logger.warning(f"📹 Video frame {self.frame_count}: None (no camera feed or stale)")

        if frame is None:
            # Return black frame if no camera data - use detector's current resolution
            resolution = getattr(self.detector, '_current_resolution', (640, 640))
            height, width = resolution[1], resolution[0]  # resolution is (width, height)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # BUILD 36: Show different message for stale frames vs no camera
            if frame_age < float('inf') and frame_age > self.MAX_FRAME_AGE_SEC:
                # Frame exists but is stale - show buffering
                cv2.putText(
                    frame, "Buffering...",
                    (width // 3, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2  # Yellow for buffering
                )
            else:
                cv2.putText(
                    frame, "No Camera Feed",
                    (width // 3, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2
                )
            # BUILD 36 Fix 1: Add status overlay to black frames so users see current mode
            # This fixes Issue 4 - no status shown during camera reconfig
            self._add_status_overlay(frame)
            frame_rgb = frame  # Already RGB (black frame)
        else:
            # Picamera2 RGB888 outputs BGR despite the name (confirmed by testing)
            # Frame is already BGR - perfect for OpenCV operations
            frame_bgr = frame  # No conversion needed - already BGR

            # Add AI overlays if enabled (OpenCV expects BGR - we have BGR)
            if self.enable_overlay:
                frame_bgr = self._add_overlays(frame_bgr)

            # Convert BGR to RGB for WebRTC/aiortc
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Create VideoFrame for aiortc
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        self.frame_count += 1

        return video_frame

    def _add_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Add AI detection overlays to frame (bounding boxes, poses, behaviors)"""
        try:
            if not hasattr(self.detector, 'ai') or self.detector.ai is None:
                return frame

            ai = self.detector.ai

            # Get tracked dogs from dog tracker
            if hasattr(ai, 'dog_tracker') and ai.dog_tracker:
                tracked = ai.dog_tracker.get_tracked_dogs()

                for dog_id, dog_data in tracked.items():
                    # Draw bounding box
                    bbox = dog_data.get('bbox')
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

                        # Green box for detected dog
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Dog name/ID label
                        # BUILD 36: Show "Dog" when ArUco identification unavailable
                        # Issue 3 from Build 35 - no label shown when name is None
                        dog_name = dog_data.get('name') or 'Dog'
                        cv2.putText(
                            frame, str(dog_name),
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2
                        )

                    # Draw behavior label
                    behavior = dog_data.get('behavior', '')
                    confidence = dog_data.get('confidence', 0)
                    if behavior and confidence > 0.5 and bbox:
                        label = f"{behavior} ({confidence:.0%})"
                        cv2.putText(
                            frame, label,
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 2
                        )

                    # Draw pose keypoints if available
                    keypoints = dog_data.get('keypoints', [])
                    if keypoints:
                        self._draw_pose(frame, keypoints)

            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                frame, timestamp,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            # Add FPS counter
            if self.frame_count > 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    frame, f"WebRTC: {fps:.1f} FPS",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

            # Add mode and status overlay (large, visible text at top)
            self._add_status_overlay(frame)

        except Exception as e:
            self.logger.debug(f"Overlay error (continuing): {e}")

        return frame

    def _draw_pose(self, frame: np.ndarray, keypoints: list):
        """Draw pose skeleton on frame"""
        if not keypoints or len(keypoints) < 17:
            return

        # Draw keypoint circles
        for i, kp in enumerate(keypoints[:17]):
            if len(kp) >= 3 and kp[2] > 0.3:  # confidence threshold
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 4, (0, 165, 255), -1)  # Orange

        # Draw skeleton connections
        for (i, j) in SKELETON_CONNECTIONS:
            if i < len(keypoints) and j < len(keypoints):
                kp1, kp2 = keypoints[i], keypoints[j]
                if len(kp1) >= 3 and len(kp2) >= 3:
                    if kp1[2] > 0.3 and kp2[2] > 0.3:
                        x1, y1 = int(kp1[0]), int(kp1[1])
                        x2, y2 = int(kp2[0]), int(kp2[1])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow

    def _add_status_overlay(self, frame: np.ndarray):
        """Add mode and status text overlay (large, visible text at top)

        BUILD 34: Removed emoji characters - OpenCV FONT_HERSHEY_SIMPLEX doesn't
        support Unicode emoji, causing ???? display. Using plain text indicators.
        """
        try:
            from core.state import get_state, SystemMode

            state = get_state()
            mode = state.get_mode()
            mode_name = mode.value if hasattr(mode, 'value') else str(mode)

            # Status text based on mode (NO EMOJI - causes ???? in OpenCV fonts)
            status_text = ""
            status_color = (255, 255, 255)  # Default white

            if mode == SystemMode.COACH:
                try:
                    from orchestrators.coaching_engine import get_coaching_engine
                    coach = get_coaching_engine()
                    if coach and hasattr(coach, 'fsm_state'):
                        fsm = coach.fsm_state.value if hasattr(coach.fsm_state, 'value') else str(coach.fsm_state)

                        if 'waiting_for_dog' in fsm:
                            status_text = "[COACH] Waiting for dog..."
                            status_color = (255, 255, 0)  # Yellow
                        elif 'watching' in fsm:
                            trick = coach.current_session.trick_requested if coach.current_session else "trick"
                            status_text = f"[COACH] Watching for {trick.upper()}"
                            status_color = (0, 255, 255)  # Cyan
                        elif 'greeting' in fsm or 'command' in fsm:
                            status_text = "[COACH] Commanding..."
                            status_color = (255, 165, 0)  # Orange
                        elif 'success' in fsm:
                            status_text = "[COACH] SUCCESS!"
                            status_color = (0, 255, 0)  # Green
                        elif 'failure' in fsm or 'retry' in fsm:
                            status_text = "[COACH] Retry..."
                            status_color = (255, 128, 0)  # Orange
                        else:
                            status_text = f"[COACH] {fsm}"
                except Exception:
                    status_text = "[COACH MODE]"

            elif mode == SystemMode.MISSION:
                try:
                    from orchestrators.mission_engine import get_mission_engine
                    engine = get_mission_engine()
                    # BUILD 38: Use thread-safe get_mission_status() instead of direct active_session access
                    # This fixes the race condition that caused overlay to show "IDLE" during active missions
                    status = engine.get_mission_status()
                    if status.get("active"):
                        fsm = status.get("state", "unknown")
                        stage_info = status.get("stage_info") or {}
                        trick = stage_info.get("name", "behavior")
                        stage_num = status.get("current_stage", 0) + 1
                        total = status.get("total_stages", 1)

                        if 'waiting_for_dog' in fsm:
                            status_text = f"[MISSION {stage_num}/{total}] Waiting for dog..."
                            status_color = (255, 255, 0)  # Yellow
                        elif 'watching' in fsm:
                            status_text = f"[MISSION {stage_num}/{total}] Watching for {trick.upper()}"
                            status_color = (0, 255, 255)  # Cyan
                        elif 'greeting' in fsm or 'command' in fsm:
                            status_text = f"[MISSION {stage_num}/{total}] Commanding {trick}..."
                            status_color = (255, 165, 0)  # Orange
                        elif 'success' in fsm:
                            status_text = f"[MISSION {stage_num}/{total}] SUCCESS!"
                            status_color = (0, 255, 0)  # Green
                        elif 'failure' in fsm or 'retry' in fsm:
                            status_text = f"[MISSION {stage_num}/{total}] Retry {trick}..."
                            status_color = (255, 128, 0)  # Orange
                        elif 'idle' in fsm or 'starting' in fsm:
                            # BUILD 38: Handle idle/starting states that shouldn't normally appear
                            status_text = f"[MISSION {stage_num}/{total}] Initializing..."
                            status_color = (255, 255, 0)  # Yellow
                        else:
                            status_text = f"[MISSION {stage_num}/{total}] {fsm}"
                    else:
                        status_text = "[MISSION MODE] Starting..."
                        status_color = (255, 255, 0)  # Yellow while initializing
                except Exception:
                    status_text = "[MISSION MODE]"

            elif mode == SystemMode.SILENT_GUARDIAN:
                status_text = "[SILENT GUARDIAN] Monitoring..."
                status_color = (128, 0, 255)  # Purple

            elif mode == SystemMode.MANUAL:
                status_text = "[MANUAL MODE]"
                status_color = (0, 165, 255)  # Orange

            elif mode == SystemMode.IDLE:
                status_text = "[IDLE]"
                status_color = (128, 128, 128)  # Gray

            # Draw status text at top of frame (large, with background)
            if status_text:
                frame_height, frame_width = frame.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0  # Large text
                thickness = 2

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    status_text, font, font_scale, thickness
                )

                # Position at top center
                x = (frame_width - text_width) // 2
                y = text_height + 20  # 20px from top

                # Draw semi-transparent background
                bg_x1 = x - 10
                bg_y1 = y - text_height - 5
                bg_x2 = x + text_width + 10
                bg_y2 = y + baseline + 5

                # Create overlay for semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Draw text
                cv2.putText(
                    frame, status_text,
                    (x, y), font, font_scale,
                    status_color, thickness, cv2.LINE_AA
                )

            # BUILD 36 Fix 2: Add frame generation timestamp at bottom
            # This helps users identify stale/delayed video from WebRTC buffering
            frame_height, frame_width = frame.shape[:2]
            frame_time = time.strftime("%H:%M:%S")
            cv2.putText(
                frame, f"Frame: {frame_time}",
                (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

        except Exception as e:
            self.logger.debug(f"Status overlay error: {e}")

    def pause(self):
        """Pause video track during camera reconfiguration"""
        self._paused = True
        self.logger.info("WIMZVideoTrack paused (camera reconfig)")

    def resume(self):
        """Resume video track after camera reconfiguration"""
        self._paused = False
        self.logger.info("WIMZVideoTrack resumed")

    def stop(self):
        """Stop the video track"""
        self._running = False
        self.logger.info(f"WIMZVideoTrack stopped after {self.frame_count} frames")

    def get_stats(self) -> dict:
        """Get video track statistics"""
        elapsed = time.time() - self.start_time
        return {
            'frame_count': self.frame_count,
            'elapsed_seconds': elapsed,
            'actual_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'target_fps': self.target_fps,
            'overlay_enabled': self.enable_overlay
        }
