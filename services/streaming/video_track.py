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

        self.logger.info(f"WIMZVideoTrack initialized at {fps} FPS, overlay={enable_overlay}")

    async def recv(self) -> VideoFrame:
        """Receive next video frame for WebRTC transmission"""
        pts, time_base = await self.next_timestamp()

        # Rate limiting - wait for next frame interval
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_interval:
            await asyncio.sleep(self.frame_interval - elapsed)
        self.last_frame_time = time.time()

        # Get frame from detector service
        frame = self.detector.get_last_frame()

        if frame is None:
            # Return black frame if no camera data
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame, "No Camera Feed",
                (200, 320), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
        else:
            # Frame is RGB from Picamera2, convert to BGR for OpenCV operations
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add AI overlays if enabled
            if self.enable_overlay:
                frame = self._add_overlays(frame)

        # Convert BGR to RGB for WebRTC
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                        dog_name = dog_data.get('name', dog_id)
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
