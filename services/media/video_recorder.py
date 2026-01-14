#!/usr/bin/env python3
"""
Video Recorder Service for WIM-Z
Records MP4 video at 640x640 with AI overlays (bounding boxes, poses, behaviors)
On-demand recording via API or Xbox controller
"""

import cv2
import time
import threading
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.bus import get_bus
from core.state import get_state

# Singleton instance
_video_recorder = None
_recorder_lock = threading.Lock()


# Skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    # Head connections
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose, eyes, ears
    # Body connections
    (5, 6),  # front shoulders
    (5, 7), (7, 9),  # front left leg
    (6, 8), (8, 10),  # front right leg
    (11, 12),  # back hips
    (11, 13), (13, 15),  # back left leg
    (12, 14), (14, 16),  # back right leg
    (5, 11), (6, 12),  # body spine
]


class VideoRecorder:
    """
    On-demand MP4 video recording with AI detection overlays

    Features:
    - 640x640 resolution at 15 FPS
    - Bounding boxes around detected dogs
    - Pose keypoints overlay
    - Behavior labels
    - ArUco ID display
    - Start/stop via API or Xbox controller
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.logger = logging.getLogger('VideoRecorder')

        # Recording state
        self.recording = False
        self.video_writer = None
        self.current_filename = None
        self.recording_start_time = None
        self.frame_count = 0

        # Output configuration
        self.output_dir = Path("/home/morgan/dogbot/recordings")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resolution = (640, 640)
        self.fps = 15.0

        # Recording thread
        self._recording_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Detection data (updated via event subscription)
        self._last_detections = []
        self._last_poses = []
        self._last_behaviors = []
        self._last_dog_assignments = {}
        self._last_aruco_markers = []  # ArUco marker positions

        # Behavior persistence - keep behavior text visible for longer
        self._behavior_display_duration = 3.0  # seconds to keep behavior visible
        self._behavior_display_buffer = {}  # {dog_id: {'behavior': str, 'confidence': float, 'timestamp': float}}

        # Subscribe to vision events
        self.bus.subscribe('vision', self._on_vision_event)

        self.logger.info("VideoRecorder initialized")

    def _on_vision_event(self, event):
        """Handle vision events to capture detection data"""
        try:
            now = time.time()

            if event.subtype == 'dog_detected':
                data = event.data
                with self._lock:
                    # Handle individual detection events (accumulate into list)
                    bbox = data.get('bbox', [])
                    if bbox and len(bbox) >= 4:
                        detection = {
                            'bbox': bbox,
                            'confidence': data.get('confidence', 0),
                            'dog_id': data.get('dog_id'),
                            'dog_name': data.get('dog_name'),
                            'timestamp': now
                        }
                        # Replace or add to detections (use dog_id as key to avoid duplicates)
                        dog_id = data.get('dog_id', 'unknown')
                        # Keep only recent detections (within 0.5s)
                        self._last_detections = [d for d in self._last_detections
                                                  if now - d.get('timestamp', 0) < 0.5 and d.get('dog_id') != dog_id]
                        self._last_detections.append(detection)

                    # Also handle batch format if available
                    if 'detections' in data:
                        self._last_detections = data.get('detections', [])
                    if 'poses' in data:
                        self._last_poses = data.get('poses', [])
                    if 'dog_assignments' in data:
                        self._last_dog_assignments = data.get('dog_assignments', {})

            elif event.subtype == 'behavior_detected':
                data = event.data
                with self._lock:
                    # Persist behavior display for longer visibility
                    dog_id = data.get('dog_id', 'dog_0')
                    behavior = data.get('behavior', '')
                    confidence = data.get('confidence', 0)

                    if behavior and confidence > 0.5:
                        self._behavior_display_buffer[dog_id] = {
                            'behavior': behavior,
                            'confidence': confidence,
                            'timestamp': now
                        }

            elif event.subtype == 'pose_detected':
                data = event.data
                with self._lock:
                    keypoints = data.get('keypoints', [])
                    if keypoints:
                        pose = {
                            'keypoints': keypoints,
                            'dog_id': data.get('dog_id'),
                            'timestamp': now
                        }
                        # Keep only recent poses
                        dog_id = data.get('dog_id', 'unknown')
                        self._last_poses = [p for p in self._last_poses
                                            if now - p.get('timestamp', 0) < 0.5 and p.get('dog_id') != dog_id]
                        self._last_poses.append(pose)

            elif event.subtype == 'aruco_detected':
                # Handle ArUco marker events if published
                data = event.data
                with self._lock:
                    self._last_aruco_markers = data.get('markers', [])

        except Exception as e:
            self.logger.error(f"Vision event error: {e}")

    def start_recording(self, filename_prefix: str = "recording") -> Dict[str, Any]:
        """
        Start video recording

        Args:
            filename_prefix: Prefix for the output filename

        Returns:
            Dict with filename and status
        """
        with self._lock:
            if self.recording:
                return {
                    "success": False,
                    "error": "Already recording",
                    "filename": self.current_filename
                }

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = f"{filename_prefix}_{timestamp}.mp4"
            output_path = self.output_dir / self.current_filename

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                self.resolution
            )

            if not self.video_writer.isOpened():
                self.logger.error("Failed to open video writer")
                return {
                    "success": False,
                    "error": "Failed to open video writer"
                }

            self.recording = True
            self.recording_start_time = time.time()
            self.frame_count = 0
            self._stop_event.clear()

            # Start recording thread
            self._recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True,
                name="VideoRecorder"
            )
            self._recording_thread.start()

            self.logger.info(f"Started recording: {self.current_filename}")

            return {
                "success": True,
                "filename": self.current_filename,
                "path": str(output_path)
            }

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop video recording and save file

        Returns:
            Dict with recording statistics
        """
        with self._lock:
            if not self.recording:
                return {
                    "success": False,
                    "error": "Not recording"
                }

            self.recording = False
            self._stop_event.set()

        # Wait for recording thread to finish
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)

        # Finalize video
        duration = 0.0
        if self.video_writer:
            self.video_writer.release()
            duration = time.time() - self.recording_start_time
            self.video_writer = None

        filename = self.current_filename
        frames = self.frame_count
        self.current_filename = None
        self.frame_count = 0

        self.logger.info(f"Stopped recording: {filename} ({frames} frames, {duration:.1f}s)")

        return {
            "success": True,
            "filename": filename,
            "frames": frames,
            "duration": duration,
            "fps_actual": frames / duration if duration > 0 else 0
        }

    def _recording_loop(self):
        """Main recording loop - captures frames and writes to video"""
        from services.perception.detector import get_detector_service

        detector = get_detector_service()
        frame_interval = 1.0 / self.fps
        last_frame_time = 0

        self.logger.info("Recording loop started")

        while not self._stop_event.is_set():
            try:
                # Rate limiting
                now = time.time()
                if now - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue

                # Get frame from detector
                frame = detector.get_last_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Resize to target resolution if needed
                if frame.shape[:2] != self.resolution:
                    frame = cv2.resize(frame, self.resolution)

                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw overlays
                with self._lock:
                    detections = self._last_detections.copy()
                    poses = self._last_poses.copy()
                    behaviors = self._last_behaviors.copy()
                    dog_assignments = self._last_dog_assignments.copy()
                    aruco_markers = self._last_aruco_markers.copy()

                frame = self._draw_overlays(frame, detections, poses, behaviors, dog_assignments, aruco_markers)

                # Write frame
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.frame_count += 1
                    last_frame_time = now

            except Exception as e:
                self.logger.error(f"Recording loop error: {e}")
                time.sleep(0.1)

        self.logger.info("Recording loop ended")

    def _draw_overlays(self, frame: np.ndarray, detections: List, poses: List,
                       behaviors: List, dog_assignments: Dict, aruco_markers: List = None) -> np.ndarray:
        """
        Draw bounding boxes, keypoints, and labels on frame

        Args:
            frame: Input frame (BGR)
            detections: List of detection dicts with bbox info
            poses: List of pose dicts with keypoints
            behaviors: List of behavior dicts (ignored - we use persistent buffer)
            dog_assignments: Dict mapping detection indices to dog names
            aruco_markers: List of ArUco marker tuples (id, cx, cy)

        Returns:
            Annotated frame
        """
        if aruco_markers is None:
            aruco_markers = []
        try:
            now = time.time()

            # Draw bounding boxes for each detection
            for i, det in enumerate(detections):
                # Get bbox coordinates
                bbox = det.get('bbox', det.get('box', []))
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

                    # Get dog name - check detection dict first, then assignments
                    dog_name = det.get('dog_name')
                    if not dog_name:
                        dog_name = dog_assignments.get(i, dog_assignments.get(str(i), None))
                    dog_id = det.get('dog_id', f'dog_{i}')

                    # Choose color based on dog
                    if dog_name == 'elsa':
                        color = (0, 255, 0)  # Green for Elsa
                    elif dog_name == 'bezik':
                        color = (255, 0, 255)  # Magenta for Bezik
                    else:
                        color = (0, 255, 255)  # Yellow for unknown

                    # Draw thick bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Draw dog name label at top of box
                    confidence = det.get('confidence', det.get('conf', 0))
                    label = f"{dog_name.upper() if dog_name else 'DOG'}"

                    # Label background at top of bounding box
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Check for persistent behavior to display under the bounding box
                    if dog_id in self._behavior_display_buffer:
                        behavior_data = self._behavior_display_buffer[dog_id]
                        age = now - behavior_data['timestamp']

                        if age < self._behavior_display_duration:
                            behavior_name = behavior_data['behavior'].upper()
                            behavior_conf = behavior_data['confidence']

                            # Calculate fade (full opacity for first 2s, then fade)
                            if age < 2.0:
                                alpha = 1.0
                            else:
                                alpha = 1.0 - (age - 2.0) / (self._behavior_display_duration - 2.0)

                            # Draw behavior label below bounding box with large text
                            behavior_label = f"{behavior_name} {behavior_conf:.0%}"
                            (bw, bh), _ = cv2.getTextSize(behavior_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

                            # Position below bounding box, centered
                            bx = x1 + (x2 - x1 - bw) // 2
                            by = y2 + bh + 10

                            # Clamp to frame bounds
                            bx = max(5, min(bx, frame.shape[1] - bw - 5))
                            by = min(by, frame.shape[0] - 5)

                            # Draw background rectangle
                            bg_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                            cv2.rectangle(frame, (bx - 5, by - bh - 5), (bx + bw + 5, by + 5), bg_color, -1)

                            # Draw behavior text
                            cv2.putText(frame, behavior_label, (bx, by),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Draw pose keypoints
            for pose in poses:
                keypoints = pose.get('keypoints', [])
                if not keypoints:
                    continue

                # Draw keypoints as circles
                for kp in keypoints:
                    if isinstance(kp, dict):
                        x, y = int(kp.get('x', 0)), int(kp.get('y', 0))
                        conf = kp.get('confidence', kp.get('conf', 1.0))
                    elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        conf = kp[2] if len(kp) > 2 else 1.0
                    else:
                        continue

                    if conf > 0.3 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(frame, (x, y), 5, (0, 128, 255), -1)  # Orange keypoints

                # Draw skeleton connections
                if len(keypoints) >= 17:
                    for start_idx, end_idx in SKELETON_CONNECTIONS:
                        if start_idx < len(keypoints) and end_idx < len(keypoints):
                            kp1 = keypoints[start_idx]
                            kp2 = keypoints[end_idx]

                            if isinstance(kp1, dict):
                                x1, y1 = int(kp1.get('x', 0)), int(kp1.get('y', 0))
                                c1 = kp1.get('confidence', kp1.get('conf', 1.0))
                            else:
                                x1, y1 = int(kp1[0]), int(kp1[1])
                                c1 = kp1[2] if len(kp1) > 2 else 1.0

                            if isinstance(kp2, dict):
                                x2, y2 = int(kp2.get('x', 0)), int(kp2.get('y', 0))
                                c2 = kp2.get('confidence', kp2.get('conf', 1.0))
                            else:
                                x2, y2 = int(kp2[0]), int(kp2[1])
                                c2 = kp2[2] if len(kp2) > 2 else 1.0

                            # Only draw if both points are confident
                            if c1 > 0.3 and c2 > 0.3:
                                if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                                    0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow skeleton

            # Draw ArUco markers if detected
            for marker in aruco_markers:
                if isinstance(marker, dict):
                    marker_id = marker.get('id')
                    cx, cy = int(marker.get('cx', 0)), int(marker.get('cy', 0))
                elif isinstance(marker, (list, tuple)) and len(marker) >= 3:
                    marker_id, cx, cy = int(marker[0]), int(marker[1]), int(marker[2])
                else:
                    continue

                # Draw marker position
                cv2.drawMarker(frame, (cx, cy), (255, 255, 0), cv2.MARKER_DIAMOND, 20, 2)

                # Label with dog name
                dog_name = {315: 'ELSA', 832: 'BEZIK'}.get(marker_id, f'ID:{marker_id}')
                cv2.putText(frame, f"[{dog_name}]", (cx + 15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Clean up expired behaviors from buffer
            expired = [k for k, v in self._behavior_display_buffer.items()
                      if now - v['timestamp'] > self._behavior_display_duration]
            for k in expired:
                del self._behavior_display_buffer[k]

            # Draw timestamp only (no recording indicator)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (frame.shape[1] - 180, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        except Exception as e:
            self.logger.error(f"Overlay drawing error: {e}")

        return frame

    def get_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        with self._lock:
            status = {
                "recording": self.recording,
                "filename": self.current_filename,
                "frames": self.frame_count,
                "output_dir": str(self.output_dir)
            }
            if self.recording and self.recording_start_time:
                status["duration"] = time.time() - self.recording_start_time
            return status

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all saved recordings"""
        recordings = []
        for f in sorted(self.output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = f.stat()
            recordings.append({
                "filename": f.name,
                "path": str(f),
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return recordings

    def toggle_recording(self, filename_prefix: str = "recording") -> Dict[str, Any]:
        """Toggle recording on/off"""
        if self.recording:
            return self.stop_recording()
        else:
            return self.start_recording(filename_prefix)


def get_video_recorder() -> VideoRecorder:
    """Get the singleton VideoRecorder instance"""
    global _video_recorder

    with _recorder_lock:
        if _video_recorder is None:
            _video_recorder = VideoRecorder()

    return _video_recorder
