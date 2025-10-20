#!/usr/bin/env python3
"""
BehaviorAnalyzer - Unified behavior detection and analysis
Consolidates behavior detection logic from multiple implementations
"""

import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass
from ..utils.event_bus import EventBus

@dataclass
class BehaviorFrame:
    """Single frame of behavior data"""
    timestamp: float
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    center: tuple  # (x, y)
    aspect_ratio: float

class BehaviorAnalyzer:
    """
    Unified behavior analysis system
    Detects sitting, lying, staying, spinning behaviors
    """

    def __init__(self, behavior_config: Dict[str, Any], event_bus: EventBus):
        self.logger = logging.getLogger('BehaviorAnalyzer')
        self.behavior_config = behavior_config
        self.event_bus = event_bus

        # Behavior history for temporal analysis
        self.behavior_history = deque(maxlen=100)  # Keep last 100 frames (~3.3 seconds at 30fps)

        # Current behavior tracking
        self.current_behavior = "idle"
        self.behavior_start_time = 0.0
        self.behavior_confidence = 0.0

        # Behavior timers
        self.behavior_timers = {}

        # Configuration
        self.required_durations = behavior_config.get('reward_behaviors', {})

        self.logger.info("Behavior analyzer initialized")

    def analyze_frame(self, detection_data: Dict[str, Any]) -> Optional[str]:
        """
        Analyze a single frame for behavior

        Args:
            detection_data: Detection information from camera manager

        Returns:
            Detected behavior string or None
        """
        if not detection_data or 'detection' not in detection_data:
            self._reset_behavior_tracking()
            return None

        detection = detection_data['detection']
        bbox = detection.get('bbox', [0, 0, 1, 1])
        confidence = detection.get('confidence', 0.0)
        center = detection_data.get('center', (0, 0))

        # Calculate aspect ratio for pose analysis
        width, height = bbox[2], bbox[3]
        aspect_ratio = height / width if width > 0 else 1.0

        # Create behavior frame
        behavior_frame = BehaviorFrame(
            timestamp=time.time(),
            bbox=bbox,
            confidence=confidence,
            center=center,
            aspect_ratio=aspect_ratio
        )

        # Add to history
        self.behavior_history.append(behavior_frame)

        # Analyze behavior
        detected_behavior = self._detect_behavior(behavior_frame)

        # Update behavior state
        self._update_behavior_state(detected_behavior, confidence)

        return detected_behavior

    def _detect_behavior(self, current_frame: BehaviorFrame) -> str:
        """Detect behavior from current frame and history"""
        # Static pose analysis
        static_behavior = self._analyze_static_pose(current_frame.aspect_ratio)

        # Temporal analysis for complex behaviors
        if len(self.behavior_history) >= 20:  # Need sufficient history
            # Check for spinning
            if self._detect_spinning():
                return "spinning"

            # Check for staying (no movement)
            if self._detect_staying():
                return "staying"

        return static_behavior

    def _analyze_static_pose(self, aspect_ratio: float) -> str:
        """Analyze static pose based on bounding box aspect ratio"""
        # Aspect ratio thresholds (tuned from multiple implementations)
        if 0.9 < aspect_ratio < 1.3:
            return "sitting"
        elif aspect_ratio < 0.7:
            return "lying"
        elif aspect_ratio > 1.3:
            return "standing"
        else:
            return "idle"

    def _detect_spinning(self) -> bool:
        """Detect spinning behavior from movement history"""
        if len(self.behavior_history) < 20:
            return False

        # Get recent positions
        recent_frames = list(self.behavior_history)[-20:]
        positions = [frame.center for frame in recent_frames]

        # Calculate center of movement
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)

        # Calculate angles relative to center
        angles = []
        for x, y in positions:
            angle = np.arctan2(y - center_y, x - center_x)
            angles.append(angle)

        # Check for consistent rotation (angles increasing or decreasing)
        angle_diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]

        # Normalize angle differences to [-π, π]
        angle_diffs = [(diff + np.pi) % (2 * np.pi) - np.pi for diff in angle_diffs]

        # Check for consistent direction
        positive_diffs = sum(1 for diff in angle_diffs if diff > 0.1)
        negative_diffs = sum(1 for diff in angle_diffs if diff < -0.1)

        # Spinning if most diffs are in same direction
        return (positive_diffs > len(angle_diffs) * 0.7 or
                negative_diffs > len(angle_diffs) * 0.7)

    def _detect_staying(self) -> bool:
        """Detect staying behavior (minimal movement)"""
        if len(self.behavior_history) < 90:  # Need ~3 seconds of history
            return False

        # Get recent positions
        recent_frames = list(self.behavior_history)[-90:]
        positions = [frame.center for frame in recent_frames]

        # Calculate movement variance
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]

        x_variance = np.var(x_positions)
        y_variance = np.var(y_positions)

        # Low variance indicates staying in place
        movement_threshold = 100  # pixels^2
        return x_variance < movement_threshold and y_variance < movement_threshold

    def _update_behavior_state(self, detected_behavior: str, confidence: float):
        """Update current behavior state and timers"""
        current_time = time.time()

        # If behavior changed, reset timers
        if detected_behavior != self.current_behavior:
            self.current_behavior = detected_behavior
            self.behavior_start_time = current_time
            self.behavior_confidence = confidence

            # Clear old timers
            self.behavior_timers.clear()

            self.logger.debug(f"Behavior changed to: {detected_behavior}")

        # Update timer for current behavior
        if detected_behavior != "idle":
            if detected_behavior not in self.behavior_timers:
                self.behavior_timers[detected_behavior] = current_time

            # Check if behavior duration meets reward criteria
            duration = current_time - self.behavior_timers[detected_behavior]
            required_duration = self.required_durations.get(detected_behavior, {}).get('duration_required', 999)
            required_confidence = self.required_durations.get(detected_behavior, {}).get('confidence_required', 0.9)

            if (duration >= required_duration and
                confidence >= required_confidence):

                # Emit behavior event
                self.event_bus.publish('behavior_detected', {
                    'behavior': detected_behavior,
                    'confidence': confidence,
                    'duration': duration,
                    'timestamp': current_time
                })

                # Reset timer to prevent immediate re-triggering
                self.behavior_timers[detected_behavior] = current_time

    def _reset_behavior_tracking(self):
        """Reset behavior tracking when no dog detected"""
        self.current_behavior = "idle"
        self.behavior_start_time = 0.0
        self.behavior_confidence = 0.0
        self.behavior_timers.clear()

    def get_current_behavior(self) -> Dict[str, Any]:
        """Get current behavior information"""
        duration = 0.0
        if self.behavior_start_time > 0:
            duration = time.time() - self.behavior_start_time

        return {
            'behavior': self.current_behavior,
            'confidence': self.behavior_confidence,
            'duration': duration,
            'frame_count': len(self.behavior_history)
        }

    def get_behavior_statistics(self) -> Dict[str, Any]:
        """Get behavior analysis statistics"""
        recent_behaviors = {}

        # Count behaviors in recent history
        for frame in self.behavior_history:
            # Would need to store behavior with frame for accurate stats
            pass

        return {
            'total_frames_analyzed': len(self.behavior_history),
            'current_behavior': self.current_behavior,
            'behavior_duration': time.time() - self.behavior_start_time if self.behavior_start_time > 0 else 0
        }