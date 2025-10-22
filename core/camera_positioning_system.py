#!/usr/bin/env python3
"""
Universal Camera Positioning System
Handles auto-centering, manual control, and API for all mission types
"""

import time
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class CameraMode(Enum):
    MANUAL = "manual"           # User/app control
    AUTO_CENTER = "auto_center" # Follow detected dogs
    PATROL = "patrol"           # Sweep area looking for dogs
    FIXED = "fixed"             # Lock position
    MISSION = "mission"         # Mission-specific positioning

class CameraPositioningSystem:
    """Universal camera control for all TreatBot systems"""

    def __init__(self, servo_controller=None):
        self.servo = servo_controller

        # Camera position tracking
        self.current_pan = 90   # 0-180 degrees
        self.current_pitch = 90 # 0-180 degrees (90 = level)

        # Optimal positioning for dog detection
        self.optimal_pitch = 80  # 10 degrees down from level
        self.optimal_pan = 90    # Center

        # Auto-centering parameters
        self.frame_width = 1920
        self.frame_height = 1080
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        # Detection zone (where we want dogs to be)
        self.target_zone = {
            'x_min': self.frame_width * 0.3,   # 30% from left
            'x_max': self.frame_width * 0.7,   # 70% from left
            'y_min': self.frame_height * 0.25, # 25% from top
            'y_max': self.frame_height * 0.75  # 75% from top
        }

        # Movement sensitivity
        self.pan_sensitivity = 0.05   # degrees per pixel offset
        self.pitch_sensitivity = 0.03 # degrees per pixel offset
        self.min_movement = 2         # minimum degrees to move
        self.max_movement = 15        # maximum degrees per adjustment

        # Auto-centering state
        self.mode = CameraMode.FIXED
        self.last_detection_time = 0
        self.lost_dog_timeout = 5.0   # seconds before patrol mode

    def initialize_optimal_position(self):
        """Set camera to optimal position for dog detection"""
        print("🎯 Setting camera to optimal dog detection position")
        self.set_position(pan=self.optimal_pan, pitch=self.optimal_pitch, smooth=True)

    def set_position(self, pan: float, pitch: float, smooth: bool = False) -> bool:
        """Set absolute camera position"""
        if not self.servo:
            print("❌ No servo controller available")
            return False

        # Clamp values to safe ranges
        pan = max(30, min(150, pan))     # Prevent over-rotation
        pitch = max(60, min(120, pitch)) # Prevent pointing too high/low

        success = True
        if pan != self.current_pan:
            success &= self.servo.set_camera_pan(pan, smooth)
            if success:
                self.current_pan = pan

        if pitch != self.current_pitch:
            success &= self.servo.set_camera_pitch(pitch, smooth)
            if success:
                self.current_pitch = pitch

        return success

    def adjust_relative(self, pan_delta: float, pitch_delta: float) -> bool:
        """Adjust camera position relative to current position"""
        new_pan = self.current_pan + pan_delta
        new_pitch = self.current_pitch + pitch_delta
        return self.set_position(new_pan, new_pitch)

    def center_on_detection(self, detection: DetectionBox) -> bool:
        """Auto-center camera on detected dog"""
        if self.mode != CameraMode.AUTO_CENTER:
            return False

        # Calculate dog center point
        dog_center_x = (detection.x1 + detection.x2) / 2
        dog_center_y = (detection.y1 + detection.y2) / 2

        # Calculate offset from frame center
        offset_x = dog_center_x - self.center_x
        offset_y = dog_center_y - self.center_y

        # Check if dog is outside target zone
        in_target_zone = (
            self.target_zone['x_min'] <= dog_center_x <= self.target_zone['x_max'] and
            self.target_zone['y_min'] <= dog_center_y <= self.target_zone['y_max']
        )

        if in_target_zone:
            # Dog is well-positioned, no movement needed
            self.last_detection_time = time.time()
            return True

        # Calculate required movement
        pan_adjustment = -offset_x * self.pan_sensitivity  # Negative: left offset = pan left
        pitch_adjustment = offset_y * self.pitch_sensitivity # Positive: down offset = pitch down

        # Apply movement limits
        pan_adjustment = max(-self.max_movement, min(self.max_movement, pan_adjustment))
        pitch_adjustment = max(-self.max_movement, min(self.max_movement, pitch_adjustment))

        # Only move if adjustment is significant
        if abs(pan_adjustment) > self.min_movement or abs(pitch_adjustment) > self.min_movement:
            print(f"🎯 Auto-centering: pan {pan_adjustment:+.1f}°, pitch {pitch_adjustment:+.1f}°")
            success = self.adjust_relative(pan_adjustment, pitch_adjustment)
            if success:
                self.last_detection_time = time.time()
            return success

        return True

    def update_auto_positioning(self, detections: List[DetectionBox]) -> bool:
        """Update camera position based on current detections"""
        current_time = time.time()

        if detections and self.mode == CameraMode.AUTO_CENTER:
            # Use strongest detection for centering
            best_detection = max(detections, key=lambda d: d.confidence)
            return self.center_on_detection(best_detection)

        elif not detections and self.mode == CameraMode.AUTO_CENTER:
            # No dogs detected - check if we should start patrol
            time_since_last = current_time - self.last_detection_time
            if time_since_last > self.lost_dog_timeout:
                print("🔍 Lost dog - starting patrol mode")
                self.set_mode(CameraMode.PATROL)

        return True

    def set_mode(self, mode: CameraMode) -> bool:
        """Change camera positioning mode"""
        print(f"📹 Camera mode: {self.mode.value} → {mode.value}")
        self.mode = mode

        if mode == CameraMode.AUTO_CENTER:
            self.last_detection_time = time.time()
        elif mode == CameraMode.FIXED:
            self.initialize_optimal_position()
        elif mode == CameraMode.PATROL:
            self._start_patrol()

        return True

    def _start_patrol(self) -> bool:
        """Begin patrol pattern to search for dogs"""
        # Simple left-right sweep pattern
        print("🔍 Starting camera patrol sweep")

        # Sweep pattern: left → center → right → center
        patrol_positions = [
            (60, self.optimal_pitch),   # Look left
            (90, self.optimal_pitch),   # Look center
            (120, self.optimal_pitch),  # Look right
            (90, self.optimal_pitch),   # Return center
        ]

        for pan, pitch in patrol_positions:
            self.set_position(pan, pitch, smooth=True)
            time.sleep(2.0)  # Pause at each position

        # Return to auto-center mode
        self.set_mode(CameraMode.AUTO_CENTER)
        return True

    def get_status(self) -> dict:
        """Get current camera positioning status"""
        return {
            'mode': self.mode.value,
            'pan': self.current_pan,
            'pitch': self.current_pitch,
            'optimal_pan': self.optimal_pan,
            'optimal_pitch': self.optimal_pitch,
            'last_detection': self.last_detection_time,
            'servo_available': self.servo is not None
        }

    # API methods for different systems
    def mission_control(self, command: str, **kwargs) -> bool:
        """API for mission system camera control"""
        if command == "center_on_dog":
            self.set_mode(CameraMode.AUTO_CENTER)
        elif command == "fixed_position":
            self.set_mode(CameraMode.FIXED)
        elif command == "manual_adjust":
            return self.adjust_relative(kwargs.get('pan', 0), kwargs.get('pitch', 0))
        elif command == "patrol":
            self.set_mode(CameraMode.PATROL)
        return True

    def app_control(self, pan: float, pitch: float) -> bool:
        """API for mobile app camera control"""
        self.set_mode(CameraMode.MANUAL)
        return self.set_position(pan, pitch, smooth=True)

    def ai_control(self, detections: List[DetectionBox]) -> bool:
        """API for AI system camera control"""
        return self.update_auto_positioning(detections)

# Suggested usage in main systems:
"""
# In mission system:
camera_pos = CameraPositioningSystem(servo_controller)
camera_pos.mission_control("center_on_dog")

# In AI detection loop:
camera_pos.ai_control(detections)

# In mobile app:
camera_pos.app_control(pan=100, pitch=85)

# For manual adjustment:
camera_pos.adjust_relative(pan_delta=5, pitch_delta=-2)
"""