#!/usr/bin/env python3
"""
Camera pan/tilt tracking service using servo_controller
Implements PID tracking of detected dogs
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, Tuple

from core.bus import get_bus, publish_motion_event
from core.state import get_state, SystemMode
from core.hardware.servo_controller import ServoController


class PanTiltService:
    """
    Camera tracking using servo controller
    PID tracking of detected dogs, scan patterns for vigilant mode
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.logger = logging.getLogger('PanTiltService')

        # Servo controller
        self.servo = None
        self.servo_initialized = False

        # Tracking state
        self.tracking_enabled = False
        self.target_position = None  # (x, y) in frame coordinates
        self.last_detection_time = 0.0
        self.lost_target_time = 3.0  # seconds before starting scan

        # PID parameters
        self.pid_params = {
            'pan': {'kp': 0.5, 'ki': 0.01, 'kd': 0.1},
            'tilt': {'kp': 0.3, 'ki': 0.01, 'kd': 0.05}
        }

        # PID state
        self.pid_state = {
            'pan': {'error_sum': 0.0, 'last_error': 0.0},
            'tilt': {'error_sum': 0.0, 'last_error': 0.0}
        }

        # Camera parameters
        self.frame_width = 640
        self.frame_height = 640
        self.frame_center = (320, 320)
        self.deadzone = 30  # pixels

        # Servo limits
        self.pan_limits = (10, 200)   # degrees
        self.tilt_limits = (20, 160)  # degrees
        self.current_pan = 90
        self.current_tilt = 90

        # Scan pattern
        self.scan_positions = [
            (50, 90),   # left
            (90, 90),   # center
            (130, 90),  # right
            (90, 60),   # up
            (90, 120)   # down
        ]
        self.scan_index = 0
        self.scan_delay = 2.0  # seconds per position

        # Subscribe to vision events
        self.bus.subscribe('vision', self._on_vision_event)

        # Control thread
        self.control_thread = None
        self.running = False
        self._stop_event = threading.Event()

    def initialize(self) -> bool:
        """Initialize servo controller"""
        try:
            self.servo = ServoController()

            if self.servo.is_initialized():
                self.servo_initialized = True
                self.logger.info("Pan/tilt servos initialized")

                # Center camera
                self.center_camera()

                self.state.update_hardware(servos_initialized=True)
                return True
            else:
                self.logger.error("Servo controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"Pan/tilt initialization error: {e}")
            return False

    def start_tracking(self) -> bool:
        """Start tracking control loop"""
        if not self.servo_initialized:
            self.logger.error("Servos not initialized")
            return False

        if self.running:
            self.logger.warning("Tracking already running")
            return True

        self.running = True
        self._stop_event.clear()
        self.tracking_enabled = True

        self.control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="PanTiltControl"
        )
        self.control_thread.start()

        self.logger.info("Pan/tilt tracking started")
        publish_motion_event('tracking_started', {}, 'pan_tilt_service')
        return True

    def stop_tracking(self) -> None:
        """Stop tracking control loop"""
        if not self.running:
            return

        self.running = False
        self.tracking_enabled = False
        self._stop_event.set()

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        self.logger.info("Pan/tilt tracking stopped")
        publish_motion_event('tracking_stopped', {}, 'pan_tilt_service')

    def _control_loop(self) -> None:
        """Main control loop"""
        last_update = time.time()

        while not self._stop_event.wait(0.05):  # 20Hz control loop
            try:
                current_mode = self.state.get_mode()
                now = time.time()
                dt = now - last_update
                last_update = now

                if current_mode == SystemMode.DETECTION:
                    self._handle_detection_mode(dt)
                elif current_mode == SystemMode.VIGILANT:
                    self._handle_vigilant_mode(dt)
                elif current_mode == SystemMode.IDLE:
                    self._handle_idle_mode()

            except Exception as e:
                self.logger.error(f"Control loop error: {e}")

    def _handle_detection_mode(self, dt: float) -> None:
        """Handle tracking in detection mode"""
        if not self.tracking_enabled:
            return

        now = time.time()

        if self.target_position:
            # Check if target is recent
            if now - self.last_detection_time < self.lost_target_time:
                # Track target
                self._track_target(self.target_position, dt)
            else:
                # Target lost, start scanning
                self.logger.debug("Target lost, starting scan")
                self.target_position = None
                self._scan_for_target()
        else:
            # No target, scan
            self._scan_for_target()

    def _handle_vigilant_mode(self, dt: float) -> None:
        """Handle scanning in vigilant mode"""
        self._scan_for_target()

    def _handle_idle_mode(self) -> None:
        """Handle idle mode - center camera"""
        if abs(self.current_pan - 90) > 5 or abs(self.current_tilt - 90) > 5:
            self.center_camera()

    def _track_target(self, target: Tuple[float, float], dt: float) -> None:
        """Track target using PID control"""
        target_x, target_y = target
        center_x, center_y = self.frame_center

        # Calculate errors
        error_x = target_x - center_x
        error_y = target_y - center_y

        # Check if target is in deadzone
        if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
            return  # Target centered, no adjustment needed

        # PID control for pan (X axis)
        pan_adjustment = self._pid_control('pan', error_x, dt)
        new_pan = self.current_pan - pan_adjustment  # Invert for camera movement

        # PID control for tilt (Y axis)
        tilt_adjustment = self._pid_control('tilt', error_y, dt)
        new_tilt = self.current_tilt + tilt_adjustment

        # Apply limits and move servos
        self._move_to_position(new_pan, new_tilt)

    def _pid_control(self, axis: str, error: float, dt: float) -> float:
        """PID control calculation"""
        if dt <= 0:
            return 0.0

        params = self.pid_params[axis]
        state = self.pid_state[axis]

        # Proportional term
        p_term = params['kp'] * error

        # Integral term
        state['error_sum'] += error * dt
        i_term = params['ki'] * state['error_sum']

        # Derivative term
        error_rate = (error - state['last_error']) / dt
        d_term = params['kd'] * error_rate

        # Update state
        state['last_error'] = error

        # Combine terms
        output = p_term + i_term + d_term

        # Limit output
        max_output = 10.0  # degrees per iteration
        return max(-max_output, min(max_output, output))

    def _scan_for_target(self) -> None:
        """Execute scan pattern"""
        # Simple time-based scanning
        scan_time = time.time() % (len(self.scan_positions) * self.scan_delay)
        scan_index = int(scan_time / self.scan_delay)

        if scan_index != self.scan_index:
            self.scan_index = scan_index
            pan, tilt = self.scan_positions[scan_index]
            self._move_to_position(pan, tilt)
            self.logger.debug(f"Scanning to position {scan_index}: ({pan}, {tilt})")

    def _move_to_position(self, pan: float, tilt: float) -> None:
        """Move servos to specified position with limits"""
        # Apply limits
        pan = max(self.pan_limits[0], min(self.pan_limits[1], pan))
        tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], tilt))

        # Only move if significant change
        if abs(pan - self.current_pan) > 1 or abs(tilt - self.current_tilt) > 1:
            try:
                self.servo.set_camera_pan(pan)
                self.servo.set_camera_pitch(tilt)

                self.current_pan = pan
                self.current_tilt = tilt

                # Update state
                self.state.update_hardware(
                    servo_positions={'pan': pan, 'tilt': tilt}
                )

                publish_motion_event('servo_moved', {
                    'pan': pan,
                    'tilt': tilt
                }, 'pan_tilt_service')

            except Exception as e:
                self.logger.error(f"Servo movement error: {e}")

    def center_camera(self) -> bool:
        """Center camera to neutral position"""
        try:
            success = self.servo.center_camera()
            if success:
                self.current_pan = 90
                self.current_tilt = 90
                self.logger.info("Camera centered")
                publish_motion_event('camera_centered', {}, 'pan_tilt_service')
            return success
        except Exception as e:
            self.logger.error(f"Center camera error: {e}")
            return False

    def _on_vision_event(self, event) -> None:
        """Handle vision events"""
        if event.subtype == 'dog_detected':
            # Update target position
            center = event.data.get('center', self.frame_center)
            self.target_position = (center[0], center[1])
            self.last_detection_time = time.time()

            self.logger.debug(f"Target updated: {self.target_position}")

        elif event.subtype == 'detection_stopped':
            # Clear target
            self.target_position = None

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable tracking"""
        self.tracking_enabled = enabled
        if enabled:
            self.logger.info("Tracking enabled")
        else:
            self.logger.info("Tracking disabled")
            self.center_camera()

    def get_status(self) -> Dict[str, Any]:
        """Get pan/tilt service status"""
        return {
            'initialized': self.servo_initialized,
            'running': self.running,
            'tracking_enabled': self.tracking_enabled,
            'current_position': {'pan': self.current_pan, 'tilt': self.current_tilt},
            'target_position': self.target_position,
            'has_target': self.target_position is not None,
            'time_since_detection': time.time() - self.last_detection_time if self.last_detection_time > 0 else 999
        }

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_tracking()
        if self.servo:
            self.servo.cleanup()
        self.logger.info("Pan/tilt service cleaned up")


# Global pan/tilt service instance
_pantilt_instance = None
_pantilt_lock = threading.Lock()

def get_pantilt_service() -> PanTiltService:
    """Get the global pan/tilt service instance (singleton)"""
    global _pantilt_instance
    if _pantilt_instance is None:
        with _pantilt_lock:
            if _pantilt_instance is None:
                _pantilt_instance = PanTiltService()
    return _pantilt_instance