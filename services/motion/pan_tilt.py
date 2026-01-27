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
        self.smoothed_target = None  # Smoothed target for less jitter
        self.last_detection_time = 0.0
        self.lost_target_time = 3.0  # seconds before starting scan

        # PID parameters - REDUCED for smoother tracking
        self.pid_params = {
            'pan': {'kp': 0.15, 'ki': 0.005, 'kd': 0.08},   # Reduced from 0.5/0.01/0.1
            'tilt': {'kp': 0.10, 'ki': 0.005, 'kd': 0.04}   # Reduced from 0.3/0.01/0.05
        }

        # PID state
        self.pid_state = {
            'pan': {'error_sum': 0.0, 'last_error': 0.0},
            'tilt': {'error_sum': 0.0, 'last_error': 0.0}
        }

        # Smoothing factor for target position (0.0 = no smoothing, 0.95 = very smooth)
        self.target_smoothing = 0.85

        # Camera parameters
        self.frame_width = 640
        self.frame_height = 640
        self.frame_center = (320, 320)
        self.deadzone = 50  # pixels - INCREASED from 30 for less jitter

        # Servo limits
        self.pan_limits = (10, 200)   # degrees
        self.tilt_limits = (20, 160)  # degrees
        self.current_pan = 90
        self.current_tilt = 90

        # Configurable center position (calibrated default viewing angle)
        # These values are the actual servo positions for "looking straight ahead"
        # pan=100 is physically centered, tilt=55 is level with ground
        self.center_pan = 100   # Pan center position (internal servo units)
        self.center_tilt = 55   # Tilt center position (internal servo units)

        # Servo command rate limiting (debounce)
        self.last_servo_command_time = 0.0
        self.min_command_interval = 0.05  # 50ms minimum between servo commands

        # Movement smoothing for API commands
        self.smoothing_factor = 0.3  # 0.0 = no smoothing, 1.0 = maximum smoothing
        self.pending_pan = None  # Target pan from API (smoothed towards)
        self.pending_tilt = None  # Target tilt from API (smoothed towards)

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

        # Manual camera control flag (app drive screen overrides auto-tracking)
        self._manual_camera_control = False

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
                # Check if Xbox controller is running - if so, do NOTHING
                import subprocess
                try:
                    result = subprocess.run(['pgrep', '-f', 'xbox_hybrid_controller'],
                                          capture_output=True, timeout=0.1)
                    if result.returncode == 0:
                        # Xbox controller is active, skip ALL camera control
                        continue
                except:
                    pass

                current_mode = self.state.get_mode()

                # Skip entire control loop in MANUAL mode to prevent conflicts
                if current_mode == SystemMode.MANUAL:
                    continue

                # Skip auto-tracking when app has manual camera control
                if self._manual_camera_control:
                    continue

                now = time.time()
                dt = now - last_update
                last_update = now

                if current_mode == SystemMode.COACH:
                    self._handle_coach_mode(dt)
                elif current_mode == SystemMode.SILENT_GUARDIAN:
                    self._handle_silent_guardian_mode(dt)
                elif current_mode == SystemMode.IDLE:
                    self._handle_idle_mode()

            except Exception as e:
                self.logger.error(f"Control loop error: {e}")

    def _handle_coach_mode(self, dt: float) -> None:
        """Handle tracking in coaching mode - active dog tracking for training"""
        if not self.tracking_enabled:
            return

        # Check if we're in manual mode - if so, don't auto-scan
        current_mode = self.state.get_mode()
        if current_mode == SystemMode.MANUAL:
            # Manual mode - stop autonomous scanning
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

    def _handle_silent_guardian_mode(self, dt: float) -> None:
        """Handle camera in Silent Guardian mode - stationary wide shot, no scanning"""
        # Silent Guardian = fixed camera position for passive bark monitoring
        # Camera stays stationary - no scanning or tracking
        pass

    def _handle_idle_mode(self) -> None:
        """Handle idle mode - center camera"""
        # Check if we're in manual mode - if so, don't auto-center
        current_mode = self.state.get_mode()
        if current_mode == SystemMode.MANUAL:
            # Manual mode - don't interfere with Xbox controller
            return

        if abs(self.current_pan - self.center_pan) > 5 or abs(self.current_tilt - self.center_tilt) > 5:
            self.center_camera()

    def _track_target(self, target: Tuple[float, float], dt: float) -> None:
        """Track target using PID control with smoothing"""
        target_x, target_y = target
        center_x, center_y = self.frame_center

        # Apply exponential smoothing to target position to reduce jitter
        if self.smoothed_target is None:
            self.smoothed_target = (target_x, target_y)
        else:
            smooth = self.target_smoothing
            self.smoothed_target = (
                smooth * self.smoothed_target[0] + (1 - smooth) * target_x,
                smooth * self.smoothed_target[1] + (1 - smooth) * target_y
            )

        # Use smoothed target for error calculation
        smooth_x, smooth_y = self.smoothed_target

        # Calculate errors from smoothed position
        error_x = smooth_x - center_x
        error_y = smooth_y - center_y

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

        # Limit output - REDUCED for smoother movement
        max_output = 3.0  # degrees per iteration (was 10.0)
        return max(-max_output, min(max_output, output))

    def _scan_for_target(self) -> None:
        """Execute smooth sweep scan pattern"""
        # Smooth continuous scanning instead of jumping between positions
        current_time = time.time()

        # Create a smooth sweep pattern
        # Period of 10 seconds for full sweep
        sweep_period = 10.0
        sweep_phase = (current_time % sweep_period) / sweep_period

        # Sweep back and forth
        if sweep_phase < 0.5:
            # Sweep left to right
            normalized_pos = sweep_phase * 2  # 0 to 1
        else:
            # Sweep right to left
            normalized_pos = 2 - (sweep_phase * 2)  # 1 to 0

        # Convert to pan angle (60 to 120 degrees)
        target_pan = 60 + (normalized_pos * 60)

        # Keep tilt steady at 75 degrees for scanning
        target_tilt = 75

        # Smooth movement - only update if significant change
        pan_diff = abs(target_pan - self.current_pan)
        if pan_diff > 2:  # Only move if > 2 degrees difference
            # Limit speed for smooth motion
            max_step = 3.0  # degrees per update
            if pan_diff > max_step:
                if target_pan > self.current_pan:
                    target_pan = self.current_pan + max_step
                else:
                    target_pan = self.current_pan - max_step

            self._move_to_position(target_pan, target_tilt)
            self.logger.debug(f"Smooth scan: pan={target_pan:.1f}°")

    def _move_to_position(self, pan: float, tilt: float, force: bool = False) -> None:
        """Move servos to specified position with limits, debounce, and smoothing

        Args:
            pan: Target pan angle
            tilt: Target tilt angle
            force: If True, bypass debounce (for internal tracking)
        """
        # Apply limits
        pan = max(self.pan_limits[0], min(self.pan_limits[1], pan))
        tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], tilt))

        # Debounce: skip if too soon since last command (unless forced)
        now = time.time()
        if not force and (now - self.last_servo_command_time) < self.min_command_interval:
            return

        # Only move if significant change (>1 degree)
        if abs(pan - self.current_pan) > 1 or abs(tilt - self.current_tilt) > 1:
            try:
                self.servo.set_camera_pan(pan)
                self.servo.set_camera_pitch(tilt)

                self.current_pan = pan
                self.current_tilt = tilt
                self.last_servo_command_time = now

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
        """Center camera to calibrated default viewing position

        Uses self.center_pan and self.center_tilt which are the calibrated
        servo positions for the desired default viewing angle.
        """
        # Check if Xbox controller is active - if so, don't center
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'xbox_hybrid_controller'],
                                  capture_output=True, timeout=0.1)
            if result.returncode == 0:
                self.logger.debug("Xbox controller active, skipping camera center")
                return True  # Return success but don't actually center
        except:
            pass

        if not self.servo_initialized or not self.servo:
            self.logger.error("Servos not initialized")
            return False

        try:
            # Use configurable center positions instead of hardcoded values
            success_pan = self.servo.set_camera_pan(self.center_pan, smooth=True)
            success_tilt = self.servo.set_camera_pitch(self.center_tilt, smooth=True)

            if success_pan and success_tilt:
                self.current_pan = self.center_pan
                self.current_tilt = self.center_tilt
                self.logger.debug(f"Camera centered to pan={self.center_pan}, tilt={self.center_tilt}")
                publish_motion_event('camera_centered', {
                    'pan': self.center_pan,
                    'tilt': self.center_tilt
                }, 'pan_tilt_service')
                return True
            else:
                self.logger.warning(f"Center camera partial failure: pan={success_pan}, tilt={success_tilt}")
                return False
        except Exception as e:
            self.logger.error(f"Center camera error: {e}")
            return False

    def set_center_position(self, pan: int = None, tilt: int = None) -> None:
        """Configure the center position for the camera

        Args:
            pan: Pan center position in servo units (10-200), None to keep current
            tilt: Tilt center position in servo units (20-160), None to keep current
        """
        if pan is not None:
            self.center_pan = max(self.pan_limits[0], min(self.pan_limits[1], pan))
            self.logger.info(f"Center pan set to {self.center_pan}")
        if tilt is not None:
            self.center_tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], tilt))
            self.logger.info(f"Center tilt set to {self.center_tilt}")

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

    def move_camera(self, pan: Optional[float] = None, tilt: Optional[float] = None,
                    smooth: bool = True) -> bool:
        """
        Move camera to specified position with optional smoothing

        Args:
            pan: Pan angle in degrees (10-200), None to keep current
            tilt: Tilt angle in degrees (20-160), None to keep current
            smooth: If True, apply smoothing to prevent jerky movements (default True)

        Returns:
            bool: True if movement was successful
        """
        if not self.servo_initialized:
            self.logger.error("Servos not initialized")
            return False

        try:
            # Use current position if not specified
            target_pan = pan if pan is not None else self.current_pan
            target_tilt = tilt if tilt is not None else self.current_tilt

            # Apply smoothing for API commands to prevent jerkiness
            if smooth and self.smoothing_factor > 0:
                # Interpolate between current and target position
                smoothed_pan = self.current_pan + (target_pan - self.current_pan) * (1 - self.smoothing_factor)
                smoothed_tilt = self.current_tilt + (target_tilt - self.current_tilt) * (1 - self.smoothing_factor)

                # Store pending targets for gradual movement
                self.pending_pan = target_pan
                self.pending_tilt = target_tilt

                # Move to smoothed position
                self._move_to_position(smoothed_pan, smoothed_tilt)
                self.logger.debug(f"Camera smoothed to pan={smoothed_pan:.1f}° (target={target_pan:.1f}°)")
            else:
                # Direct movement
                self._move_to_position(target_pan, target_tilt)
                self.logger.info(f"Camera moved to pan={target_pan:.1f}°, tilt={target_tilt:.1f}°")

            return True

        except Exception as e:
            self.logger.error(f"Camera movement error: {e}")
            return False

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable tracking"""
        self.tracking_enabled = enabled
        if enabled:
            self.logger.info("Tracking enabled")
        else:
            self.logger.info("Tracking disabled")
            self.center_camera()

    def set_manual_camera(self, active: bool) -> None:
        """Enable/disable manual camera control from app drive screen.

        When active, auto-tracking and scanning are suppressed so the app
        can control the camera via move_camera() without interference.
        """
        self._manual_camera_control = active
        self.logger.info(f"Manual camera control {'enabled' if active else 'disabled'}")

    def get_status(self) -> Dict[str, Any]:
        """Get pan/tilt service status"""
        return {
            'initialized': self.servo_initialized,
            'running': self.running,
            'tracking_enabled': self.tracking_enabled,
            'manual_camera_control': self._manual_camera_control,
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