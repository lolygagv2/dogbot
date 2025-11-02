#!/usr/bin/env python3
"""
Hybrid Xbox Controller for DogBot
- Direct motor control for low latency movement
- API calls for other features (photos, sounds, treats)
"""

import struct
import time
import os
import sys
import logging
import requests
from threading import Thread, Event, Timer
from dataclasses import dataclass
from typing import Optional, Tuple

# Add project root to path for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct hardware import for motors only
try:
    from core.hardware.motor_controller import MotorController
    MOTOR_DIRECT = True
    motor_controller = MotorController()
    logger = logging.getLogger('XboxHybrid')
    logger.info("Direct motor control initialized")
except ImportError:
    # Fallback to gpioset if PWM not available
    try:
        from core.hardware.motor_controller_gpioset import MotorController
        MOTOR_DIRECT = True
        motor_controller = MotorController()
        logger = logging.getLogger('XboxHybrid')
        logger.info("Direct motor control initialized (gpioset mode)")
    except ImportError:
        MOTOR_DIRECT = False
        motor_controller = None
        logger = logging.getLogger('XboxHybrid')
        logger.warning("No direct motor control available, will use API")

logging.basicConfig(level=logging.INFO)

@dataclass
class ControllerState:
    """Track controller button/axis states"""
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    a_button: bool = False
    b_button: bool = False
    x_button: bool = False
    y_button: bool = False
    left_bumper: bool = False
    right_bumper: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False

    # Track motor state to avoid redundant commands
    last_left_speed: int = 0
    last_right_speed: int = 0
    motors_stopped: bool = True


class XboxHybridController:
    """Hybrid Xbox controller - direct motors, API for rest"""

    # API configuration
    API_BASE_URL = "http://localhost:8000"

    # Controller configuration
    DEADZONE = 0.15
    TRIGGER_DEADZONE = 0.1
    MAX_SPEED = 100
    TURN_SPEED_FACTOR = 0.6

    # Sound track numbers on SD card (D-pad navigation)
    # Track 1-8 are different sound effects
    SOUND_TRACKS = [
        (1, "Track 1"),
        (2, "Track 2"),
        (3, "Track 3"),
        (4, "Track 4"),
        (5, "Track 5"),
        (6, "Track 6"),
        (7, "Track 7"),
        (8, "Track 8")
    ]

    def __init__(self, device_path: str = '/dev/input/js0'):
        self.device_path = device_path
        self.device = None
        self.running = False
        self.state = ControllerState()
        self.stop_event = Event()

        # Sound navigation
        self.current_sound_index = 0

        # Photo capture cooldown
        self.last_photo_time = 0
        self.photo_cooldown = 2.0  # seconds

        # Camera control timer for smooth movement
        self.camera_timer = None
        self.camera_update_interval = 0.05  # 20Hz update rate for smooth movement

        # API session for non-motor functions
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        logger.info(f"Xbox Hybrid Controller initialized for {device_path}")
        logger.info(f"Motor control: {'DIRECT' if MOTOR_DIRECT else 'API'}")
        logger.info(f"API endpoint: {self.API_BASE_URL}")

    def api_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """Make API request with error handling"""
        url = f"{self.API_BASE_URL}{endpoint}"
        try:
            # Longer timeout for audio commands since DFPlayer is slow
            timeout = 10.0 if 'audio' in endpoint else 2.0

            if method == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            return None

    def connect(self) -> bool:
        """Connect to the Xbox controller"""
        try:
            # Check if API is available for other features
            health = self.api_request('GET', '/health')
            if health:
                logger.info(f"API health check: {health}")
            else:
                logger.warning("API server not responding - only motor control will work")

            # Open the joystick device
            self.device = open(self.device_path, 'rb')
            logger.info(f"Connected to Xbox controller at {self.device_path}")

            # Motors are initialized in __init__, just log status
            if MOTOR_DIRECT and motor_controller:
                if motor_controller.is_initialized():
                    logger.info("Direct motor control ready")
                else:
                    logger.warning("Motor controller not properly initialized")

            return True

        except FileNotFoundError:
            logger.error(f"Controller not found at {self.device_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def read_event(self) -> Optional[Tuple]:
        """Read a single joystick event"""
        try:
            event_data = self.device.read(8)
            if event_data:
                # Parse as signed integers (fixed the issue)
                timestamp, value, event_type, number = struct.unpack('IhBB', event_data)
                return (timestamp, value, event_type, number)
        except Exception as e:
            logger.error(f"Error reading event: {e}")
        return None

    def process_axis(self, number: int, value: int):
        """Process axis movement"""
        # Normalize to -1.0 to 1.0
        normalized = value / 32767.0

        # Apply deadzone
        if abs(normalized) < self.DEADZONE:
            normalized = 0.0

        # Update state based on axis
        if number == 0:  # Left stick X
            self.state.left_x = normalized
        elif number == 1:  # Left stick Y (inverted for forward)
            self.state.left_y = -normalized
        elif number == 3:  # Right stick X (camera pan)
            self.state.right_x = normalized
            # Control camera continuously for smooth movement
        elif number == 4:  # Right stick Y (camera tilt)
            self.state.right_y = -normalized
            # Control camera continuously for smooth movement
        elif number == 2:  # Left trigger
            self.state.left_trigger = (value + 32767) / 65534.0
        elif number == 5:  # Right trigger
            self.state.right_trigger = (value + 32767) / 65534.0

        # Update motor control for left stick
        if number in [0, 1]:
            self.update_motor_control()

    def process_button(self, number: int, pressed: bool):
        """Process button press/release"""
        if number == 0:  # A button
            self.state.a_button = pressed
            if pressed:
                logger.info("A button: Emergency stop")
                self.emergency_stop()

        elif number == 1:  # B button
            self.state.b_button = pressed
            if pressed:
                logger.info("B button: Stop motors")
                self.stop_motors()

        elif number == 2:  # X button
            self.state.x_button = pressed

        elif number == 3:  # Y button
            self.state.y_button = pressed
            if pressed:
                self.play_sound_effect()

        elif number == 4:  # Left bumper (LB) - Dispense treat
            self.state.left_bumper = pressed
            if pressed:
                self.dispense_treat()

        elif number == 5:  # Right bumper (RB) - Take photo
            self.state.right_bumper = pressed
            if pressed:
                self.take_photo()

    def process_dpad(self, number: int, value: int):
        """Process D-pad input for audio control"""
        if number == 6:  # D-pad X axis
            self.state.dpad_left = (value < 0)
            self.state.dpad_right = (value > 0)

            if value < 0:  # Left - Previous track
                self.current_sound_index = (self.current_sound_index - 1) % len(self.SOUND_TRACKS)
                track_num, track_name = self.SOUND_TRACKS[self.current_sound_index]
                logger.info(f"Selected: {track_name}")
            elif value > 0:  # Right - Next track
                self.current_sound_index = (self.current_sound_index + 1) % len(self.SOUND_TRACKS)
                track_num, track_name = self.SOUND_TRACKS[self.current_sound_index]
                logger.info(f"Selected: {track_name}")

        elif number == 7:  # D-pad Y axis
            self.state.dpad_up = (value < 0)
            self.state.dpad_down = (value > 0)

            if value < 0:  # Up - Pause/Resume
                logger.info("D-pad up: Pause/Resume")
                self.api_request('POST', '/audio/pause')
            elif value > 0:  # Down - Play selected track
                track_num, track_name = self.SOUND_TRACKS[self.current_sound_index]
                logger.info(f"D-pad down: Play {track_name}")
                self.play_sound_effect()

    def update_motor_control(self):
        """Update motor speeds based on joystick input"""
        # Variable speed based on trigger
        speed_multiplier = 0.3 + (self.state.right_trigger * 0.7)

        # Calculate motor speeds from left stick
        forward = self.state.left_y * self.MAX_SPEED * speed_multiplier
        turn = self.state.left_x * self.MAX_SPEED * self.TURN_SPEED_FACTOR * speed_multiplier

        left_speed = int(forward + turn)
        right_speed = int(forward - turn)

        # Clamp to valid range
        left_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, left_speed))
        right_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, right_speed))

        # Only send if changed significantly
        if (abs(left_speed - self.state.last_left_speed) > 5 or
            abs(right_speed - self.state.last_right_speed) > 5 or
            (left_speed == 0 and right_speed == 0 and not self.state.motors_stopped)):

            self.set_motor_speeds(left_speed, right_speed)
            self.state.last_left_speed = left_speed
            self.state.last_right_speed = right_speed
            self.state.motors_stopped = (left_speed == 0 and right_speed == 0)

    def set_motor_speeds(self, left: int, right: int):
        """Set motor speeds - DIRECT hardware control for low latency"""
        if MOTOR_DIRECT and motor_controller:
            # Direct hardware control - minimal latency
            try:
                if left == 0 and right == 0:
                    motor_controller.emergency_stop()
                    logger.debug("Motors stopped (direct)")
                else:
                    # Set individual motor speeds directly
                    # Motor A (left), Motor B (right)
                    left_dir = 'forward' if left >= 0 else 'backward'
                    right_dir = 'forward' if right >= 0 else 'backward'
                    left_speed = abs(left)
                    right_speed = abs(right)

                    motor_controller.set_motor_speed('A', left_speed, left_dir)
                    motor_controller.set_motor_speed('B', right_speed, right_dir)
                    logger.debug(f"Motors (direct): L={left:4d}, R={right:4d}")
            except Exception as e:
                logger.error(f"Direct motor control error: {e}")
                # Fallback to API
                self._set_motor_speeds_api(left, right)
        else:
            # Use API if no direct control
            self._set_motor_speeds_api(left, right)

    def _set_motor_speeds_api(self, left: int, right: int):
        """Fallback API motor control"""
        data = {
            "left_speed": left,
            "right_speed": right
        }
        result = self.api_request('POST', '/motor/control', data)
        if result and result.get('success'):
            if left != 0 or right != 0:
                logger.debug(f"Motors (API): L={left:4d}, R={right:4d}")

    def stop_motors(self):
        """Stop all motors"""
        if MOTOR_DIRECT and motor_controller:
            try:
                motor_controller.emergency_stop()
                logger.info("Motors stopped (direct)")
                self.state.motors_stopped = True
                return
            except Exception as e:
                logger.error(f"Direct motor stop error: {e}")

        # Fallback to API
        result = self.api_request('POST', '/motor/stop', {"reason": "controller_stop"})
        if result:
            logger.info("Motors stopped (API)")
        self.state.motors_stopped = True

    def emergency_stop(self):
        """Emergency stop - try both direct and API"""
        logger.warning("EMERGENCY STOP activated")

        # Try direct first for fastest response
        if MOTOR_DIRECT and motor_controller:
            try:
                motor_controller.emergency_stop()
            except:
                pass

        # Also send via API to ensure all systems stop
        self.api_request('POST', '/motor/stop', {"reason": "emergency"})
        self.state.motors_stopped = True

    def control_camera_pan(self):
        """Control camera pan via API with smooth movement"""
        # Lower threshold for more precise control
        if abs(self.state.right_x) > 0.15:
            # Calculate target angle
            target_pan = int(90 - (self.state.right_x * 180))  # Full 360 degree range
            target_pan = max(-90, min(270, target_pan))  # Match extended PWM limits

            # Initialize or update with smoother transitions
            if not hasattr(self, '_current_pan_angle'):
                self._current_pan_angle = 90  # Start at center

            # Only update if change is significant (reduce jitter)
            if abs(target_pan - self._current_pan_angle) > 3:
                # Smooth interpolation - move partway to target
                smooth_factor = 0.3  # Adjust for smoother/faster response
                delta = int((target_pan - self._current_pan_angle) * smooth_factor)

                # Ensure minimum movement to avoid stuck positions
                if delta != 0 or abs(target_pan - self._current_pan_angle) > 10:
                    if delta == 0:
                        delta = 1 if target_pan > self._current_pan_angle else -1

                    self._current_pan_angle += delta
                    self._current_pan_angle = max(-90, min(270, self._current_pan_angle))

                    # Send with smooth speed parameter
                    data = {"pan": self._current_pan_angle, "smooth": True}
                    try:
                        self.api_request('POST', '/camera/pantilt', data)
                    except:
                        pass

    def control_camera_tilt(self):
        """Control camera tilt via API with smooth movement"""
        # Manual control only - no auto-centering
        if abs(self.state.right_y) > 0.15:  # Lower threshold for more control
            # Calculate target angle
            target_tilt = int(90 + (self.state.right_y * 90))
            target_tilt = max(0, min(180, target_tilt))

            # Initialize or update with smoother transitions
            if not hasattr(self, '_current_tilt_angle'):
                self._current_tilt_angle = 90  # Start at center

            # Only update if change is significant
            if abs(target_tilt - self._current_tilt_angle) > 3:
                # Smooth interpolation
                smooth_factor = 0.3
                delta = int((target_tilt - self._current_tilt_angle) * smooth_factor)

                # Ensure minimum movement
                if delta != 0 or abs(target_tilt - self._current_tilt_angle) > 10:
                    if delta == 0:
                        delta = 1 if target_tilt > self._current_tilt_angle else -1

                    self._current_tilt_angle += delta
                    self._current_tilt_angle = max(0, min(180, self._current_tilt_angle))

                    # Send with smooth speed parameter
                    data = {"tilt": self._current_tilt_angle, "smooth": True}
                    try:
                        self.api_request('POST', '/camera/pantilt', data)
                    except:
                        pass

    def dispense_treat(self):
        """Dispense treat via API"""
        logger.info("LB pressed: Dispensing treat")
        data = {
            "dog_id": "xbox_test",
            "reason": "manual_xbox",
            "count": 1
        }
        result = self.api_request('POST', '/treat/dispense', data)
        if result and result.get('success'):
            logger.info("Treat dispensed!")

    def take_photo(self):
        """Take photo via API"""
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_cooldown:
            return

        logger.info("RB pressed: Taking photo")
        self.last_photo_time = current_time

        result = self.api_request('POST', '/camera/photo')
        if result and result.get('success'):
            logger.info(f"Photo saved: {result.get('filename')} ({result.get('resolution')})")

    def play_sound_effect(self):
        """Play sound effect via API"""
        track_num, track_name = self.SOUND_TRACKS[self.current_sound_index]
        logger.info(f"Playing {track_name} (track #{track_num})")

        # Play by track number directly - more reliable
        data = {"number": track_num}
        result = self.api_request('POST', '/audio/play/number', data)
        if result and result.get('success'):
            logger.info(f"Now playing: {track_name}")
        else:
            logger.error(f"Failed to play {track_name}")

    def update_camera_smooth(self):
        """Continuous camera update for smooth movement"""
        if self.running:
            # Update pan and tilt based on current joystick state
            self.control_camera_pan()
            self.control_camera_tilt()

            # Schedule next update
            self.camera_timer = Timer(self.camera_update_interval, self.update_camera_smooth)
            self.camera_timer.daemon = True
            self.camera_timer.start()

    def run(self):
        """Main control loop"""
        if not self.connect():
            logger.error("Failed to connect to controller")
            return

        self.running = True
        logger.info("Xbox Hybrid controller ready!")
        logger.info("Movement: Left stick (DIRECT control for low latency)")
        logger.info("Controls: LB=treat, RB=photo, Y=sound, A=emergency stop")
        logger.info("D-pad left/right to select sounds, D-pad down to play")

        # Start smooth camera update timer
        self.update_camera_smooth()

        try:
            while self.running and not self.stop_event.is_set():
                event = self.read_event()
                if not event:
                    continue

                timestamp, value, event_type, number = event

                # Process based on event type
                if event_type == 0x01:  # Button event
                    pressed = (value == 1)
                    self.process_button(number, pressed)

                elif event_type == 0x02:  # Axis event
                    if number in [6, 7]:  # D-pad
                        self.process_dpad(number, value)
                    else:
                        self.process_axis(number, value)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Controller error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False

        # Stop camera timer
        if self.camera_timer:
            self.camera_timer.cancel()

        # Stop motors
        self.stop_motors()

        # Clean up motor controller
        if MOTOR_DIRECT and motor_controller:
            try:
                motor_controller.cleanup()
            except:
                pass

        # Close device
        if self.device:
            self.device.close()

        # Close API session
        self.session.close()

        logger.info("Xbox controller disconnected")

    def stop(self):
        """Stop the controller"""
        self.stop_event.set()


def main():
    """Main entry point"""
    # Check for joystick device
    js_device = '/dev/input/js0'
    if not os.path.exists(js_device):
        logger.error(f"No joystick at {js_device}")
        logger.info("Make sure Xbox controller is connected and run:")
        logger.info("  sudo ./fix_xbox_controller.sh")
        return

    controller = XboxHybridController(js_device)

    try:
        controller.run()
    except Exception as e:
        logger.error(f"Controller failed: {e}")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()