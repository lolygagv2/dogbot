#!/usr/bin/env python3
"""
Xbox Controller for DogBot using REST API
Uses HTTP requests to control the robot instead of direct service imports
"""

import struct
import time
import os
import logging
import requests
from threading import Thread, Event
from dataclasses import dataclass
from typing import Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('XboxAPI')

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

class XboxAPIController:
    """Xbox controller using REST API for robot control"""

    # API configuration
    API_BASE_URL = "http://localhost:8000"

    # Controller configuration
    DEADZONE = 0.15
    TRIGGER_DEADZONE = 0.1
    MAX_SPEED = 100
    TURN_SPEED_FACTOR = 0.6

    # Sound effects list (D-pad navigation)
    SOUND_EFFECTS = [
        "success", "bark", "whistle", "celebrate",
        "startup", "shutdown", "alert", "reward"
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

        # API session
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        logger.info(f"Xbox API Controller initialized for {device_path}")
        logger.info(f"API endpoint: {self.API_BASE_URL}")

    def api_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """Make API request with error handling"""
        url = f"{self.API_BASE_URL}{endpoint}"
        try:
            if method == 'GET':
                response = self.session.get(url, timeout=1.0)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=1.0)
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
            # First check if API is available
            health = self.api_request('GET', '/health')
            if not health:
                logger.error("API server not responding! Start with: uvicorn api.server:app --host 0.0.0.0 --port 8000")
                return False

            logger.info(f"API health check: {health}")

            # Open the joystick device
            self.device = open(self.device_path, 'rb')
            logger.info(f"Connected to Xbox controller at {self.device_path}")

            # Set to manual control mode
            result = self.api_request('POST', '/manual/mode/manual')
            if result:
                logger.info("Set to manual control mode")

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
            self.control_camera_pan()
        elif number == 4:  # Right stick Y (camera tilt)
            self.state.right_y = -normalized
            self.control_camera_tilt()
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

            if value < 0:  # Left - Previous sound
                self.current_sound_index = (self.current_sound_index - 1) % len(self.SOUND_EFFECTS)
                logger.info(f"Selected sound: {self.SOUND_EFFECTS[self.current_sound_index]}")
            elif value > 0:  # Right - Next sound
                self.current_sound_index = (self.current_sound_index + 1) % len(self.SOUND_EFFECTS)
                logger.info(f"Selected sound: {self.SOUND_EFFECTS[self.current_sound_index]}")

        elif number == 7:  # D-pad Y axis
            self.state.dpad_up = (value < 0)
            self.state.dpad_down = (value > 0)

            if value < 0:  # Up - Audio off
                logger.info("D-pad up: Audio off")
                # Would need audio control endpoint
            elif value > 0:  # Down - Play selected sound
                logger.info(f"D-pad down: Play {self.SOUND_EFFECTS[self.current_sound_index]}")
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
        """Send motor speed command via API"""
        data = {
            "left_speed": left,
            "right_speed": right
        }
        result = self.api_request('POST', '/motor/control', data)
        if result and result.get('success'):
            if left != 0 or right != 0:
                logger.debug(f"Motors: L={left:4d}, R={right:4d}")
        else:
            logger.error("Failed to set motor speeds")

    def stop_motors(self):
        """Stop all motors"""
        result = self.api_request('POST', '/motor/stop', {"reason": "controller_stop"})
        if result:
            logger.info("Motors stopped")
            self.state.motors_stopped = True

    def emergency_stop(self):
        """Emergency stop via API"""
        result = self.api_request('POST', '/motor/stop', {"reason": "emergency"})
        if result:
            logger.warning("EMERGENCY STOP activated")
            self.state.motors_stopped = True

    def control_camera_pan(self):
        """Control camera pan based on right stick X"""
        if abs(self.state.right_x) > self.DEADZONE:
            # Convert -1 to 1 range to 0-180 degrees
            pan_angle = int(90 + (self.state.right_x * 90))
            pan_angle = max(0, min(180, pan_angle))

            data = {"pan": pan_angle, "speed": 5}
            self.api_request('POST', '/camera/pantilt', data)

    def control_camera_tilt(self):
        """Control camera tilt based on right stick Y"""
        if abs(self.state.right_y) > self.DEADZONE:
            # Convert -1 to 1 range to 0-180 degrees
            tilt_angle = int(90 + (self.state.right_y * 90))
            tilt_angle = max(0, min(180, tilt_angle))

            data = {"tilt": tilt_angle, "speed": 5}
            self.api_request('POST', '/camera/pantilt', data)

    def dispense_treat(self):
        """Dispense a treat via API"""
        logger.info("LB pressed: Dispensing treat")
        data = {
            "dog_id": "xbox_test",
            "reason": "manual_xbox",
            "count": 1
        }
        result = self.api_request('POST', '/treat/dispense', data)
        if result and result.get('success'):
            logger.info("Treat dispensed!")
        else:
            logger.error("Failed to dispense treat")

    def take_photo(self):
        """Take a photo via API"""
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_cooldown:
            return

        logger.info("RB pressed: Taking photo")
        self.last_photo_time = current_time

        result = self.api_request('POST', '/camera/photo')
        if result and result.get('success'):
            logger.info(f"Photo saved: {result.get('filename')} ({result.get('resolution')})")
        else:
            logger.error("Failed to capture photo")

    def play_sound_effect(self):
        """Play selected sound effect via API"""
        sound_name = self.SOUND_EFFECTS[self.current_sound_index]
        logger.info(f"Y button: Playing sound '{sound_name}'")

        # Try audio play sound endpoint
        data = {"sound_name": sound_name}
        result = self.api_request('POST', '/audio/play/sound', data)
        if result and result.get('success'):
            logger.info(f"Playing: {sound_name}")
        else:
            # If that fails, try by number (1-8 for the sounds)
            data = {"number": self.current_sound_index + 1}
            result = self.api_request('POST', '/audio/play/number', data)
            if result:
                logger.info(f"Playing sound #{self.current_sound_index + 1}")

    def run(self):
        """Main control loop"""
        if not self.connect():
            logger.error("Failed to connect to controller or API")
            return

        self.running = True
        logger.info("Xbox controller ready! Use left stick to move, right stick for camera")
        logger.info("Controls: LB=treat, RB=photo, Y=sound, A=emergency stop")
        logger.info("D-pad left/right to select sounds, D-pad down to play")

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

        # Stop motors
        self.stop_motors()

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

    controller = XboxAPIController(js_device)

    try:
        controller.run()
    except Exception as e:
        logger.error(f"Controller failed: {e}")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()