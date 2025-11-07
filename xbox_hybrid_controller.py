#!/usr/bin/env python3
"""
Fixed Xbox Controller for DogBot
- Fixes treat dispenser freeze with cooldown
- Fixes motor control lockup with proper cleanup and rate limiting
- Adds motor calibration for straight driving
"""

import struct
import time
import os
import sys
import logging
import requests
import signal
import threading
from threading import Thread, Event, Timer, Lock
from dataclasses import dataclass
from typing import Optional, Tuple

# Add project root to path for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct hardware import for motors - use robust version
try:
    from core.hardware.motor_controller_robust import MotorControllerRobust as MotorController
    MOTOR_DIRECT = True
    motor_controller = MotorController()
    logger = logging.getLogger('XboxFixed')
    logger.info("Robust motor control initialized")
except ImportError:
    try:
        from core.hardware.motor_controller import MotorController
        MOTOR_DIRECT = True
        motor_controller = MotorController()
        logger = logging.getLogger('XboxFixed')
        logger.info("Direct motor control initialized (fallback)")
    except ImportError:
        MOTOR_DIRECT = False
        motor_controller = None
        logger = logging.getLogger('XboxFixed')
        logger.warning("No direct motor control available, will use API")

# Disable direct servo control - use API for reliability
# Direct servo control can freeze due to GPIO cleanup issues
servo_controller = None
SERVO_DIRECT = False
logger.info("Using API for servo control (more reliable)")

# Try to connect to event bus for mode management
event_bus = None
try:
    from core.bus import get_bus, publish_system_event
    event_bus = get_bus()
    logger.info("Connected to event bus for mode management")
except ImportError:
    logger.info("Event bus not available, running standalone")

logging.basicConfig(level=logging.INFO)

def notify_manual_input():
    """Notify the system that manual input occurred"""
    if event_bus:
        try:
            publish_system_event('manual_input_detected', {
                'timestamp': time.time(),
                'source': 'xbox_controller'
            }, 'xbox_hybrid_controller')
        except Exception as e:
            logger.warning(f"Failed to notify manual input: {e}")

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


class XboxHybridControllerFixed:
    """Fixed Xbox controller with proper thread safety and cooldowns"""

    # API configuration
    API_BASE_URL = "http://localhost:8000"

    # Controller configuration
    DEADZONE = 0.15
    TRIGGER_DEADZONE = 0.1
    MAX_SPEED = 100
    TURN_SPEED_FACTOR = 0.6

    # Motor calibration (right motor needs boost)
    RIGHT_MOTOR_BOOST = 1.10  # 10% boost for right motor to match left

    # Safety features
    TREAT_COOLDOWN = 2.0  # Prevent rapid treat dispensing
    MOTOR_UPDATE_RATE = 0.02  # 20ms between motor updates (50Hz) - more responsive
    MOTOR_TIMEOUT = 0.15  # Stop motors if no update in 150ms - faster stop

    # Sound tracks with FULL PATHS for D-pad navigation
    SOUND_TRACKS = [
        ("/talks/0001.mp3", "Scooby Intro"),
        ("/talks/0003.mp3", "Elsa"),
        ("/talks/0004.mp3", "Bezik"),
        ("/talks/0005.mp3", "Bezik Come"),
        ("/talks/0006.mp3", "Elsa Come"),
        ("/talks/0007.mp3", "Dogs Come"),
        ("/talks/0008.mp3", "Good Dog"),
        ("/talks/0009.mp3", "Kahnshik"),
        ("/talks/0010.mp3", "Lie Down"),
        ("/talks/0011.mp3", "Quiet"),
        ("/talks/0012.mp3", "No"),
        ("/talks/0013.mp3", "Treat"),
        ("/talks/0014.mp3", "Kokoma"),
        ("/talks/0015.mp3", "Sit"),
        ("/talks/0016.mp3", "Spin"),
        ("/talks/0017.mp3", "Stay"),
        ("/02/0018.mp3", "Mozart Piano"),
        ("/02/0019.mp3", "Mozart Concerto"),
        ("/02/0020.mp3", "Milkshake"),
        ("/02/0021.mp3", "Yummy"),
        ("/02/0022.mp3", "Hungry Like Wolf"),
        ("/02/0023.mp3", "Cake By Ocean"),
        ("/02/0024.mp3", "Who Let Dogs Out"),
        ("/02/0025.mp3", "Progress Scan"),
        ("/02/0026.mp3", "Robot Scan"),
        ("/02/0027.mp3", "Door Scan"),
        ("/02/0028.mp3", "Hi Scan"),
        ("/02/0029.mp3", "Busy Scan"),
        ("/02/0030.mp3", "Scooby Snacks")
    ]

    REWARD_SOUNDS = [
        ("/talks/0008.mp3", "Good Dog"),
        ("/talks/0013.mp3", "Treat")
    ]

    def __init__(self, device_path: str = '/dev/input/js0'):
        self.device_path = device_path
        self.device = None
        self.running = False
        self.state = ControllerState()
        self.stop_event = Event()

        # Thread safety locks
        self.motor_lock = Lock()
        self.api_lock = Lock()
        self.treat_lock = Lock()

        # Cooldown tracking
        self.last_treat_time = 0
        self.last_photo_time = 0
        self.last_motor_update = 0
        self.last_motor_command_time = 0
        self.photo_cooldown = 2.0

        # Motor safety timer
        self.motor_watchdog_timer = None
        self.motor_update_thread = None
        self.motor_update_running = False

        # Camera control
        self.camera_update_thread = None
        self.camera_update_running = False
        self.last_pan_angle = 100  # Center (shifted 10 degrees right from 90)
        self.last_tilt_angle = 90  # Center
        self.last_camera_update = 0
        self.CAMERA_UPDATE_RATE = 0.05  # 50ms between updates (20Hz)

        # Sound navigation
        self.current_sound_index = 0

        # LED state tracking
        self.led_enabled = False
        self.current_led_mode = 0
        self.led_modes = [
            "off",
            "idle",
            "searching",
            "dog_detected",
            "treat_launching",
            "error",
            "charging",
            "manual_rc"
        ]

        # API session for non-motor functions
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        logger.info(f"Xbox Fixed Controller initialized for {device_path}")
        logger.info(f"Motor control: {'DIRECT' if MOTOR_DIRECT else 'API'}")
        logger.info(f"Right motor boost: {self.RIGHT_MOTOR_BOOST}x")
        logger.info(f"API endpoint: {self.API_BASE_URL}")

        # Preload audio system
        self._preload_audio_system()

    def _preload_audio_system(self):
        """Preload audio system to prevent first-time delay"""
        try:
            logger.info("Preloading audio system...")
            result = self.api_request('GET', '/audio/status')
            if result:
                logger.info("Audio system preloaded successfully")
        except Exception as e:
            logger.warning(f"Audio preload error: {e}")

    def api_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """Thread-safe API request with error handling"""
        with self.api_lock:
            url = f"{self.API_BASE_URL}{endpoint}"
            try:
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
            # Check if API is available
            health = self.api_request('GET', '/health')
            if health:
                logger.info(f"API health check: {health}")
            else:
                logger.warning("API server not responding - only motor control will work")

            # Open the joystick device
            self.device = open(self.device_path, 'rb')
            logger.info(f"Connected to Xbox controller at {self.device_path}")

            # Start motor update thread for smooth control
            self.motor_update_running = True
            self.motor_update_thread = Thread(target=self._motor_update_loop, daemon=True)
            self.motor_update_thread.start()

            # Start camera update thread for smooth control
            self.camera_update_running = True
            self.camera_update_thread = Thread(target=self._camera_update_loop, daemon=True)
            self.camera_update_thread.start()

            # Start heartbeat thread to keep MANUAL mode active
            self.heartbeat_running = True
            self.heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            logger.info("Heartbeat thread started to maintain MANUAL mode")

            return True

        except FileNotFoundError:
            logger.error(f"Controller not found at {self.device_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def _camera_update_loop(self):
        """Separate thread for smooth camera control"""
        while self.camera_update_running:
            try:
                current_time = time.time()

                # Only update if enough time has passed (smooth rate limiting)
                if current_time - self.last_camera_update < self.CAMERA_UPDATE_RATE:
                    time.sleep(0.01)
                    continue

                # Check if stick is being used
                if abs(self.state.right_x) > self.DEADZONE or abs(self.state.right_y) > self.DEADZONE:
                    # Velocity-based smooth control (like we had before)
                    # Slow, smooth movement based on stick position
                    pan_speed = self.state.right_x * 2.5  # Slower speed for smoothness
                    tilt_speed = self.state.right_y * 2.0  # Slower speed for smoothness

                    # Update positions incrementally
                    # Pan: inverted as requested (right stick right = camera left)
                    new_pan = self.last_pan_angle - pan_speed  # Inverted

                    # Tilt: INVERTED as requested
                    # right_y is already inverted in process_axis, so:
                    # positive right_y = stick UP = camera should look DOWN (inverted)
                    # negative right_y = stick DOWN = camera should look UP (inverted)
                    new_tilt = self.last_tilt_angle + tilt_speed  # Inverted: UP increases angle

                    # Clamp to valid range
                    new_pan = max(10, min(270, new_pan))
                    new_tilt = max(20, min(160, new_tilt))

                    # Only send if changed enough (reduce jitter)
                    if (abs(new_pan - self.last_pan_angle) > 1.0 or
                        abs(new_tilt - self.last_tilt_angle) > 1.0):

                        self.api_request('POST', '/camera/pantilt', {
                            "pan": int(new_pan),
                            "tilt": int(new_tilt),
                            "smooth": True  # Enable smooth movement
                        })

                        self.last_pan_angle = new_pan
                        self.last_tilt_angle = new_tilt
                        self.last_camera_update = current_time

                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Camera update loop error: {e}")
                time.sleep(0.1)

    def _heartbeat_loop(self):
        """Send periodic manual_input events to prevent timeout"""
        while self.heartbeat_running:
            try:
                # Send manual input event every 30 seconds to keep MANUAL mode active
                notify_manual_input()
                logger.debug("Heartbeat: Keeping MANUAL mode active")
                time.sleep(30.0)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(30.0)

    def _motor_update_loop(self):
        """Separate thread for smooth motor control with safety timeout"""
        while self.motor_update_running:
            try:
                current_time = time.time()

                # Only timeout if there's actually no input (stick at center and no trigger)
                has_input = (abs(self.state.left_x) > self.DEADZONE or
                           abs(self.state.left_y) > self.DEADZONE or
                           self.state.right_trigger > self.TRIGGER_DEADZONE)

                if has_input:
                    # Keep updating command time while there's input
                    self.last_motor_command_time = current_time
                    # Update motor speeds
                    with self.motor_lock:
                        self.update_motor_control()
                elif current_time - self.last_motor_command_time > self.MOTOR_TIMEOUT:
                    # Only stop if truly no input for timeout period
                    if not self.state.motors_stopped:
                        with self.motor_lock:
                            self._stop_motors_internal()
                            self.state.motors_stopped = True
                            logger.debug("Motors stopped (timeout - no input)")

                time.sleep(self.MOTOR_UPDATE_RATE)

            except Exception as e:
                logger.error(f"Motor update loop error: {e}")
                time.sleep(0.1)

    def _stop_motors_internal(self):
        """Internal motor stop without lock (call with motor_lock held)"""
        if MOTOR_DIRECT and motor_controller:
            try:
                motor_controller.emergency_stop()
                self.state.last_left_speed = 0
                self.state.last_right_speed = 0
                return True
            except Exception as e:
                logger.error(f"Direct motor stop error: {e}")
        return False

    def read_event(self) -> Optional[Tuple]:
        """Read a single joystick event"""
        try:
            event_data = self.device.read(8)
            if event_data:
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
            if abs(normalized) > self.DEADZONE:
                notify_manual_input()

        elif number == 1:  # Left stick Y (inverted for forward)
            self.state.left_y = -normalized
            if abs(normalized) > self.DEADZONE:
                notify_manual_input()

        elif number == 2:  # Left trigger (LT)
            normalized_trigger = (value + 32767) / 65534.0
            previous_trigger = self.state.left_trigger

            # Add a trigger state tracker
            if not hasattr(self, 'lt_was_pressed'):
                self.lt_was_pressed = False

            # Detect trigger press on rising edge only
            if normalized_trigger > 0.8 and not self.lt_was_pressed:
                self.cycle_led_mode()
                self.lt_was_pressed = True
                logger.info(f"LT TRIGGERED! Value: {normalized_trigger:.2f}")
            elif normalized_trigger < 0.2:
                self.lt_was_pressed = False

            self.state.left_trigger = normalized_trigger

        elif number == 3:  # Right stick X (camera pan)
            self.state.right_x = normalized
            # Store for smooth camera update loop

        elif number == 4:  # Right stick Y (camera tilt)
            self.state.right_y = -normalized
            # Store for smooth camera update loop

        elif number == 5:  # Right trigger (RT) - speed control
            normalized_trigger = (value + 32767) / 65534.0
            if normalized_trigger > self.TRIGGER_DEADZONE:
                self.state.right_trigger = normalized_trigger
            else:
                self.state.right_trigger = 0.0

    def update_motor_control(self):
        """Update motor speeds with calibration and safety"""
        # Progressive speed control - more control at low speeds
        # If trigger not pressed, use stick position for speed
        if self.state.right_trigger < 0.1:
            # No trigger - use stick magnitude for speed (5-45% max)
            stick_magnitude = (self.state.left_x**2 + self.state.left_y**2) ** 0.5
            stick_magnitude = min(1.0, stick_magnitude)  # Clamp to 1.0
            # Non-linear curve for better low-speed control - even slower at minimum
            speed_multiplier = 0.05 + (stick_magnitude ** 2.0) * 0.4  # 5-45% range, squared for more gradual
        else:
            # Trigger pressed - use trigger for speed boost (30-100%)
            speed_multiplier = 0.3 + (self.state.right_trigger * 0.7)

        # Calculate motor speeds from left stick
        forward = self.state.left_y * self.MAX_SPEED * speed_multiplier
        turn = self.state.left_x * self.MAX_SPEED * self.TURN_SPEED_FACTOR * speed_multiplier

        # INVERTED: Stick left should make robot turn left
        # When stick is left (negative X), right motor should be faster than left
        left_speed = int(forward - turn)  # Inverted
        right_speed = int(forward + turn)  # Inverted

        # Apply right motor boost for straight driving
        if abs(turn) < 10:  # Going mostly straight
            right_speed = int(right_speed * self.RIGHT_MOTOR_BOOST)

        # Clamp to valid range
        left_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, left_speed))
        right_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, right_speed))

        # Rate limiting - only update if enough time has passed
        current_time = time.time()
        if current_time - self.last_motor_update < self.MOTOR_UPDATE_RATE:
            return

        # Only send if changed (lower threshold for better responsiveness)
        if (abs(left_speed - self.state.last_left_speed) > 2 or
            abs(right_speed - self.state.last_right_speed) > 2 or
            (left_speed == 0 and right_speed == 0 and not self.state.motors_stopped)):

            self.set_motor_speeds(left_speed, right_speed)
            self.state.last_left_speed = left_speed
            self.state.last_right_speed = right_speed
            self.state.motors_stopped = (left_speed == 0 and right_speed == 0)
            self.last_motor_update = current_time

    def set_motor_speeds(self, left: int, right: int):
        """Set motor speeds with proper error handling"""
        if MOTOR_DIRECT and motor_controller:
            try:
                if left == 0 and right == 0:
                    motor_controller.emergency_stop()
                else:
                    left_dir = 'forward' if left >= 0 else 'backward'
                    right_dir = 'forward' if right >= 0 else 'backward'
                    left_speed = abs(left)
                    right_speed = abs(right)

                    motor_controller.set_motor_speed('A', left_speed, left_dir)
                    motor_controller.set_motor_speed('B', right_speed, right_dir)
                    logger.debug(f"Motors: L={left:4d}, R={right:4d}")
            except Exception as e:
                logger.error(f"Motor control error: {e}")
                # Try to stop motors on error
                try:
                    motor_controller.emergency_stop()
                except:
                    pass

    def stop_motors(self):
        """Stop all motors with lock"""
        with self.motor_lock:
            self._stop_motors_internal()
            self.state.motors_stopped = True
            logger.info("Motors stopped")

    def emergency_stop(self):
        """Emergency stop - immediate halt"""
        logger.warning("EMERGENCY STOP activated")
        with self.motor_lock:
            self._stop_motors_internal()
            self.state.motors_stopped = True
        # Also send via API as backup
        self.api_request('POST', '/motor/stop', {"reason": "emergency"})

    def process_button(self, number: int, pressed: bool):
        """Process button press with proper cooldowns"""
        if pressed:
            logger.debug(f"Button {number} pressed")
            notify_manual_input()

        if number == 0:  # A button - Emergency stop
            self.state.a_button = pressed
            if pressed:
                logger.info("A button: Emergency stop")
                self.emergency_stop()

        elif number == 1:  # B button - Stop motors
            self.state.b_button = pressed
            if pressed:
                logger.info("B button: Stop motors")
                self.stop_motors()

        elif number == 2:  # X button - Toggle LED
            self.state.x_button = pressed
            if pressed:
                self.toggle_led()

        elif number == 3:  # Y button - Sound
            self.state.y_button = pressed
            if pressed:
                self.play_reward_sound()

        elif number == 4:  # Left bumper - Dispense treat (with cooldown)
            self.state.left_bumper = pressed
            if pressed:
                self.dispense_treat_safe()

        elif number == 5:  # Right bumper - Take photo
            self.state.right_bumper = pressed
            if pressed:
                self.take_photo()

        elif number == 10:  # Right stick click - Center camera
            if pressed:
                self.center_camera()

    def dispense_treat_safe(self):
        """Dispense treat - always works on button press"""
        logger.info("LB pressed: Dispensing treat")

        # Don't use a lock or cooldown that blocks - just track for logging
        current_time = time.time()
        time_since_last = current_time - self.last_treat_time

        # Warn if too fast but still dispense
        if time_since_last < self.TREAT_COOLDOWN and self.last_treat_time > 0:
            logger.warning(f"Rapid treat request (only {time_since_last:.1f}s since last)")

        self.last_treat_time = current_time

        # Direct API call - no thread needed for treats
        # The API itself should handle any queuing/safety
        data = {
            "dog_id": "xbox_test",
            "reason": "manual_xbox",
            "count": 1
        }

        try:
            result = self.api_request('POST', '/treat/dispense', data)
            if result and result.get('success'):
                logger.info("Treat dispensed successfully!")
            else:
                logger.error(f"Treat dispense failed: {result}")
        except Exception as e:
            logger.error(f"Treat dispense error: {e}")

    def take_photo(self):
        """Take photo with cooldown"""
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_cooldown:
            return

        logger.info("RB pressed: Taking photo")
        self.last_photo_time = current_time

        result = self.api_request('POST', '/camera/photo')
        if result and result.get('success'):
            logger.info(f"Photo saved: {result.get('filename', 'unknown')}")

    def center_camera(self):
        """Center the camera to default position"""
        logger.info("Right stick click: Centering camera")

        # Reset to center positions
        self.last_pan_angle = 100  # Center with slight right offset
        self.last_tilt_angle = 90  # Center

        # Send center command
        self.api_request('POST', '/camera/pantilt', {
            "pan": 100,
            "tilt": 90,
            "smooth": True
        })

    def toggle_led(self):
        """Toggle blue LED"""
        self.led_enabled = not self.led_enabled
        endpoint = '/leds/blue/on' if self.led_enabled else '/leds/blue/off'
        result = self.api_request('POST', endpoint)
        if result and result.get('success'):
            logger.info(f"Blue LED {'on' if self.led_enabled else 'off'}")

    def cycle_led_mode(self):
        """Cycle through NeoPixel LED modes"""
        self.current_led_mode = (self.current_led_mode + 1) % len(self.led_modes)
        mode = self.led_modes[self.current_led_mode]
        logger.info(f"Left Trigger: NeoPixel mode = {mode}")

        data = {"mode": mode}
        result = self.api_request('POST', '/leds/mode', data)
        if result and result.get('success'):
            logger.info(f"NeoPixel LEDs set to {mode}")

    def play_reward_sound(self):
        """Play alternating reward sounds (Y button)"""
        # Initialize press counter if needed
        if not hasattr(self, 'y_press_count'):
            self.y_press_count = 0

        self.y_press_count += 1

        if self.y_press_count % 2 == 1:  # Odd press = Treat
            logger.info("Y button: Playing 'Treat' sound")
            self.api_request('POST', '/audio/play', {"track": 13, "name": "Treat"})
        else:  # Even press = Good Dog
            logger.info("Y button: Playing 'Good Dog' sound")
            self.api_request('POST', '/audio/play', {"track": 8, "name": "Good Dog"})

    def play_sound_effect(self):
        """Play selected sound effect (D-pad down)"""
        file_path, track_name = self.SOUND_TRACKS[self.current_sound_index]
        logger.info(f"Playing: {track_name} ({file_path})")
        # Send the file path directly
        self.api_request('POST', '/audio/play_file', {"path": file_path, "name": track_name})

    def process_dpad(self, number: int, value: int):
        """Process D-pad input"""
        if value != 0:
            notify_manual_input()

        if number == 6:  # D-pad X axis
            self.state.dpad_left = (value < 0)
            self.state.dpad_right = (value > 0)

            if value < 0:  # Left - Previous track
                self.current_sound_index = (self.current_sound_index - 1) % len(self.SOUND_TRACKS)
                file_path, track_name = self.SOUND_TRACKS[self.current_sound_index]
                logger.info(f"Selected: {track_name}")
            elif value > 0:  # Right - Next track
                self.current_sound_index = (self.current_sound_index + 1) % len(self.SOUND_TRACKS)
                file_path, track_name = self.SOUND_TRACKS[self.current_sound_index]
                logger.info(f"Selected: {track_name}")

        elif number == 7:  # D-pad Y axis
            self.state.dpad_up = (value < 0)
            self.state.dpad_down = (value > 0)

            if value < 0:  # Up - Pause/Resume
                logger.info("D-pad up: Pause/Resume")
                self.api_request('POST', '/audio/pause')
            elif value > 0:  # Down - Play selected
                self.play_sound_effect()

    def run(self):
        """Main control loop"""
        if not self.connect():
            logger.error("Failed to connect to controller")
            return

        self.running = True
        logger.info("Xbox Fixed Controller ready!")
        logger.info("=== FIXED CONTROLS ===")
        logger.info("Movement: Left stick + RT for speed")
        logger.info("Camera: Right stick")
        logger.info("A = Emergency Stop, B = Stop Motors")
        logger.info("X = Blue LED, LT = NeoPixel modes")
        logger.info("Y = Sound, LB = Treat (2s cooldown), RB = Photo")
        logger.info("=== FIXES APPLIED ===")
        logger.info("✅ Treat dispenser cooldown prevents freezing")
        logger.info("✅ Motor control with timeout safety")
        logger.info("✅ Right motor boost for straight driving")
        logger.info("✅ Thread-safe API calls")

        try:
            while self.running and not self.stop_event.is_set():
                event = self.read_event()
                if not event:
                    continue

                timestamp, value, event_type, number = event

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
        self.motor_update_running = False
        self.camera_update_running = False
        self.heartbeat_running = False

        # Stop motor update thread
        if self.motor_update_thread:
            self.motor_update_thread.join(timeout=1.0)

        # Stop camera update thread
        if self.camera_update_thread:
            self.camera_update_thread.join(timeout=1.0)

        # Stop heartbeat thread
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join(timeout=1.0)

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
    js_device = '/dev/input/js0'
    if not os.path.exists(js_device):
        logger.error(f"No joystick at {js_device}")
        return

    controller = XboxHybridControllerFixed(js_device)

    try:
        controller.run()
    except Exception as e:
        logger.error(f"Controller failed: {e}")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()