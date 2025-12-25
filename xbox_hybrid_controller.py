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
import subprocess
import atexit
import select
from threading import Thread, Event, Timer, Lock
from dataclasses import dataclass
from typing import Optional, Tuple
from queue import Queue, Empty
from collections import defaultdict

# Add project root to path for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import motor control modules but don't initialize yet
try:
    from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
    MOTOR_BUS_AVAILABLE = True
except ImportError:
    MOTOR_BUS_AVAILABLE = False

try:
    from core.hardware.proper_pid_motor_controller import MotorControllerPolling
    MOTOR_CONTROLLER_AVAILABLE = True
except ImportError:
    MOTOR_CONTROLLER_AVAILABLE = False

# Initialize after imports
motor_bus = None
motor_controller = None
MOTOR_DIRECT = False
logger = logging.getLogger('XboxFixed')

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

# CRITICAL SAFETY: Global emergency stop function
def global_emergency_stop():
    """Emergency stop all motors - works even if Python is frozen"""
    logger = logging.getLogger('XboxFixed')
    logger.critical("GLOBAL EMERGENCY STOP TRIGGERED")

    # CORRECT motor control pins from config/pins.py:
    # Direction: IN1=17, IN2=18, IN3=27, IN4=22
    # PWM Enable: ENA=13, ENB=19
    motor_pins = [17, 18, 27, 22, 13, 19]

    # Method 1: Try gpiozero (most reliable)
    try:
        from gpiozero import OutputDevice, PWMOutputDevice
        for pin in motor_pins:
            try:
                if pin in [13, 19]:  # PWM pins
                    d = PWMOutputDevice(pin)
                    d.value = 0
                else:
                    d = OutputDevice(pin)
                    d.off()
                d.close()
            except:
                pass
        logger.info("Motors stopped via gpiozero")
        return
    except:
        pass

    # Method 2: Try lgpio directly
    try:
        import lgpio
        h = lgpio.gpiochip_open(0)
        for pin in motor_pins:
            try:
                lgpio.gpio_claim_output(h, pin)
                lgpio.gpio_write(h, pin, 0)
            except:
                pass
        lgpio.gpiochip_close(h)
        logger.info("Motors stopped via lgpio")
        return
    except:
        pass

    # Method 3: Fallback to subprocess gpioset
    try:
        for pin in motor_pins:
            subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                          capture_output=True, timeout=0.1)
        logger.info("Motors stopped via gpioset")
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")

# CRITICAL SAFETY: Signal handlers
def signal_handler(sig, frame):
    """Handle termination signals safely"""
    logger = logging.getLogger('XboxFixed')
    logger.warning(f"Received signal {sig} - stopping motors")
    global_emergency_stop()
    sys.exit(0)

# Register signal handlers IMMEDIATELY for safety
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

# Register emergency stop on program exit
atexit.register(global_emergency_stop)

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

    # Controller configuration - SIMPLIFIED FOR RELIABILITY
    DEADZONE = 0.20  # 20% deadzone - larger for reliable stop detection
    TRIGGER_DEADZONE = 0.1
    MAX_SPEED = 75  # Capped at 75% for safety (no turbo mode)
    TURN_SPEED_FACTOR = 0.6  # Reduced for smoother turns

    # RPM Control - Convert speed percentages to RPM targets
    MAX_RPM = 150  # Conservative RPM target for smoother control
    USE_PID_CONTROL = True  # ENABLED - closed-loop control with encoder feedback

    # Motor calibration - RESET (let encoders handle balancing)
    LEFT_MOTOR_BOOST = 1.0  # No boost - PID controller should handle this

    # Safety features - FIXED FOR SMOOTH CONTROL
    TREAT_COOLDOWN = 2.0  # Prevent rapid treat dispensing
    MOTOR_UPDATE_RATE = 0.05  # 50ms between motor updates - prevent conflicts with motor bus
    MOTOR_TIMEOUT = 1.0  # Longer timeout to prevent jerky stops
    MOTOR_WATCHDOG_TIMEOUT = 0.5  # Emergency stop if no heartbeat for 0.5s
    CONNECTION_TIMEOUT = 1.0  # Emergency stop if controller disconnected for 1s

    # Sound tracks - dynamically populated from VOICEMP3/talks and VOICEMP3/songs folders
    # See _preload_audio_system() for dynamic scanning
    # Add new MP3 files to those folders and they'll be automatically discovered on startup

    REWARD_SOUNDS = [
        ("/talks/good_dog.mp3", "Good Dog"),
        ("/talks/treat.mp3", "Treat")
    ]

    def __init__(self, device_path: str = '/dev/input/js0'):
        self.device_path = device_path
        self.device = None
        self.running = False
        self.state = ControllerState()
        self.stop_event = Event()

        # Thread safety locks
        self.motor_lock = Lock()
        self.treat_lock = Lock()
        # REMOVED: api_lock - replaced with async queue (non-blocking)

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
        # REMOVED: Watchdog variables - using motor command bus safety instead

        # Camera control
        self.camera_update_thread = None
        self.camera_update_running = False
        self.last_pan_angle = 100  # Center (shifted 10 degrees right from 90)
        self.last_tilt_angle = 90  # Center
        self.last_camera_update = 0
        self.CAMERA_UPDATE_RATE = 0.05  # 50ms between updates (20Hz)

        # Sound navigation
        self.current_sound_index = 0
        self.last_dpad_time = 0
        self.dpad_cooldown = 0.3  # 300ms cooldown between D-pad presses

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
            "manual_rc",
            # New patterns for 165 LED strip
            "gradient_flow",
            "chase",
            "fire"
        ]

        # API session for non-motor functions
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        # ASYNC COMMAND QUEUE - prevents main loop blocking
        self.api_queue = Queue(maxsize=50)  # Bounded queue prevents memory issues
        self.api_worker_running = False
        self.api_worker_thread = None

        # Command debouncing - prevents rapid duplicate commands
        self.last_command_time = defaultdict(float)  # endpoint -> timestamp
        self.DEBOUNCE_TIME = 0.05  # 50ms minimum between same commands

        # Motor control system
        self.motor_bus = None
        self.motor_controller = None
        self.motor_direct = False

        # Initialize motor control
        self._initialize_motor_system()

        logger.info(f"Xbox Fixed Controller initialized for {device_path}")
        logger.info(f"Motor control: {'DIRECT' if self.motor_direct else 'API'}")
        logger.info(f"API endpoint: {self.API_BASE_URL}")

        # Preload audio system
        self._preload_audio_system()

        # DISABLED: Xbox watchdog conflicts with motor command bus watchdog
        # Motor command bus already provides safety with 5-second timeout
        # self._start_watchdog()

    def _initialize_motor_system(self):
        """Initialize motor control system with fallback options"""
        global motor_bus, motor_controller, MOTOR_DIRECT

        if MOTOR_BUS_AVAILABLE:
            try:
                self.motor_bus = get_motor_bus()
                if self.motor_bus.start():
                    self.motor_direct = True
                    motor_bus = self.motor_bus  # Update global reference
                    MOTOR_DIRECT = True
                    # CRITICAL FIX: Set motor controller reference
                    self.motor_controller = self.motor_bus.motor_controller
                    logger.info("✅ Motor command bus with polling encoders initialized")
                    return
                else:
                    logger.warning("Motor command bus failed to start")
                    self.motor_bus = None
            except Exception as e:
                logger.error(f"Motor bus initialization error: {e}")
                self.motor_bus = None

        # Use the motor controller from motor bus if available
        if self.motor_bus and hasattr(self.motor_bus, 'motor_controller'):
            self.motor_controller = self.motor_bus.motor_controller
            self.motor_direct = True
            motor_controller = self.motor_controller  # Update global reference
            MOTOR_DIRECT = True
            logger.info("✅ Using motor controller from motor bus for PID control")
            return

        if MOTOR_CONTROLLER_AVAILABLE:
            try:
                self.motor_controller = MotorControllerPolling()
                if self.motor_controller.initialize():
                    self.motor_direct = True
                    motor_controller = self.motor_controller  # Update global reference
                    MOTOR_DIRECT = True
                    logger.info("✅ DFRobot polling motor control initialized (fallback)")
                    return
                else:
                    logger.warning("DFRobot motor controller failed to initialize")
                    self.motor_controller = None
            except Exception as e:
                logger.error(f"Motor controller initialization error: {e}")
                self.motor_controller = None

        logger.warning("❌ No motor control available, will use API")
        self.motor_direct = False
        MOTOR_DIRECT = False

    def _scan_folder(self, folder_name):
        """Scan a folder for MP3 files and return list of (api_path, display_name) tuples"""
        import os
        import glob

        VOICEMP3_BASE = "/home/morgan/dogbot/VOICEMP3"
        folder_path = os.path.join(VOICEMP3_BASE, folder_name)
        tracks = []

        if os.path.exists(folder_path):
            mp3_files = sorted(glob.glob(os.path.join(folder_path, "*.mp3")))
            for mp3_path in mp3_files:
                filename = os.path.basename(mp3_path)
                # Create API path format: /talks/filename.mp3 or /songs/filename.mp3
                api_path = f"/{folder_name}/{filename}"
                # Create display name: remove .mp3, replace underscores with spaces, title case
                display_name = filename.replace(".mp3", "").replace("_", " ").title()
                tracks.append((api_path, display_name))

        return tracks

    def _scan_audio_folders(self):
        """Scan VOICEMP3 folders and update track lists. Can be called to refresh after new recordings."""
        # Scan talks and songs folders
        self.TALK_TRACKS = self._scan_folder("talks")
        self.SONG_TRACKS = self._scan_folder("songs")

        # Fallback if folders are empty
        if not self.TALK_TRACKS:
            self.TALK_TRACKS = [("/talks/treat.mp3", "Treat")]
            logger.warning("No talks found, using fallback")
        if not self.SONG_TRACKS:
            self.SONG_TRACKS = [("/songs/scooby_snacks.mp3", "Scooby Snacks")]
            logger.warning("No songs found, using fallback")

        # Reset indices if they exceed new list bounds
        if not hasattr(self, 'current_talk_index') or self.current_talk_index >= len(self.TALK_TRACKS):
            self.current_talk_index = 0
        if not hasattr(self, 'current_song_index') or self.current_song_index >= len(self.SONG_TRACKS):
            self.current_song_index = 0

        # Initialize queued track if not exists
        if not hasattr(self, 'queued_track'):
            self.queued_track = None
            self.queued_type = "talk"

    def _preload_audio_system(self):
        """Preload audio system and dynamically discover sound tracks"""
        try:
            logger.info("Preloading audio system...")
            result = self.api_request('GET', '/audio/status')
            if result:
                logger.info("Audio system preloaded successfully")
        except Exception as e:
            logger.warning(f"Audio preload error: {e}")

        # Scan folders for audio files
        self._scan_audio_folders()

        logger.info(f"Audio tracks discovered: {len(self.TALK_TRACKS)} talks, {len(self.SONG_TRACKS)} songs")
        logger.info(f"Talks: {[t[1] for t in self.TALK_TRACKS]}")
        logger.info(f"Songs: {[s[1] for s in self.SONG_TRACKS]}")

    def _start_api_worker(self):
        """Start the async API worker thread"""
        if self.api_worker_running:
            return
        self.api_worker_running = True
        self.api_worker_thread = Thread(target=self._api_worker_loop, daemon=True)
        self.api_worker_thread.start()
        logger.info("Async API worker started")

    def _api_worker_loop(self):
        """Worker thread that processes API commands from queue - NEVER blocks main loop"""
        while self.api_worker_running:
            try:
                # Wait for command with timeout (allows clean shutdown)
                cmd = self.api_queue.get(timeout=0.1)
                method, endpoint, data = cmd

                # Execute the actual API call
                self._api_request_sync(method, endpoint, data)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"API worker error: {e}")

    def _api_request_sync(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """Synchronous API request - called by worker thread only"""
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

    def api_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """NON-BLOCKING API request - queues command and returns immediately

        This prevents button presses from blocking the main event loop.
        Commands are processed asynchronously by the worker thread.
        """
        current_time = time.time()

        # Debounce: skip if same command sent too recently
        cmd_key = f"{method}:{endpoint}"
        if current_time - self.last_command_time[cmd_key] < self.DEBOUNCE_TIME:
            logger.debug(f"Debounced: {endpoint}")
            return {"success": True, "debounced": True}

        self.last_command_time[cmd_key] = current_time

        # Queue the command (non-blocking)
        try:
            self.api_queue.put_nowait((method, endpoint, data))
            return {"success": True, "queued": True}
        except:
            # Queue full - drop oldest and add new
            try:
                self.api_queue.get_nowait()  # Remove oldest
                self.api_queue.put_nowait((method, endpoint, data))
                logger.warning(f"API queue full, dropped oldest command")
                return {"success": True, "queued": True}
            except:
                logger.error(f"Failed to queue API command: {endpoint}")
                return None

    def api_request_blocking(self, method: str, endpoint: str, data: Optional[dict] = None) -> Optional[dict]:
        """BLOCKING API request - use ONLY when response is needed (e.g., recording status)"""
        return self._api_request_sync(method, endpoint, data)

    def _start_watchdog(self):
        """Start watchdog thread for motor safety"""
        self.watchdog_running = True
        self.watchdog_thread = Thread(target=self._watchdog_loop, daemon=False)  # NOT daemon!
        self.watchdog_thread.start()
        logger.info("Safety watchdog started")

    def _watchdog_loop(self):
        """Watchdog that stops motors if controller freezes or disconnects"""
        while self.watchdog_running:
            try:
                current_time = time.time()

                # Check for heartbeat timeout
                if current_time - self.last_heartbeat_time > self.MOTOR_WATCHDOG_TIMEOUT:
                    if not self.state.motors_stopped:
                        logger.critical("WATCHDOG: No heartbeat - emergency stop!")
                        self.emergency_stop()
                        self.state.motors_stopped = True

                # Check for controller disconnection
                if self.controller_connected:
                    if current_time - self.last_heartbeat_time > self.CONNECTION_TIMEOUT:
                        logger.critical("WATCHDOG: Controller disconnected - emergency stop!")
                        self.emergency_stop()
                        self.controller_connected = False

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Watchdog error: {e} - stopping motors")
                self.emergency_stop()

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
                else:
                    # FIXED: Send stop commands to keep motor bus alive even when stick centered
                    if not self.state.motors_stopped:
                        with self.motor_lock:
                            self.set_motor_speeds(0, 0)  # Send explicit stop command
                            self.state.motors_stopped = True
                            logger.debug("Motors stopped (no input)")
                    else:
                        # Send periodic heartbeat stop commands to prevent watchdog timeout
                        if current_time - self.last_motor_command_time > 2.0:  # Every 2 seconds
                            with self.motor_lock:
                                self.set_motor_speeds(0, 0)  # Heartbeat stop command
                            self.last_motor_command_time = current_time

                time.sleep(self.MOTOR_UPDATE_RATE)

            except Exception as e:
                logger.error(f"Motor update loop error: {e}")
                time.sleep(0.1)

    def _stop_motors_internal(self):
        """Internal motor stop without lock (call with motor_lock held)"""
        if self.motor_direct and self.motor_bus:
            try:
                cmd = create_motor_command(0, 0, CommandSource.XBOX_CONTROLLER)
                self.motor_bus.send_command(cmd)
                self.state.last_left_speed = 0
                self.state.last_right_speed = 0
                return True
            except Exception as e:
                logger.error(f"Motor bus stop error: {e}")
        elif self.motor_direct and self.motor_controller:
            try:
                self.motor_controller.emergency_stop()
                self.state.last_left_speed = 0
                self.state.last_right_speed = 0
                return True
            except Exception as e:
                logger.error(f"Direct motor stop error: {e}")
        return False

    def read_event(self) -> Optional[Tuple]:
        """Read a single joystick event with timeout to prevent blocking"""
        try:
            # Use select to add timeout - prevents hanging if device issues
            ready, _, _ = select.select([self.device], [], [], 0.1)  # 100ms timeout
            if ready:
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
            # SAFETY: If stick centered, ensure motors can stop
            if normalized == 0.0 and abs(self.state.left_y) < self.DEADZONE:
                self.state.motors_stopped = False  # Allow stop command to be sent

        elif number == 1:  # Left stick Y (inverted for forward)
            self.state.left_y = -normalized
            if abs(normalized) > self.DEADZONE:
                notify_manual_input()
            # SAFETY: If stick centered, immediately stop motors
            if normalized == 0.0 and abs(self.state.left_x) < self.DEADZONE:
                self.state.motors_stopped = False  # Allow stop command to be sent
                with self.motor_lock:
                    self.set_motor_speeds(0, 0)
                    self.state.motors_stopped = True

        elif number == 2:  # Left trigger (LT)
            # LT ranges from 0 to 32767 (not -32767 to 32767)
            normalized_trigger = value / 32767.0
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

        elif number == 5:  # Right trigger (RT) - play "good dog" audio
            # RT ranges from 0 to 32767 (not -32767 to 32767)
            normalized_trigger = value / 32767.0
            if normalized_trigger > self.TRIGGER_DEADZONE:
                self.state.right_trigger = normalized_trigger
                # Play good_dog.mp3 with cooldown to prevent spam
                current_time = time.time()
                if not hasattr(self, '_last_rt_time'):
                    self._last_rt_time = 0
                if current_time - self._last_rt_time > 1.0:  # 1 second cooldown
                    logger.info("RT: Playing good_dog.mp3")
                    self.api_request('POST', '/audio/play/file', {"filepath": "/talks/good_dog.mp3"})
                    self._last_rt_time = current_time
            else:
                self.state.right_trigger = 0.0

    def update_motor_control(self):
        """Update motor speeds - SIMPLIFIED, no turbo mode, reliable stopping"""
        # Left stick controls movement at MAX_SPEED (75%)
        # No turbo mode - consistent speed for safety

        # Check if stick is in deadzone (should stop)
        stick_in_deadzone = (abs(self.state.left_y) < self.DEADZONE and
                            abs(self.state.left_x) < self.DEADZONE)

        if stick_in_deadzone:
            # Force stop when stick is centered
            left_speed = 0
            right_speed = 0
        else:
            # Calculate motor speeds from left stick position
            forward = self.state.left_y * self.MAX_SPEED
            turn = -self.state.left_x * self.MAX_SPEED * self.TURN_SPEED_FACTOR

            # Tank drive mixing
            left_speed = int(forward - turn)
            right_speed = int(forward + turn)

            # Clamp to valid range
            left_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, left_speed))
            right_speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, right_speed))

        # Rate limiting - 20Hz for motor commands (faster response)
        current_time = time.time()
        if current_time - self.last_motor_update < 0.05:
            return

        # ALWAYS send stop command immediately when stopping
        # For moving, send on any change
        should_send = False
        if left_speed == 0 and right_speed == 0:
            # Stop command - always send if not already stopped
            if not self.state.motors_stopped:
                should_send = True
        else:
            # Moving - send on change or periodically
            is_change = (abs(left_speed - self.state.last_left_speed) > 2 or
                        abs(right_speed - self.state.last_right_speed) > 2)
            should_send = is_change or (current_time - self.last_motor_update > 0.2)

        if should_send:
            self.set_motor_speeds(left_speed, right_speed)
            self.state.last_left_speed = left_speed
            self.state.last_right_speed = right_speed
            self.state.motors_stopped = (left_speed == 0 and right_speed == 0)
            self.last_motor_update = current_time

    def set_motor_speeds(self, left: int, right: int):
        """
        Set motor speeds using closed-loop RPM control or fallback methods

        Converts speed percentages to RPM targets for precise control
        """
        if self.USE_PID_CONTROL and self.motor_direct and self.motor_controller:
            try:
                # Convert speed percentages to RPM targets
                left_rpm = (left / self.MAX_SPEED) * self.MAX_RPM
                right_rpm = (right / self.MAX_SPEED) * self.MAX_RPM

                # Use PID closed-loop control for precise motor matching
                self.motor_controller.set_motor_rpm(left_rpm, right_rpm)

            except Exception as e:
                logger.error(f"PID motor control error: {e}")
                # Fallback to direct PWM control
                self._fallback_pwm_control(left, right)

        elif self.motor_direct and self.motor_bus:
            try:
                # Use motor command bus for proper closed-loop control
                cmd = create_motor_command(left, right, CommandSource.XBOX_CONTROLLER)
                success = self.motor_bus.send_command(cmd)

                if success:
                    # Get encoder feedback
                    if hasattr(self.motor_bus.motor_controller, 'get_encoder_counts'):
                        left_enc, right_enc = self.motor_bus.motor_controller.get_encoder_counts()
                        logger.debug(f"Closed-loop: L={left:4d}, R={right:4d} | Encoders: L={left_enc}, R={right_enc}")
                else:
                    logger.error("Motor command bus failed - falling back to direct control")
                    # Fallback to direct control
                    if left == 0:
                        self.motor_controller.set_motor_speed('left', 0, 'forward')
                    else:
                        direction = 'forward' if left > 0 else 'backward'
                        self.motor_controller.set_motor_speed('left', abs(left), direction)

                    if right == 0:
                        self.motor_controller.set_motor_speed('right', 0, 'forward')
                    else:
                        direction = 'forward' if right > 0 else 'backward'
                        self.motor_controller.set_motor_speed('right', abs(right), direction)

            except Exception as e:
                logger.error(f"Closed-loop motor control error: {e}")

        elif self.motor_direct and self.motor_controller:
            # Fallback to direct motor controller (legacy path with manual compensation)
            try:
                if left == 0 and right == 0:
                    # Gentle stop - set both motors to 0 speed
                    motor_controller.set_motor_speed('left', 0, 'forward')
                    motor_controller.set_motor_speed('right', 0, 'forward')
                    logger.debug(f"Motors: STOPPED (gentle)")
                else:
                    # HARDWARE COMPENSATION (legacy - motor bus handles this now):
                    # Swap left/right because wires are physically swapped
                    actual_left_speed = right  # What we want left track to do
                    actual_right_speed = left  # What we want right track to do

                    # Calculate directions
                    actual_left_dir = 'forward' if actual_left_speed >= 0 else 'backward'
                    actual_right_dir = 'forward' if actual_right_speed >= 0 else 'backward'

                    # Invert right motor direction (hardware is wired backwards)
                    if actual_right_dir == 'forward':
                        actual_right_dir = 'backward'
                    elif actual_right_dir == 'backward':
                        actual_right_dir = 'forward'

                    # Get absolute speeds
                    actual_left_speed_abs = abs(actual_left_speed)
                    actual_right_speed_abs = abs(actual_right_speed)

                    # Apply minimum speed for DFRobot motors (they need 30% minimum)
                    if actual_left_speed_abs > 0 and actual_left_speed_abs < 30:
                        actual_left_speed_abs = 30
                    if actual_right_speed_abs > 0 and actual_right_speed_abs < 30:
                        actual_right_speed_abs = 30

                    # Send commands to motor controller with swapped assignment
                    # (motor_controller 'left' controls actual right track)
                    # (motor_controller 'right' controls actual left track)
                    self.motor_controller.set_motor_speed('right', actual_left_speed_abs, actual_left_dir)  # Swapped!
                    self.motor_controller.set_motor_speed('left', actual_right_speed_abs, actual_right_dir)  # Swapped!

                    logger.debug(f"LEGACY: L={left:4d}, R={right:4d} | "
                                f"MC_L→RightTrack={actual_right_speed_abs}{actual_right_dir[0]}, "
                                f"MC_R→LeftTrack={actual_left_speed_abs}{actual_left_dir[0]}")
            except Exception as e:
                logger.error(f"Motor control error: {e}")
                # Try to stop motors on error
                try:
                    self.motor_controller.set_motor_speed('left', 0, 'forward')
                    self.motor_controller.set_motor_speed('right', 0, 'forward')
                except:
                    pass

    def _fallback_pwm_control(self, left: int, right: int):
        """Fallback to direct PWM control when PID fails"""
        try:
            if left == 0 and right == 0:
                self.motor_controller.set_motor_speed('left', 0, 'forward')
                self.motor_controller.set_motor_speed('right', 0, 'forward')
                return

            # Apply left motor boost for hardware compensation (left is weaker)
            left_compensated = min(100, left * self.LEFT_MOTOR_BOOST) if left != 0 else 0

            # Set motor speeds with direction control
            if left_compensated >= 0:
                self.motor_controller.set_motor_speed('left', abs(left_compensated), 'forward')
            else:
                self.motor_controller.set_motor_speed('left', abs(left_compensated), 'backward')

            if right >= 0:
                self.motor_controller.set_motor_speed('right', abs(right), 'forward')
            else:
                self.motor_controller.set_motor_speed('right', abs(right), 'backward')

            logger.debug(f"Fallback PWM: L={left} (boosted={left_compensated:.1f}), R={right}")

        except Exception as e:
            logger.error(f"Fallback PWM control error: {e}")

    def stop_motors(self):
        """Stop all motors with lock"""
        with self.motor_lock:
            self.set_motor_speeds(0, 0)
            self.state.motors_stopped = True
            logger.info("Motors stopped")

    def emergency_stop(self):
        """Emergency stop - immediate halt"""
        logger.warning("EMERGENCY STOP activated")
        with self.motor_lock:
            self.set_motor_speeds(0, 0)
            self.state.motors_stopped = True

    def _emergency_stop_all_motors(self):
        """CRITICAL: B button emergency stop - immediate motor halt"""
        logger.critical("B BUTTON EMERGENCY STOP - Stopping all motors immediately")
        try:
            # Stop via motor bus
            if self.motor_bus:
                cmd = create_motor_command(0, 0, CommandSource.XBOX_CONTROLLER)
                self.motor_bus.send_command(cmd)

            # Stop via direct motor controller
            if self.motor_controller:
                self.motor_controller.set_motor_speed('left', 0, 'stop')
                self.motor_controller.set_motor_speed('right', 0, 'stop')

            # Update state
            with self.motor_lock:
                self.state.motors_stopped = True
                self.state.last_left_speed = 0
                self.state.last_right_speed = 0

            logger.critical("B BUTTON EMERGENCY STOP COMPLETE")
        except Exception as e:
            logger.critical(f"B button emergency stop failed: {e}")
            # Fallback to global emergency stop
            global_emergency_stop()
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

        elif number == 1:  # B button - EMERGENCY STOP (CRITICAL SAFETY)
            self.state.b_button = pressed
            if pressed:
                logger.critical("B BUTTON EMERGENCY STOP - All motors halted immediately")
                self._emergency_stop_all_motors()
                # Also trigger global emergency stop as backup
                global_emergency_stop()

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

        elif number == 7:  # Start/Menu button (☰) - Record new talk
            if pressed:
                self.handle_record_button()

    # ============== AUDIO RECORDING ==============
    def handle_record_button(self):
        """
        Handle Start/Menu button for recording new talk audio.
        First press: Start recording (2 sec) + playback
        Second press within 10s: Save to VOICEMP3/talks
        """
        current_time = time.time()
        logger.warning(f"🎙️ handle_record_button CALLED at {current_time:.2f}")

        if not hasattr(self, '_last_record_button'):
            self._last_record_button = 0

        # Check recording status FIRST (BLOCKING - needs response)
        status = self.api_request_blocking('GET', '/audio/record/status')
        logger.warning(f"🎙️ Recording status: {status}")

        # If recording in progress, ignore (server will also reject)
        if status and status.get('in_progress'):
            logger.warning("🎙️ Recording in progress - IGNORING button press")
            return

        if status and status.get('has_pending'):
            # Second press - confirm and save (BLOCKING - needs response)
            logger.info("🎙️ START BUTTON: Confirming recording save")
            result = self.api_request_blocking('POST', '/audio/record/confirm')
            if result and result.get('success'):
                logger.info(f"✅ Recording saved: {result.get('filename')}")
                # Rescan folders to include the new recording in D-pad list
                self._scan_audio_folders()
                logger.info(f"🔄 Audio list refreshed: {len(self.TALK_TRACKS)} talks available")
            else:
                logger.warning(f"Recording save failed: {result}")
        else:
            # First press - start new recording (with cooldown to prevent double-trigger)
            if current_time - self._last_record_button < 6.0:
                logger.debug("Record button ignored (cooldown - recording may be in progress)")
                return

            self._last_record_button = current_time
            logger.info("🎙️ START BUTTON: Starting new recording (2 seconds)")
            result = self.api_request_blocking('POST', '/audio/record/start')
            if result and result.get('success'):
                logger.info("🎙️ Recording complete - press START again within 10s to save")
            else:
                logger.warning(f"Recording failed: {result}")

    # ============== END AUDIO RECORDING ==============

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
        """Toggle blue LED with cooldown to prevent double-triggers"""
        current_time = time.time()

        # Cooldown: ignore if less than 500ms since last toggle
        if not hasattr(self, '_last_led_toggle'):
            self._last_led_toggle = 0

        if current_time - self._last_led_toggle < 0.5:
            logger.debug("Blue LED toggle ignored (cooldown)")
            return

        self._last_led_toggle = current_time
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
        """Play TREAT sound (Y button)"""
        logger.info("Y button: Playing 'Treat' sound")
        self.api_request('POST', '/audio/play/file', {"filepath": "/talks/treat.mp3"})


    def play_selected_talk(self):
        """Play selected talk track (D-pad right)"""
        file_path, track_name = self.TALK_TRACKS[self.current_talk_index]
        logger.info(f"Playing Talk: {track_name} ({file_path})")
        self.api_request('POST', '/audio/play/file', {"filepath": file_path})

    def play_selected_song(self):
        """Play selected song track (D-pad left)"""
        file_path, track_name = self.SONG_TRACKS[self.current_song_index]
        logger.info(f"Playing Song: {track_name} ({file_path})")
        self.api_request('POST', '/audio/play/file', {"filepath": file_path})

    def process_dpad(self, number: int, value: int):
        """Process D-pad input"""
        if value != 0:
            notify_manual_input()

        if number == 6:  # D-pad X axis
            self.state.dpad_left = (value < 0)
            self.state.dpad_right = (value > 0)

            # Add cooldown to prevent rapid cycling
            current_time = time.time()
            if current_time - self.last_dpad_time < self.dpad_cooldown:
                return

            if value < 0:  # Left - Queue next SONG for playing
                self.current_song_index = (self.current_song_index + 1) % len(self.SONG_TRACKS)
                self.queued_track = self.SONG_TRACKS[self.current_song_index]
                self.queued_type = "song"
                file_path, track_name = self.queued_track
                logger.info(f"🎵 QUEUED Song: {track_name} - Press DOWN to play")
                self.last_dpad_time = current_time
            elif value > 0:  # Right - Queue next TALK for playing
                self.current_talk_index = (self.current_talk_index + 1) % len(self.TALK_TRACKS)
                self.queued_track = self.TALK_TRACKS[self.current_talk_index]
                self.queued_type = "talk"
                file_path, track_name = self.queued_track
                logger.info(f"🗣️ QUEUED Talk: {track_name} - Press DOWN to play")
                self.last_dpad_time = current_time

        elif number == 7:  # D-pad Y axis
            self.state.dpad_up = (value < 0)
            self.state.dpad_down = (value > 0)

            if value < 0:  # Up - Stop audio
                logger.info("D-pad up: Stop audio")
                self.api_request('POST', '/audio/stop')
            elif value > 0:  # Down - Play the QUEUED track
                # Add cooldown to prevent rapid audio triggering
                current_time = time.time()
                if current_time - self.last_dpad_time >= self.dpad_cooldown:
                    if self.queued_track is not None:
                        file_path, track_name = self.queued_track
                        logger.info(f"▶️ PLAYING {self.queued_type.upper()}: {track_name} ({file_path})")
                        self.api_request('POST', '/audio/play/file', {"filepath": file_path})
                    else:
                        logger.info("No track queued - use LEFT/RIGHT D-pad first")
                    self.last_dpad_time = current_time

    def run(self):
        """Main control loop"""
        if not self.connect():
            logger.error("Failed to connect to controller")
            return

        # Start async API worker FIRST - prevents button blocking
        self._start_api_worker()

        self.running = True
        logger.info("Xbox Fixed Controller ready!")
        logger.info("=== CONTROLS ===")
        logger.info("Movement: Left stick = 60% power, RT = TURBO (100%)")
        logger.info("Camera: Right stick")
        logger.info("A = Emergency Stop, B = Stop Motors")
        logger.info("X = Blue LED, LT = NeoPixel modes")
        logger.info("Y = Sound, LB = Treat (2s cooldown), RB = Photo")
        logger.info("=== ASYNC API QUEUE ===")
        logger.info("✅ Non-blocking button presses")
        logger.info("✅ 50ms command debouncing")
        logger.info("✅ No more system freezes from rapid inputs")

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
        """Clean up resources with emergency stop"""
        logger.info("Cleaning up...")

        # FIRST: Emergency stop motors immediately
        self.emergency_stop()

        # Then stop threads
        self.running = False
        self.motor_update_running = False
        self.camera_update_running = False
        self.heartbeat_running = False
        self.api_worker_running = False  # Stop async API worker

        # Stop API worker thread
        if self.api_worker_thread:
            self.api_worker_thread.join(timeout=0.5)

        # Stop motor update thread
        if self.motor_update_thread:
            self.motor_update_thread.join(timeout=1.0)

        # Stop camera update thread
        if self.camera_update_thread:
            self.camera_update_thread.join(timeout=1.0)

        # Stop heartbeat thread
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join(timeout=1.0)

        # Final motor stop to be sure
        self.stop_motors()

        # Clean up motor system
        if self.motor_direct and self.motor_bus:
            try:
                self.motor_bus.stop()
            except:
                pass
        elif self.motor_direct and self.motor_controller:
            try:
                self.motor_controller.cleanup()
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
    # MOTOR SAFETY WARNING
    logger.warning("=" * 60)
    logger.warning("MOTOR SAFETY ACTIVE: 6V motors on 14V system")
    logger.warning("Controllers now limit PWM to 50% max (6.3V)")
    logger.warning("Your motors are protected from overvoltage damage")
    logger.warning("=" * 60)

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