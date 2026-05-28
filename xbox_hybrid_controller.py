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
    from config.config_loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

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

# =============================================================================
# Singleton Enforcement - Kill any existing instances before starting
# =============================================================================
def ensure_single_instance():
    """Ensure only one instance of this controller is running"""
    my_pid = os.getpid()
    script_name = 'xbox_hybrid_controller.py'

    try:
        # Find all processes running this script
        result = subprocess.run(
            ['pgrep', '-f', script_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            pids = [int(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]

            # Kill any instances that aren't us
            killed = []
            for pid in pids:
                if pid != my_pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed.append(pid)
                    except ProcessLookupError:
                        pass  # Already dead
                    except PermissionError:
                        pass  # Can't kill it

            if killed:
                print(f"[XboxController] Killed {len(killed)} existing instance(s): {killed}")
                time.sleep(0.5)  # Brief pause to let them die

    except Exception as e:
        print(f"[XboxController] Singleton check error (non-fatal): {e}")

# Run singleton check immediately on import
ensure_single_instance()
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

# Rate limit for manual input notifications (prevents thread spam)
_last_manual_notify_time = 0
_MANUAL_NOTIFY_INTERVAL = 0.1  # Max 10 notifications per second

def notify_manual_input():
    """Notify the system that manual input occurred - NON-BLOCKING with rate limit"""
    global _last_manual_notify_time

    # Rate limit to prevent thread spam on rapid button presses
    current_time = time.time()
    if current_time - _last_manual_notify_time < _MANUAL_NOTIFY_INTERVAL:
        return
    _last_manual_notify_time = current_time

    if event_bus:
        # Use a thread to prevent blocking main loop if event bus is slow
        def _notify():
            try:
                publish_system_event('manual_input_detected', {
                    'timestamp': time.time(),
                    'source': 'xbox_controller'
                }, 'xbox_hybrid_controller')
            except Exception as e:
                logger.warning(f"Failed to notify manual input: {e}")

        # Fire and forget - don't wait for event bus
        threading.Thread(target=_notify, daemon=True).start()

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

    # Default controller configuration - OVERRIDDEN BY ROBOT CONFIG
    # These defaults are fallbacks if config loading fails
    DEADZONE = 0.20  # 20% deadzone - larger for reliable stop detection
    TRIGGER_DEADZONE = 0.1

    # Per-axis stick center offsets (raw normalized -1..1 space) to correct
    # worn / off-center potentiometers. Keyed by js axis: 0=LX, 1=LY, 3=RX, 4=RY.
    # Overridden by controller.xbox.stick_centers in robot_profiles/<unit>.yaml.
    STICK_CENTERS = {0: 0.0, 1: 0.0, 3: 0.0, 4: 0.0}
    MAX_SPEED = 80   # Match treatbot1.yaml config
    TURN_SPEED_FACTOR = 0.8  # Reduced for smoother turns

    # RPM Control - Convert speed percentages to RPM targets
    MAX_RPM = 110    # Match treatbot1.yaml config
    USE_PID_CONTROL = True  # ENABLED - closed-loop control with encoder feedback

    # Motor calibration - per-motor power adjustment for hardware imbalance
    LEFT_MOTOR_MULTIPLIER = 0.20   # Default - will be overridden by config
    RIGHT_MOTOR_MULTIPLIER = 2.0   # Default - will be overridden by config
    MIN_PWM_THRESHOLD = 0          # Minimum PWM to overcome motor deadzone

    # Gimbal stick-driven clamps - fallback defaults; per-device values come from yaml
    # camera.{pan_min,pan_max,tilt_min,tilt_max,pan_center,tilt_center} in robot_profiles/<unit>.yaml
    PAN_MIN = 10
    PAN_MAX = 270
    TILT_MIN = 20
    TILT_MAX = 160
    PAN_CENTER = 100   # Where the right-stick-click "center" command sends camera
    TILT_CENTER = 90

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
        ("/talks/default/good.mp3", "Good Dog"),
        ("/talks/default/treat.mp3", "Treat")
    ]

    def __init__(self, device_path: str = '/dev/input/js0'):
        self.device_path = device_path
        self.device = None
        self.running = False
        self.state = ControllerState()
        self.stop_event = Event()

        # Load robot-specific configuration
        self._load_robot_config()

        # Thread safety locks
        self.motor_lock = Lock()
        self.treat_lock = Lock()
        # REMOVED: api_lock - replaced with async queue (non-blocking)

        # Cooldown tracking
        self.last_treat_time = 0
        self._lb_press_time = 0
        self._lb_refill_active = False
        self._lb_refill_thread = None
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

        # Mode cycling (SELECT/Back button) - includes IDLE for passive listening
        self.cycle_modes = ['manual', 'idle', 'coach', 'silent_guardian']
        self.current_mode_index = 0  # Start with manual
        self.last_mode_cycle_time = 0
        self.mode_cycle_cooldown = 0.15  # 150ms debounce - fast mode cycling allowed

        # Audio announcements for mode changes (in /VOICEMP3/wimz/)
        self.mode_audio = {
            'manual': '/wimz/ManualMode.mp3',
            'idle': '/wimz/IdleMode.mp3',
            'coach': '/wimz/CoachMode.mp3',
            'silent_guardian': '/wimz/SilentGuardianMode.mp3'
        }

        # Trick cycling (Xbox Guide button) - for coach mode testing
        self._trick_cycle_index = 0
        self._available_tricks = ['sit', 'laydown', 'come', 'spin', 'speak']
        # Audio files for each trick (from trick_rules.yaml)
        self._trick_audio = {
            'sit': 'sit.mp3',
            'laydown': 'laydown.mp3',
            'come': 'come.mp3',
            'spin': 'spin.mp3',
            'speak': 'speak.mp3'
        }
        self._last_trick_cycle_time = 0
        self._trick_cycle_cooldown = 1.0  # 1 second cooldown

        # Shutdown via Start button (removed A+B combo)
        self._shutdown_in_progress = False

        # Share button (the capture button below the Xbox logo) cycles volume.
        # xpadneo routes Share to KEY_F12 on a SEPARATE keyboard input node,
        # not the gamepad/js0 node — see _share_button_loop().
        self._volume_levels = [0, 20, 40, 60, 80, 100]
        self._volume_cycle_index = None  # resolved from live volume on first press
        self._last_volume_cycle_time = 0
        self._volume_cycle_cooldown = 0.3  # 300ms debounce between presses
        self.share_reader_running = False
        self.share_reader_thread = None

        # Dog identity cycling (D-pad up in coach mode) - for demo recording
        self._last_dog_cycle_time = 0
        self._dog_cycle_cooldown = 1.0

        # Command cycling (B button) - Sit, Speak, Stay, Quiet, LieDown, Spin
        # Uses sound names for dog-aware path resolution
        self._command_cycle_index = 0
        self._command_names = ["sit", "speak", "stay", "quiet", "laydown", "spin"]
        self._command_pending_timer: Optional[Timer] = None
        self._command_last_cycle_time = 0  # For auto-reset after idle
        self._CYCLE_RESET_TIMEOUT = 2.0  # Reset to first track after 2s idle

        # RT now only plays "Good" (no cycling)
        # RB now plays "No" (instead of photo)

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

        # When spawned by main_treatbot (services/control/xbox_controller.py
        # sets WIMZ_XBOX_SUBPROCESS=1), the MAIN process owns the motor
        # hardware. This subprocess must NOT create its own motor bus — a
        # second ProperPIDMotorController claims the same GPIO lines and
        # leaves one motor undrivable. Drive motors via the HTTP API instead.
        if os.environ.get('WIMZ_XBOX_SUBPROCESS') == '1':
            self.motor_bus = None
            self.motor_controller = None
            self.motor_direct = False
            logger.info("Xbox subprocess: motor control via HTTP API "
                        "(main process owns motor hardware)")
            return

        if MOTOR_BUS_AVAILABLE:
            try:
                self.motor_bus = get_motor_bus()
                if self.motor_bus.start():
                    self.motor_direct = True
                    motor_bus = self.motor_bus  # Update global reference
                    MOTOR_DIRECT = True
                    # CRITICAL FIX: Set motor controller reference
                    self.motor_controller = self.motor_bus.motor_controller
                    logger.info("Motor command bus with polling encoders initialized")
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
            logger.info("Using motor controller from motor bus for PID control")
            return

        if MOTOR_CONTROLLER_AVAILABLE:
            try:
                self.motor_controller = MotorControllerPolling()
                if self.motor_controller.initialize():
                    self.motor_direct = True
                    motor_controller = self.motor_controller  # Update global reference
                    MOTOR_DIRECT = True
                    logger.info("DFRobot polling motor control initialized (fallback)")
                    return
                else:
                    logger.warning("DFRobot motor controller failed to initialize")
                    self.motor_controller = None
            except Exception as e:
                logger.error(f"Motor controller initialization error: {e}")
                self.motor_controller = None

        logger.warning("No motor control available, will use API")
        self.motor_direct = False
        MOTOR_DIRECT = False

    def _load_robot_config(self):
        """Load robot-specific configuration from YAML profile"""
        if not CONFIG_AVAILABLE:
            logger.warning("Config loader not available, using default values")
            return

        try:
            config = get_config()
            controller_config = config.controller

            # Override class defaults with robot-specific values
            self.DEADZONE = controller_config.xbox_deadzone
            self.MAX_SPEED = controller_config.max_speed
            self.TURN_SPEED_FACTOR = controller_config.turn_speed_factor
            self.MAX_RPM = controller_config.max_rpm
            self.USE_PID_CONTROL = controller_config.use_pid_control
            self.LEFT_MOTOR_MULTIPLIER = controller_config.left_motor_multiplier
            self.RIGHT_MOTOR_MULTIPLIER = controller_config.right_motor_multiplier
            self.MIN_PWM_THRESHOLD = controller_config.min_pwm_threshold

            # Gimbal limits for stick-driven pan/tilt — read from yaml so per-device
            # ranges (treatbot3 has wider servos than treatbot1/2) are honored.
            # Defaults match the previous hardcoded values for backwards compat.
            cam_cfg = config.raw.get('camera', {})
            self.PAN_MIN = cam_cfg.get('pan_min', 10)
            self.PAN_MAX = cam_cfg.get('pan_max', 270)
            self.TILT_MIN = cam_cfg.get('tilt_min', 20)
            self.TILT_MAX = cam_cfg.get('tilt_max', 160)
            self.PAN_CENTER = cam_cfg.get('pan_center', 100)
            self.TILT_CENTER = cam_cfg.get('tilt_center', 90)
            # Sync stick-tracking state to the calibrated center so the next stick
            # nudge moves relative to actual hardware center, not stale init values.
            self.last_pan_angle = self.PAN_CENTER
            self.last_tilt_angle = self.TILT_CENTER

            # Per-axis stick center offsets — corrects worn/off-center sticks.
            # yaml uses named keys; map to js axis numbers (0=LX,1=LY,3=RX,4=RY).
            sc = config.raw.get('controller', {}).get('xbox', {}).get('stick_centers', {})
            self.STICK_CENTERS = {
                0: float(sc.get('left_x', 0.0)),
                1: float(sc.get('left_y', 0.0)),
                3: float(sc.get('right_x', 0.0)),
                4: float(sc.get('right_y', 0.0)),
            }

            logger.info(f"[Config] Robot profile loaded: {config.robot_id}")
            logger.info(f"[Config] DEADZONE={self.DEADZONE}")
            if any(self.STICK_CENTERS.values()):
                logger.info(f"[Config] Stick center calibration active: {self.STICK_CENTERS}")
            logger.debug(f"[Config] MAX_SPEED={self.MAX_SPEED}, MAX_RPM={self.MAX_RPM}")
            logger.debug(f"[Config] LEFT_MULT={self.LEFT_MOTOR_MULTIPLIER}, RIGHT_MULT={self.RIGHT_MOTOR_MULTIPLIER}")
            logger.debug(f"[Config] MIN_PWM_THRESHOLD={self.MIN_PWM_THRESHOLD}")
            logger.debug(f"[Config] USE_PID_CONTROL={self.USE_PID_CONTROL}")

        except Exception as e:
            logger.error(f"Failed to load robot config: {e}")
            logger.warning("Using default controller values")

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
        # Scan talks/default and songs/default folders
        self.TALK_TRACKS = self._scan_folder("talks/default")
        self.SONG_TRACKS = self._scan_folder("songs/default")

        # Fallback if folders are empty
        if not self.TALK_TRACKS:
            self.TALK_TRACKS = [("/talks/default/treat.mp3", "Treat")]
            logger.warning("No talks found, using fallback")
        if not self.SONG_TRACKS:
            self.SONG_TRACKS = [("/songs/default/scooby_snacks.mp3", "Scooby Snacks")]
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
            logger.debug("Preloading audio system...")
            result = self.api_request('GET', '/audio/status')
            if result:
                logger.debug("Audio system preloaded successfully")
        except Exception as e:
            logger.warning(f"Audio preload error: {e}")

        # Scan folders for audio files
        self._scan_audio_folders()

        logger.info(f"Audio tracks discovered: {len(self.TALK_TRACKS)} talks, {len(self.SONG_TRACKS)} songs")
        logger.debug(f"Talks: {[t[1] for t in self.TALK_TRACKS]}")
        logger.debug(f"Songs: {[s[1] for s in self.SONG_TRACKS]}")

        # Cache for active dog (updated from API periodically)
        self._active_dog = None
        self._last_dog_check = 0

    def _get_active_dog(self) -> Optional[str]:
        """Get the currently active dog using C3.2 fallback chain.

        Priority (highest wins):
        (a) ArUco-identified dog within 5 seconds
        (b) Session dog_id (coaching/mission)
        (c) App's select_dog (persisted)

        Caches result for 10 seconds to avoid excessive state checks.
        Returns dog_id string or None if no dog is active.
        """
        current_time = time.time()
        if current_time - self._last_dog_check < 10.0:
            return self._active_dog

        try:
            # Use state's centralized fallback chain (C3.2)
            from core.state import get_state
            state = get_state()
            dog_id = state.get_active_dog_id(aruco_ttl=5.0)
            if dog_id:
                self._active_dog = dog_id
                self._last_dog_check = current_time
                return self._active_dog
        except Exception as e:
            logger.debug(f"Could not get active dog from state: {e}")

        # Fallback to API (for cases where state isn't available)
        try:
            result = self.api_request_blocking('GET', '/coaching/status', timeout=1)
            if result:
                forced = result.get('forced_dog')
                if forced:
                    self._active_dog = forced.lower()
                    self._last_dog_check = current_time
                    return self._active_dog
                session = result.get('current_session')
                if session and session.get('dog_name'):
                    dog_name = session['dog_name']
                    if dog_name.lower() not in ['dog', 'unknown']:
                        self._active_dog = dog_name.lower()
                        self._last_dog_check = current_time
                        return self._active_dog
        except Exception:
            pass

        self._active_dog = None
        self._last_dog_check = current_time
        return None

    def _play_voice_command(self, command_name: str):
        """Play a per-dog voice command via the server's /audio/play_command endpoint.

        Per-dog override resolution (ArUco / session dog / select_dog / default)
        happens server-side because the Xbox controller is a subprocess of
        main_treatbot — it has its own (empty) state singleton and cannot see
        the main robot's active dog. Resolving locally would always fall
        through to the default folder.
        """
        self.api_request('POST', '/audio/play_command', {"command": command_name})

    def _start_api_worker(self):
        """Start the async API worker thread"""
        if self.api_worker_running:
            return
        self.api_worker_running = True
        self.api_worker_thread = Thread(target=self._api_worker_loop, daemon=True)
        self.api_worker_thread.start()
        logger.debug("Async API worker started")

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

    def _api_request_sync(self, method: str, endpoint: str, data: Optional[dict] = None, timeout: float = None) -> Optional[dict]:
        """Synchronous API request - called by worker thread only"""
        url = f"{self.API_BASE_URL}{endpoint}"
        try:
            if timeout is None:
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

    def api_request_blocking(self, method: str, endpoint: str, data: Optional[dict] = None, timeout: float = None) -> Optional[dict]:
        """BLOCKING API request - use ONLY when response is needed (e.g., recording status)"""
        return self._api_request_sync(method, endpoint, data, timeout=timeout)

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
        """Connect to the Xbox controller with retry logic for first-connect issues"""
        max_retries = 3
        retry_delay = 1.5  # seconds

        for attempt in range(max_retries):
            try:
                # Check if API is available
                health = self.api_request('GET', '/health')
                if health:
                    logger.debug(f"API health check: {health}")
                else:
                    logger.warning("API server not responding - only motor control will work")

                # Open the joystick device
                self.device = open(self.device_path, 'rb')
                logger.info(f"Connected to Xbox controller at {self.device_path}")

                # Wait briefly for device to stabilize on first connect
                if attempt == 0:
                    time.sleep(0.5)

                # Test read to verify device is responding
                ready, _, _ = select.select([self.device], [], [], 0.5)
                if not ready and attempt < max_retries - 1:
                    logger.warning(f"Controller not responding (attempt {attempt + 1}/{max_retries}), retrying...")
                    self.device.close()
                    time.sleep(retry_delay)
                    continue

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

                # Start Share-button reader (separate keyboard node, KEY_F12)
                self.share_reader_running = True
                self.share_reader_thread = Thread(target=self._share_button_loop, daemon=True)
                self.share_reader_thread.start()
                logger.info("Share button reader thread started (volume cycle)")

                return True

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    logger.warning(f"Controller not found (attempt {attempt + 1}/{max_retries}), waiting...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Controller not found at {self.device_path} after {max_retries} attempts")
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts: {e}")
                    return False

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

                    # Clamp to per-device gimbal range (loaded from profile yaml)
                    new_pan = max(self.PAN_MIN, min(self.PAN_MAX, new_pan))
                    new_tilt = max(self.TILT_MIN, min(self.TILT_MAX, new_tilt))

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

    def _find_keyboard_event_device(self) -> Optional[str]:
        """Find the evdev path for the Xbox controller's keyboard node.

        The Share button is exposed there as KEY_F12. The event number is not
        stable across reconnects, so resolve it by device name each time.
        """
        try:
            with open('/proc/bus/input/devices') as f:
                blocks = f.read().split('\n\n')
            for block in blocks:
                if 'Xbox Wireless Controller Keyboard' in block:
                    for line in block.split('\n'):
                        if line.startswith('H: Handlers='):
                            for tok in line.split('Handlers=')[1].split():
                                if tok.startswith('event'):
                                    return f'/dev/input/{tok}'
        except Exception as e:
            logger.debug(f"Keyboard device lookup failed: {e}")
        return None

    def _cycle_volume(self):
        """Share button action: cycle volume 0→20→40→60→80→100→0.

        Posts to /audio/volume, which routes through VolumeManager (the single
        source of truth) so the new level is applied AND persists across reboots.
        """
        current_time = time.time()
        if current_time - self._last_volume_cycle_time < self._volume_cycle_cooldown:
            return
        self._last_volume_cycle_time = current_time

        # On the first press, align the cycle to the level nearest the current
        # actual volume so the first tap steps UP from where we already are.
        if self._volume_cycle_index is None:
            cur = None
            try:
                resp = self.api_request_blocking('GET', '/audio/volume', timeout=1.0)
                if resp and 'volume' in resp:
                    cur = int(resp['volume'])
            except Exception:
                pass
            if cur is None:
                # Server unreachable — start so the first tap lands on 0.
                self._volume_cycle_index = len(self._volume_levels) - 1
            else:
                self._volume_cycle_index = min(
                    range(len(self._volume_levels)),
                    key=lambda i: abs(self._volume_levels[i] - cur),
                )

        self._volume_cycle_index = (self._volume_cycle_index + 1) % len(self._volume_levels)
        volume = self._volume_levels[self._volume_cycle_index]
        logger.info(f"Share button: volume -> {volume}%")
        self.api_request('POST', '/audio/volume', {"volume": volume})

    def _share_button_loop(self):
        """Watch the controller keyboard node for the Share button (KEY_F12).

        Share is NOT on the gamepad/js0 node — xpadneo routes it to a separate
        virtual keyboard device. We read that evdev stream here and cycle the
        system volume on each press.
        """
        EV_KEY = 0x01
        KEY_F12 = 88
        EVENT_SIZE = 24          # struct input_event on 64-bit Linux
        EVENT_FMT = 'llHHi'      # timeval(2×long) + type + code + value

        dev = None
        while self.share_reader_running:
            try:
                if dev is None:
                    path = self._find_keyboard_event_device()
                    if not path:
                        time.sleep(3.0)  # controller likely disconnected; retry
                        continue
                    dev = open(path, 'rb')
                    logger.info(f"Share button reader: watching {path} for KEY_F12")

                ready, _, _ = select.select([dev], [], [], 0.5)
                if not ready:
                    continue
                data = dev.read(EVENT_SIZE)
                if not data or len(data) < EVENT_SIZE:
                    continue
                _sec, _usec, etype, code, value = struct.unpack(EVENT_FMT, data)
                if etype == EV_KEY and code == KEY_F12 and value == 1:  # press only
                    self._cycle_volume()

            except OSError:
                # Keyboard node vanished (controller disconnect) — re-find later.
                logger.debug("Share button reader: device gone, will re-find")
                try:
                    if dev:
                        dev.close()
                except Exception:
                    pass
                dev = None
                time.sleep(3.0)
            except Exception as e:
                logger.error(f"Share button reader error: {e}")
                time.sleep(1.0)

        if dev:
            try:
                dev.close()
            except Exception:
                pass

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
                    # No stick input — ensure motors are stopped.
                    if not self.state.motors_stopped:
                        # Just transitioned to neutral. Send the stop and open a
                        # short window during which 0 is re-sent every loop tick.
                        # A single dropped stop POST otherwise left the motors
                        # running their last command (e.g. reverse) until the 2s
                        # heartbeat — which felt like the robot "sticking".
                        with self.motor_lock:
                            self.set_motor_speeds(0, 0)
                        self.state.motors_stopped = True
                        self.last_motor_command_time = current_time
                        self._stop_confirm_until = current_time + 0.5
                        logger.debug("Motors stopped (no input)")
                    elif current_time < getattr(self, '_stop_confirm_until', 0.0):
                        # Confirmation window: keep re-sending 0 every ~50ms so a
                        # lost command cannot leave a wheel creeping.
                        with self.motor_lock:
                            self.set_motor_speeds(0, 0)
                        self.last_motor_command_time = current_time
                    else:
                        # Idle heartbeat to keep the motor-bus watchdog satisfied
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

    def _reopen_device(self):
        """Reopen the joystick device after a disconnect.

        A Bluetooth link blip destroys and recreates /dev/input/js0; the fd
        opened in connect() then goes permanently stale (read -> errno 19).
        Without this the controller stays dead until the process restarts.
        """
        try:
            if self.device:
                try:
                    self.device.close()
                except Exception:
                    pass
            self.device = open(self.device_path, 'rb')
            logger.info(f"Xbox controller reconnected — reopened {self.device_path}")
            self._device_disconnected = False
        except (FileNotFoundError, OSError):
            # Device node not back yet; the next retry cycle will try again.
            pass

    def read_event(self) -> Optional[Tuple]:
        """Read a single joystick event with timeout to prevent blocking"""
        try:
            # Use select to add timeout - prevents hanging if device issues
            ready, _, _ = select.select([self.device], [], [], 0.1)  # 100ms timeout
            if ready:
                event_data = self.device.read(8)
                if event_data:
                    # Device is working - clear disconnect state
                    if getattr(self, '_device_disconnected', False):
                        logger.info("Xbox controller reconnected")
                        self._device_disconnected = False
                    timestamp, value, event_type, number = struct.unpack('IhBB', event_data)
                    return (timestamp, value, event_type, number)
        except OSError as e:
            # Errno 19 = No such device — a Bluetooth link blip destroys and
            # recreates /dev/input/js0, leaving our fd stale. Reopen it.
            if e.errno == 19:
                if not getattr(self, '_device_disconnected', False):
                    logger.warning("Xbox controller disconnected — reopening device every 5s")
                    self._device_disconnected = True
                # Slow poll when disconnected to avoid log spam
                time.sleep(5.0)
                self._reopen_device()
            else:
                logger.error(f"Error reading event: {e}")
        except Exception as e:
            logger.error(f"Error reading event: {e}")
        return None

    def _calibrate_stick(self, number: int, normalized: float) -> float:
        """Correct a worn / off-center stick axis.

        Subtracts the per-axis hardware rest offset (STICK_CENTERS) and rescales
        each side independently so full physical throw still maps to +/-1.0.
        Axis numbers: 0=LX, 1=LY, 3=RX, 4=RY. A zero offset is a no-op.
        """
        center = self.STICK_CENTERS.get(number, 0.0)
        if center == 0.0:
            return normalized
        corrected = normalized - center
        # Asymmetric rescale: each side spans from center out to its rail.
        span = (1.0 - center) if corrected >= 0.0 else (1.0 + center)
        if span <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, corrected / span))

    def process_axis(self, number: int, value: int):
        """Process axis movement"""
        # Normalize to -1.0 to 1.0
        normalized = value / 32767.0

        # Center-calibrate worn/off-center sticks (axes 0,1,3,4) before deadzone.
        # Triggers (2,5) recompute from raw `value` below and are unaffected.
        if number in (0, 1, 3, 4):
            normalized = self._calibrate_stick(number, normalized)

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
                logger.debug(f"LT triggered, value: {normalized_trigger:.2f}")
            elif normalized_trigger < 0.2:
                self.lt_was_pressed = False

            self.state.left_trigger = normalized_trigger

        elif number == 3:  # Right stick X (camera pan)
            self.state.right_x = normalized
            # Store for smooth camera update loop

        elif number == 4:  # Right stick Y (camera tilt)
            self.state.right_y = -normalized
            # Store for smooth camera update loop

        elif number == 5:  # Right trigger (RT) - play "Good" only (no cycling)
            # RT ranges from 0 to 32767 (not -32767 to 32767)
            normalized_trigger = value / 32767.0
            if normalized_trigger > self.TRIGGER_DEADZONE:
                self.state.right_trigger = normalized_trigger
                current_time = time.time()
                if not hasattr(self, '_last_rt_time'):
                    self._last_rt_time = 0
                if current_time - self._last_rt_time > 0.3:  # 300ms cooldown
                    logger.info("RT: Playing 'Good'")
                    self._play_voice_command("good")
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
            # Apply motor balance correction to forward component only.
            # This corrects straight-line drift without starving the
            # outer wheel during turns (multipliers were killing right turns).
            forward_left = self.state.left_y * self.MAX_SPEED * self.LEFT_MOTOR_MULTIPLIER
            forward_right = self.state.left_y * self.MAX_SPEED * self.RIGHT_MOTOR_MULTIPLIER
            turn = -self.state.left_x * self.MAX_SPEED * self.TURN_SPEED_FACTOR

            # Tank drive mixing — turn component is equal for both motors
            left_speed = int(forward_left - turn)
            right_speed = int(forward_right + turn)

            # DEBUG: Log raw joystick to motor conversion (always log when moving)
            logger.debug(f"JOY Y={self.state.left_y:.2f} X={self.state.left_x:.2f} -> L={left_speed} R={right_speed}")

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
        Set motor speeds with proper PWM mapping for open-loop control.

        Input: left/right in range -MAX_SPEED to +MAX_SPEED (e.g., -60 to +60)
        Output: PWM in usable range 35-70% with motor balance correction
        """
        # PWM limits - wider range for gradual control
        PWM_MIN = 20  # Low starting point - motors may not move below ~30%
        PWM_MAX = 70  # Safety limit for 6V motors on 14V system

        # OPEN-LOOP MODE: Map speed to PWM range directly
        if not self.USE_PID_CONTROL:
            # Calculate PWM for each motor
            # Balance multipliers already applied in mixing stage above
            def speed_to_pwm(speed):
                if speed == 0:
                    return 0
                direction = 1 if speed > 0 else -1
                magnitude = abs(speed)
                # Map magnitude (0 to MAX_SPEED) to PWM (PWM_MIN to PWM_MAX)
                pwm = PWM_MIN + (magnitude / self.MAX_SPEED) * (PWM_MAX - PWM_MIN)
                pwm = min(PWM_MAX, pwm)
                return direction * pwm

            left_pwm = speed_to_pwm(left)
            right_pwm = speed_to_pwm(right)

            logger.debug(f"OPEN-LOOP: Speed L={left} R={right} -> PWM L={left_pwm:.1f}% R={right_pwm:.1f}%")

            # Send to motor controller - try direct GPIO first, then API
            if self.motor_controller and hasattr(self.motor_controller, 'set_motor_pwm_direct'):
                try:
                    self.motor_controller.set_motor_pwm_direct(left_pwm, right_pwm)
                    logger.debug("Sent to motor controller")
                    return
                except Exception as e:
                    logger.error(f"Direct PWM error: {e}")
            elif self.motor_bus and hasattr(self.motor_bus, 'motor_controller'):
                try:
                    self.motor_bus.motor_controller.set_motor_pwm_direct(left_pwm, right_pwm)
                    return
                except Exception as e:
                    logger.error(f"Motor bus PWM error: {e}")

            # API fallback - main process owns motor bus, send commands via HTTP
            try:
                self.session.post(
                    f"{self.API_BASE_URL}/motor/control",
                    json={"left_speed": int(left_pwm), "right_speed": int(right_pwm)},
                    timeout=0.2
                )
                return
            except Exception as e:
                logger.debug(f"Motor API error: {e}")
            return

        # PID MODE (USE_PID_CONTROL=true)
        # NOTE: Multipliers already applied in update_motor_control() - do NOT apply again here

        if self.MIN_PWM_THRESHOLD > 0:
            if left > 0 and left < self.MIN_PWM_THRESHOLD:
                left = self.MIN_PWM_THRESHOLD
            elif left < 0 and left > -self.MIN_PWM_THRESHOLD:
                left = -self.MIN_PWM_THRESHOLD
            if right > 0 and right < self.MIN_PWM_THRESHOLD:
                right = self.MIN_PWM_THRESHOLD
            elif right < 0 and right > -self.MIN_PWM_THRESHOLD:
                right = -self.MIN_PWM_THRESHOLD

        # PID MODE: Use direct PWM when PID is disabled - LEGACY FALLBACK
        if not self.USE_PID_CONTROL:
            # Try direct PWM control first (preferred for open-loop)
            if self.motor_controller and hasattr(self.motor_controller, 'set_motor_pwm_direct'):
                try:
                    self.motor_controller.set_motor_pwm_direct(float(left), float(right))
                    logger.debug(f"PWM: L={left} R={right}")
                    return
                except Exception as e:
                    logger.error(f"Direct PWM error: {e}")
            # Fallback to motor_bus if available
            elif self.motor_bus and hasattr(self.motor_bus, 'motor_controller'):
                try:
                    self.motor_bus.motor_controller.set_motor_pwm_direct(float(left), float(right))
                    logger.debug(f"Open-loop PWM (via bus): L={left}%, R={right}%")
                    return
                except Exception as e:
                    logger.error(f"Motor bus PWM error: {e}")
            # API fallback - main process owns motor bus
            try:
                self.session.post(
                    f"{self.API_BASE_URL}/motor/control",
                    json={"left_speed": int(left), "right_speed": int(right)},
                    timeout=0.2
                )
                return
            except Exception as e:
                logger.debug(f"Motor API error: {e}")
            return

        # PID MODE: Use closed-loop RPM control
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

        else:
            # No direct motor control available - use API fallback
            try:
                self.api_request('POST', '/motor/control', {
                    'left_speed': left,
                    'right_speed': right
                })
            except Exception as e:
                logger.error(f"API motor control error: {e}")

    def _fallback_pwm_control(self, left: int, right: int):
        """Fallback to direct PWM control when PID fails

        Note: Motor calibration is already applied in set_motor_speeds()
        """
        try:
            if left == 0 and right == 0:
                self.motor_controller.set_motor_speed('left', 0, 'forward')
                self.motor_controller.set_motor_speed('right', 0, 'forward')
                return

            # Set motor speeds with direction control (calibration already applied)
            if left >= 0:
                self.motor_controller.set_motor_speed('left', abs(left), 'forward')
            else:
                self.motor_controller.set_motor_speed('left', abs(left), 'backward')

            if right >= 0:
                self.motor_controller.set_motor_speed('right', abs(right), 'forward')
            else:
                self.motor_controller.set_motor_speed('right', abs(right), 'backward')

            logger.debug(f"Fallback PWM: L={left}, R={right}")

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

    def _trigger_shutdown(self):
        """Gracefully shutdown the Raspberry Pi"""
        if self._shutdown_in_progress:
            return

        self._shutdown_in_progress = True
        logger.warning("=" * 50)
        logger.warning("SHUTDOWN TRIGGERED - A+B held for 3 seconds")
        logger.warning("=" * 50)

        try:
            # Stop motors first
            self.emergency_stop()

            # Play shutdown announcement (use lowpower as fallback if no shutdown audio)
            logger.info("Playing shutdown announcement...")
            import os
            shutdown_audio = "/home/morgan/dogbot/VOICEMP3/wimz/shutting_down.mp3"
            fallback_audio = "/home/morgan/dogbot/VOICEMP3/wimz/Wimz_lowpower.mp3"

            if os.path.exists(shutdown_audio):
                audio_file = "/wimz/shutting_down.mp3"
            elif os.path.exists(fallback_audio):
                audio_file = "/wimz/Wimz_lowpower.mp3"
            else:
                audio_file = None

            if audio_file:
                self.api_request_blocking('POST', '/audio/play/file',
                    {"filepath": audio_file}, timeout=3)
                time.sleep(2.0)  # Wait for audio
            else:
                logger.warning("No shutdown audio file found")
                time.sleep(0.5)

            # Trigger system shutdown
            logger.warning("Executing sudo poweroff...")
            import subprocess
            subprocess.run(['sudo', 'poweroff'], check=False)

        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            # Try shutdown anyway
            import subprocess
            subprocess.run(['sudo', 'poweroff'], check=False)

    def process_button(self, number: int, pressed: bool):
        """Process button press with proper cooldowns"""
        if pressed:
            logger.debug(f"Button {number} pressed")
            # NOTE: notify_manual_input() removed here - buttons should NOT auto-switch to MANUAL
            # Joystick movement and triggers already call notify_manual_input() in process_axis()

        if number == 0:  # A button - Emergency stop
            self.state.a_button = pressed
            if pressed:
                logger.info("A button: Emergency stop")
                self.emergency_stop()

        elif number == 1:  # B button - Cycle commands: Sit→Speak→Stay→Quiet→LieDown→Spin
            self.state.b_button = pressed
            if pressed:
                current_time = time.time()
                # Reset to first track if idle for 2+ seconds
                if current_time - self._command_last_cycle_time > self._CYCLE_RESET_TIMEOUT:
                    self._command_cycle_index = 0
                self._command_last_cycle_time = current_time
                # Cancel any pending playback timer
                if self._command_pending_timer:
                    self._command_pending_timer.cancel()
                # Stop any playing audio
                self.api_request('POST', '/audio/stop')
                # Get current command name and advance index
                sound_name = self._command_names[self._command_cycle_index]
                self._command_cycle_index = (self._command_cycle_index + 1) % len(self._command_names)
                logger.debug(f"B button: Queued '{sound_name}' (0.3s delay)")
                # Schedule playback after 0.3s delay (allows rapid cycling)
                def play_command():
                    logger.info(f"B button: Playing '{sound_name}'")
                    self._play_voice_command(sound_name)
                self._command_pending_timer = Timer(0.3, play_command)
                self._command_pending_timer.start()

        elif number == 2:  # X button - Toggle LED
            self.state.x_button = pressed
            if pressed:
                self.toggle_led()

        elif number == 3:  # Y button - Sound
            self.state.y_button = pressed
            if pressed:
                self.play_reward_sound()

        elif number == 4:  # Left bumper - Tap=dispense, Hold(5s)=refill
            self.state.left_bumper = pressed
            if pressed:
                self._lb_press_time = time.time()
                self._lb_refill_active = False
                # Dispense one treat in background — don't block event loop
                Thread(target=self.dispense_treat_safe, daemon=True).start()
                # Start background thread to detect hold for refill
                self._lb_refill_thread = Thread(target=self._lb_hold_check, daemon=True)
                self._lb_refill_thread.start()
            else:
                # Released — stop refill AND any dispense immediately
                if self._lb_refill_active:
                    self._lb_refill_active = False
                    try:
                        self.api_request('POST', '/treat/stop')
                    except Exception:
                        pass
                    logger.info("LB released: Refill stopped")

        elif number == 5:  # Right bumper - Play "No" audio
            self.state.right_bumper = pressed
            if pressed:
                logger.info("RB: Playing 'No'")
                self._play_voice_command("no")

        elif number == 10:  # Right stick click - Center camera
            if pressed:
                self.center_camera()

        elif number == 6:  # Select/Back button (⧉) - Cycle modes
            if pressed:
                self.cycle_mode()

        elif number == 7:  # Start/Menu button (☰) - Shutdown
            if pressed:
                logger.info("START button: Triggering shutdown")
                self._trigger_shutdown()

        elif number == 8:  # Xbox Guide button - Cycle tricks (coach mode only)
            if pressed:
                self.cycle_trick()

        elif number == 9:  # Left stick click - Manual unjam treat dispenser
            if pressed:
                logger.info("Left stick click: Running manual unjam")
                Thread(target=self._run_unjam, daemon=True).start()

    # ============== AUDIO RECORDING ==============
    def handle_record_button(self):
        """
        Handle Start/Menu button for recording new talk audio.
        First press: Start recording (2 sec) + playback
        Second press within 10s: Save to VOICEMP3/talks
        """
        current_time = time.time()
        logger.debug(f"handle_record_button called at {current_time:.2f}")

        if not hasattr(self, '_last_record_button'):
            self._last_record_button = 0

        # Check recording status FIRST (BLOCKING - needs response)
        status = self.api_request_blocking('GET', '/audio/record/status')
        logger.debug(f"Recording status: {status}")

        # If recording in progress, ignore (server will also reject)
        if status and status.get('in_progress'):
            logger.debug("Recording in progress - ignoring button press")
            return

        if status and status.get('has_pending'):
            # Second press - confirm and save (BLOCKING - needs response)
            logger.info("START BUTTON: Confirming recording save")
            result = self.api_request_blocking('POST', '/audio/record/confirm')
            if result and result.get('success'):
                logger.info(f"Recording saved: {result.get('filename')}")
                # Rescan folders to include the new recording in D-pad list
                self._scan_audio_folders()
                logger.debug(f"Audio list refreshed: {len(self.TALK_TRACKS)} talks available")
            else:
                logger.warning(f"Recording save failed: {result}")
        else:
            # First press - start new recording (with cooldown to prevent double-trigger)
            if current_time - self._last_record_button < 6.0:
                logger.debug("Record button ignored (cooldown - recording may be in progress)")
                return

            self._last_record_button = current_time
            logger.info("START BUTTON: Starting new recording (2 seconds)")
            result = self.api_request_blocking('POST', '/audio/record/start')
            if result and result.get('success'):
                logger.info("Recording complete - press START again within 10s to save")
            else:
                logger.warning(f"Recording failed: {result}")

    # ============== END AUDIO RECORDING ==============

    # ============== VIDEO RECORDING ==============
    def toggle_video_recording(self):
        """
        Toggle video recording on/off via long-press of Start button.
        Records MP4 at 640x640 with AI overlays (bounding boxes, poses, behaviors).
        """
        logger.info("LONG PRESS: Toggling video recording")

        try:
            # Call toggle API endpoint
            result = self.api_request_blocking('POST', '/video/record/toggle')

            if result:
                if result.get('recording', False):
                    # Started recording
                    logger.info(f"VIDEO RECORDING STARTED: {result.get('filename')}")
                    # Play audio feedback
                    self.api_request_async('POST', '/audio/play/file', {'filepath': '/wimz/Wimz_recording.mp3'})
                else:
                    # Stopped recording
                    logger.info(f"VIDEO RECORDING STOPPED: {result.get('filename')} ({result.get('frames', 0)} frames, {result.get('duration', 0):.1f}s)")
                    # Play save audio feedback
                    self.api_request_async('POST', '/audio/play/file', {'filepath': '/wimz/Wimz_saved.mp3'})
            else:
                logger.warning("Video toggle failed - no response")
        except Exception as e:
            logger.error(f"Video toggle error: {e}")

    # ============== END VIDEO RECORDING ==============

    def _lb_hold_check(self):
        """Background thread: if LB held >5s, enter refill mode (continuous spin)"""
        # Check every 100ms for 5 seconds
        for _ in range(50):
            time.sleep(0.1)
            if not self.state.left_bumper:
                return  # Released before 5s — was a normal tap (already dispensed on press)

        self._lb_refill_active = True
        logger.info("LB held 5s: Starting refill — release to stop")

        # Start refill on server side
        try:
            self.api_request('POST', '/treat/refill/start')
        except Exception as e:
            logger.error(f"Refill start error: {e}")
            self._lb_refill_active = False
            return

        # Just poll button state — no API calls in the loop
        try:
            while self._lb_refill_active and self.state.left_bumper:
                time.sleep(0.05)
        finally:
            self._lb_refill_active = False
            try:
                self.api_request('POST', '/treat/stop')
            except Exception:
                pass
            logger.info("LB released: Refill stopped")

    def _run_unjam(self):
        """Run manual unjam via API (called from Share button thread)"""
        try:
            result = self.api_request('POST', '/treat/unjam')
            if result and result.get('success'):
                logger.info("Unjam complete")
            else:
                logger.error(f"Unjam failed: {result}")
        except Exception as e:
            logger.error(f"Unjam error: {e}")

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

    def _get_current_mode(self) -> str:
        """Get current system mode from API"""
        try:
            result = self.api_request_blocking('GET', '/mode', timeout=2)
            if result and 'current_mode' in result:
                return result.get('current_mode', 'unknown')
        except Exception as e:
            logger.warning(f"Could not get current mode: {e}")
        return 'unknown'

    def take_photo(self):
        """Take photo based on current mode:
        - MANUAL mode: Use 4K photo (camera is released)
        - Other modes: Use snapshot from AI stream (camera is busy)
        """
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_cooldown:
            return

        self.last_photo_time = current_time

        # Check current mode FIRST to decide which photo method to use
        current_mode = self._get_current_mode()
        logger.info(f"RB pressed: Taking photo (mode: {current_mode})...")

        if current_mode == 'manual':
            # MANUAL mode: Camera is released, use 4K photo
            try:
                result = self.api_request_blocking('POST', '/camera/photo', timeout=8)
                if result and result.get('success'):
                    logger.info(f"4K Photo saved: {result.get('filename', 'unknown')} ({result.get('resolution', '?')})")
                    return
                else:
                    logger.warning(f"4K photo failed: {result.get('detail', 'unknown error')}")
            except Exception as e:
                logger.error(f"4K photo error: {e}")
        else:
            # Other modes: AI is running, grab snapshot from stream
            try:
                result = self.api_request_blocking('POST', '/camera/snapshot', timeout=3)
                if result and result.get('success'):
                    logger.info(f"Snapshot saved: {result.get('filename', 'unknown')} ({result.get('resolution', '?')})")
                    return
                else:
                    logger.warning(f"Snapshot failed: {result.get('detail', 'unknown error')}")
            except Exception as e:
                logger.error(f"Snapshot error: {e}")

    def center_camera(self):
        """Center the camera to default position (per-device, from yaml)"""
        logger.info(f"Right stick click: Centering camera to pan={self.PAN_CENTER}, tilt={self.TILT_CENTER}")

        self.last_pan_angle = self.PAN_CENTER
        self.last_tilt_angle = self.TILT_CENTER

        self.api_request('POST', '/camera/pantilt', {
            "pan": self.PAN_CENTER,
            "tilt": self.TILT_CENTER,
            "smooth": False  # discrete center action — snap immediately, don't smooth
        })

    def cycle_mode(self):
        """Cycle between MANUAL, IDLE, COACH, and SILENT_GUARDIAN modes"""
        current_time = time.time()

        # Cooldown to prevent rapid cycling
        if current_time - self.last_mode_cycle_time < self.mode_cycle_cooldown:
            logger.debug("Mode cycle ignored (cooldown)")
            return

        self.last_mode_cycle_time = current_time

        # Sync index with actual system mode first
        actual_mode = self._get_current_mode()
        if actual_mode in self.cycle_modes:
            self.current_mode_index = self.cycle_modes.index(actual_mode)
            logger.debug(f"Synced mode index to {actual_mode} (index {self.current_mode_index})")

        # Move to next mode in cycle
        self.current_mode_index = (self.current_mode_index + 1) % len(self.cycle_modes)
        new_mode = self.cycle_modes[self.current_mode_index]

        logger.info(f"SELECT button: Cycling from {actual_mode} to {new_mode.upper()} mode")

        # Call API to change mode (async - don't block controller on mode change)
        # Audio announcement handled by main_treatbot._announce_mode() via mode_change event
        self.api_request('POST', '/mode/set', {"mode": new_mode})

    def cycle_trick(self):
        """Cycle through tricks and set forced trick (coach mode only)"""
        current_time = time.time()

        # Cooldown to prevent rapid cycling
        if current_time - self._last_trick_cycle_time < self._trick_cycle_cooldown:
            logger.debug("Trick cycle ignored (cooldown)")
            return

        self._last_trick_cycle_time = current_time

        # Only works in coach mode
        actual_mode = self._get_current_mode()
        if actual_mode != 'coach':
            logger.debug(f"Guide button: Not in coach mode ({actual_mode}), ignoring")
            return

        # Cycle to next trick
        self._trick_cycle_index = (self._trick_cycle_index + 1) % len(self._available_tricks)
        trick = self._available_tricks[self._trick_cycle_index]

        # Reset session cooldown - allows new session to start immediately
        self.api_request('POST', '/coaching/reset_session_cooldown')

        # Set forced trick via API. audio_pre_played=1 because we play the mp3 below
        # as press feedback — the coach engine MUST skip its own redundant TTS.
        result = self.api_request_blocking(
            'POST', f'/coaching/force_trick/{trick}?audio_pre_played=1', timeout=2)
        if result and not result.get('error'):
            logger.info(f"Trick set to: {trick} (cooldown reset)")
        else:
            logger.warning(f"Failed to set trick: {result}")

        # Play trick audio as feedback so user knows what's queued
        audio_file = self._trick_audio.get(trick, f'{trick}.mp3')
        self.api_request('POST', '/audio/play/file', {'filepath': f'/talks/{audio_file}'})

    def cycle_dog(self):
        """Cycle dog identity override: auto -> elsa -> bezik -> auto (coach mode only)"""
        current_time = time.time()

        if current_time - self._last_dog_cycle_time < self._dog_cycle_cooldown:
            logger.debug("Dog cycle ignored (cooldown)")
            return

        self._last_dog_cycle_time = current_time

        result = self.api_request_blocking('POST', '/coaching/cycle_dog', timeout=2)
        if result and not result.get('error'):
            forced = result.get('forced_dog')
            display = forced if forced else 'auto'
            logger.info(f"D-pad up: Dog identity set to: {display}")

            # Audio feedback so user knows which dog is selected
            if forced:
                self.api_request('POST', '/audio/play/file', {'filepath': f'/talks/{forced}.mp3'})
            else:
                self.api_request('POST', '/audio/play/file', {'filepath': '/talks/treat.mp3'})
        else:
            logger.warning(f"Failed to cycle dog: {result}")

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
        """Play TREAT sound (Y button) - uses per-dog voice if available"""
        logger.info("Y button: Playing 'Treat'")
        self._play_voice_command("treat")


    def play_selected_talk(self):
        """Play selected talk track (D-pad right)"""
        file_path, track_name = self.TALK_TRACKS[self.current_talk_index]
        logger.debug(f"Playing talk: {track_name} ({file_path})")
        self.api_request('POST', '/audio/play/file', {"filepath": file_path})

    def play_selected_song(self):
        """Play selected song track (D-pad left)"""
        file_path, track_name = self.SONG_TRACKS[self.current_song_index]
        logger.debug(f"Playing song: {track_name} ({file_path})")
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
                logger.debug(f"Queued song: {track_name} - press DOWN to play")
                self.last_dpad_time = current_time
            elif value > 0:  # Right - Cycle only "Quiet" and "Come"
                if not hasattr(self, '_dpad_right_index'):
                    self._dpad_right_index = 0
                # Sound names for dog-aware lookup
                dpad_sounds = ["quiet", "come"]
                sound_name = dpad_sounds[self._dpad_right_index]
                self._dpad_right_index = (self._dpad_right_index + 1) % len(dpad_sounds)
                logger.info(f"D-pad right: Playing '{sound_name}'")
                self._play_voice_command(sound_name)
                self.last_dpad_time = current_time

        elif number == 7:  # D-pad Y axis
            self.state.dpad_up = (value < 0)
            self.state.dpad_down = (value > 0)

            # Add cooldown to prevent rapid audio triggering (same as X axis)
            current_time = time.time()
            if current_time - self.last_dpad_time < self.dpad_cooldown:
                return

            if value < 0:  # Up - Cycle dog in coach mode, stop audio otherwise
                actual_mode = self._get_current_mode()
                if actual_mode == 'coach':
                    self.cycle_dog()
                else:
                    logger.info("D-pad up: Stop audio")
                    self.api_request('POST', '/audio/stop')
                self.last_dpad_time = current_time
            elif value > 0:  # Down - Play the QUEUED track
                if self.queued_track is not None:
                    file_path, track_name = self.queued_track
                    logger.info(f"Playing {self.queued_type.upper()}: {track_name} ({file_path})")
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
        logger.info("Movement: Left stick")
        logger.info("Camera: Right stick")
        logger.info("A = Emergency Stop, B = Cycle commands (Sit/Speak/Stay/Quiet/LieDown/Spin)")
        logger.info("X = Blue LED, LT = NeoPixel modes")
        logger.info("Y = Treat sound, LB = Dispense treat, RB = 'No' audio")
        logger.info("RT = 'Good' audio")
        logger.info("D-pad Right = Quiet/Come, D-pad Left = Songs")
        logger.info("SELECT = Cycle modes (MANUAL→IDLE→COACH→SILENT_GUARDIAN)")
        logger.info("START = Shutdown")
        logger.info("=== ASYNC API QUEUE ===")
        logger.info("Non-blocking button presses")
        logger.info("50ms command debouncing")
        logger.info("No more system freezes from rapid inputs")

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
        self.share_reader_running = False  # Stop Share-button reader
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

        # Stop Share-button reader thread
        if self.share_reader_thread:
            self.share_reader_thread.join(timeout=1.0)

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