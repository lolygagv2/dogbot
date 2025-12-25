#!/usr/bin/env python3
"""
TreatDispenserRobot - Unified main orchestrator class
Consolidates all functionality from multiple conversation threads
"""

import yaml
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

# Import existing working hardware controllers
from .hardware.motor_controller import MotorController, MotorDirection
from .hardware.audio_controller import AudioController
from .hardware.led_controller import LEDController, LEDMode
from .hardware.servo_controller import ServoController

# Import new unified modules (to be created)
from .vision.camera_manager import CameraManager
from .behavior.behavior_analyzer import BehaviorAnalyzer
from .behavior.reward_system import RewardSystem
from .utils.state_manager import StateManager
from .utils.event_bus import EventBus

class RobotMode(Enum):
    MANUAL = "manual"
    TRACKING = "tracking"
    TRAINING = "training"
    PATROL = "patrol"
    STOPPED = "stopped"

class TreatDispenserRobot:
    """
    Main robot orchestrator class that coordinates all subsystems.
    Consolidates functionality from all conversation threads.
    """

    def __init__(self, config_path: str = "/home/morgan/dogbot/config/robot_config.yaml"):
        """Initialize the complete robot system"""
        self.logger = self._setup_logging()
        self.logger.info("Initializing TreatDispenser Robot...")

        # Load configuration
        self.config = self._load_config(config_path)

        # System state
        self.current_mode = RobotMode.STOPPED
        self.is_running = False
        self.emergency_stop_requested = False

        # Initialize event system
        self.event_bus = EventBus()
        self.state_manager = StateManager(self.event_bus)

        # Initialize hardware subsystems (existing proven modules)
        self.hardware = self._initialize_hardware()

        # Initialize AI/vision subsystems (new unified modules)
        self.vision = self._initialize_vision()

        # Initialize behavior and reward systems
        self.behavior_analyzer = BehaviorAnalyzer(self.config['behavior'], self.event_bus)
        self.reward_system = RewardSystem(
            self.config['behavior'],
            self.hardware['audio'],
            self.hardware['leds'],
            self.hardware['servos'],
            self.event_bus
        )

        # Setup event handlers
        self._setup_event_handlers()

        # Validate initialization
        self.initialization_successful = self._validate_initialization()

        if self.initialization_successful:
            self.logger.info("TreatDispenser Robot initialization complete!")
            self._set_initial_state()
        else:
            self.logger.error("Robot initialization failed!")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('TreatDispenserRobot')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if YAML file unavailable"""
        return {
            'hardware': {'motors': {'default_speed': 50}},
            'detection': {'confidence_threshold': 0.7},
            'behavior': {'treat_cooldown_seconds': 30},
            'audio': {'default_volume': 75},
            'leds': {'brightness': 0.3},
            'missions': {'default_mode': 'tracking'},
            'safety': {'emergency_stop_timeout': 0.5}
        }

    def _initialize_hardware(self) -> Dict[str, Any]:
        """Initialize all hardware subsystems using existing proven modules"""
        hardware = {}

        # Initialize each hardware component separately to allow partial failures
        # Motors
        try:
            self.logger.info("Initializing motor controller...")
            hardware['motors'] = MotorController()
            self.logger.info("Motor controller ready")
        except Exception as e:
            self.logger.error(f"Motor controller initialization failed: {e}")
            hardware['motors'] = None

        # Audio
        try:
            self.logger.info("Initializing audio controller...")
            hardware['audio'] = AudioController()
            if hardware['audio'].is_initialized():
                self.logger.info("Audio controller ready")
            else:
                self.logger.warning("Audio controller partially initialized")
        except Exception as e:
            self.logger.error(f"Audio controller initialization failed: {e}")
            hardware['audio'] = None

        # LEDs
        try:
            self.logger.info("Initializing LED controller...")
            hardware['leds'] = LEDController()
            if hardware['leds'].is_initialized():
                self.logger.info("LED controller ready")
            else:
                self.logger.warning("LED controller partially initialized")
        except Exception as e:
            self.logger.error(f"LED controller initialization failed: {e}")
            hardware['leds'] = None

        # Servos
        try:
            self.logger.info("Initializing servo controller...")
            hardware['servos'] = ServoController()
            self.logger.info("Servo controller ready")
        except Exception as e:
            self.logger.error(f"Servo controller initialization failed: {e}")
            hardware['servos'] = None

        return hardware

    def _initialize_vision(self) -> Optional[CameraManager]:
        """Initialize vision subsystem"""
        try:
            self.logger.info("Initializing camera and vision system...")
            return CameraManager(self.config['detection'], self.event_bus)
        except Exception as e:
            self.logger.error(f"Vision initialization failed: {e}")
            return None

    def _setup_event_handlers(self):
        """Setup event handlers for inter-module communication"""
        # Dog detection events
        self.event_bus.subscribe('dog_detected', self._on_dog_detected)
        self.event_bus.subscribe('dog_lost', self._on_dog_lost)

        # Behavior events
        self.event_bus.subscribe('behavior_detected', self._on_behavior_detected)
        self.event_bus.subscribe('reward_given', self._on_reward_given)

        # System events
        self.event_bus.subscribe('emergency_stop', self._on_emergency_stop)
        self.event_bus.subscribe('hardware_fault', self._on_hardware_fault)

    def _validate_initialization(self) -> bool:
        """Validate that critical systems are initialized"""
        # Only motors are truly critical for safety
        critical_systems = ['motors']
        optional_systems = ['audio', 'leds', 'servos']

        # Check critical systems
        for system in critical_systems:
            if system not in self.hardware or self.hardware[system] is None:
                self.logger.error(f"Critical system {system} not initialized")
                return False

        # Check optional systems (warn but don't fail)
        for system in optional_systems:
            if system not in self.hardware or self.hardware[system] is None:
                self.logger.warning(f"Optional system {system} not available")
            elif hasattr(self.hardware[system], 'is_initialized'):
                if not self.hardware[system].is_initialized():
                    self.logger.warning(f"Optional system {system} partially initialized")

        return True

    def _set_initial_state(self):
        """Set robot to safe initial state"""
        try:
            # Stop all motors
            self.hardware['motors'].emergency_stop()

            # Set LEDs to idle mode
            if 'leds' in self.hardware and self.hardware['leds']:
                try:
                    self.hardware['leds'].set_mode(LEDMode.IDLE)
                except Exception as e:
                    self.logger.warning(f"Could not set initial LED mode: {e}")

            # Center camera
            if 'servos' in self.hardware and self.hardware['servos']:
                try:
                    self.hardware['servos'].center_camera()
                except Exception as e:
                    self.logger.warning(f"Could not center camera: {e}")

            # Switch to DFPlayer and play startup sound
            if 'audio' in self.hardware and self.hardware['audio']:
                try:
                    self.hardware['audio'].switch_to_dfplayer()
                    # Play startup sound with proper sound name
                    self.hardware['audio'].play_sound("DOOR_SCAN")
                except Exception as e:
                    self.logger.warning(f"Could not play startup sound: {e}")

            self.current_mode = RobotMode.MANUAL
            self.logger.info("Robot set to safe initial state")

        except Exception as e:
            self.logger.error(f"Initial state setup failed: {e}")

    # Event Handlers
    def _on_dog_detected(self, event_data):
        """Handle dog detection event"""
        if self.current_mode in [RobotMode.TRACKING, RobotMode.TRAINING]:
            self.hardware['leds'].set_mode(LEDMode.DOG_DETECTED)

            # Track with camera if vision is available
            if self.vision:
                center_x, center_y = event_data.get('center', (320, 240))
                self._track_target(center_x, center_y)

    def _on_dog_lost(self, event_data):
        """Handle dog lost event"""
        if self.current_mode in [RobotMode.TRACKING, RobotMode.TRAINING]:
            self.hardware['leds'].set_mode(LEDMode.SEARCHING)

    def _on_behavior_detected(self, event_data):
        """Handle behavior detection event"""
        behavior = event_data.get('behavior')
        confidence = event_data.get('confidence', 0.0)

        self.logger.info(f"Behavior detected: {behavior} (confidence: {confidence:.2f})")

        # Let reward system handle the logic
        if self.current_mode in [RobotMode.TRACKING, RobotMode.TRAINING]:
            self.reward_system.process_behavior(behavior, confidence)

    def _on_reward_given(self, event_data):
        """Handle reward dispensed event"""
        behavior = event_data.get('behavior')
        self.logger.info(f"🎉 Treat dispensed for behavior: {behavior}")

    def _on_emergency_stop(self, event_data):
        """Handle emergency stop event"""
        self.emergency_stop()

    def _on_hardware_fault(self, event_data):
        """Handle hardware fault event"""
        fault_type = event_data.get('fault_type', 'unknown')
        self.logger.error(f"Hardware fault detected: {fault_type}")
        self.hardware['leds'].set_mode(LEDMode.ERROR)

    def _track_target(self, center_x: int, center_y: int):
        """Track target with camera servos"""
        try:
            frame_center_x = self.config['hardware']['camera']['resolution'][0] // 2

            # Pan tracking
            pan_threshold = 50
            if center_x < frame_center_x - pan_threshold:
                current_pan = getattr(self.hardware['servos'], 'current_pan', 90)
                new_pan = max(10, current_pan - 5)
                self.hardware['servos'].set_camera_pan(new_pan)
            elif center_x > frame_center_x + pan_threshold:
                current_pan = getattr(self.hardware['servos'], 'current_pan', 90)
                new_pan = min(200, current_pan + 5)
                self.hardware['servos'].set_camera_pan(new_pan)

        except Exception as e:
            self.logger.error(f"Target tracking failed: {e}")

    # Public API Methods
    def start(self):
        """Start the robot in configured default mode"""
        if not self.initialization_successful:
            self.logger.error("Cannot start - initialization incomplete")
            return False

        self.is_running = True
        self.emergency_stop_requested = False

        # Set default mode from config
        default_mode = self.config['missions']['default_mode']
        self.set_mode(RobotMode(default_mode))

        self.logger.info(f"Robot started in {self.current_mode.value} mode")
        return True

    def stop(self):
        """Stop the robot gracefully"""
        self.logger.info("Stopping robot...")
        self.is_running = False
        self.set_mode(RobotMode.STOPPED)

    def emergency_stop(self):
        """Emergency stop all robot systems"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        self.emergency_stop_requested = True

        # Stop all hardware immediately
        try:
            self.hardware['motors'].emergency_stop()
            self.hardware['servos'].release_all_servos()
            self.hardware['leds'].set_mode(LEDMode.ERROR)
        except Exception as e:
            self.logger.error(f"Emergency stop execution failed: {e}")

        self.current_mode = RobotMode.STOPPED

    def set_mode(self, mode: RobotMode):
        """Change robot operating mode"""
        if mode == self.current_mode:
            return

        self.logger.info(f"Changing mode: {self.current_mode.value} -> {mode.value}")

        # Stop current mode activities
        if self.vision and self.current_mode in [RobotMode.TRACKING, RobotMode.PATROL]:
            self.vision.stop_detection()

        # Set new mode
        self.current_mode = mode

        # Start new mode activities
        if mode == RobotMode.TRACKING:
            if self.vision:
                self.vision.start_detection()
            self.hardware['leds'].set_mode(LEDMode.SEARCHING)

        elif mode == RobotMode.MANUAL:
            self.hardware['leds'].set_mode(LEDMode.IDLE)

        elif mode == RobotMode.STOPPED:
            self.hardware['leds'].set_mode(LEDMode.IDLE)

        self.event_bus.publish('mode_changed', {'new_mode': mode.value})

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive robot status"""
        status = {
            'initialization_successful': self.initialization_successful,
            'is_running': self.is_running,
            'current_mode': self.current_mode.value,
            'emergency_stop': self.emergency_stop_requested,
            'hardware': {},
            'vision': {'available': self.vision is not None}
        }

        # Get hardware status
        for name, controller in self.hardware.items():
            try:
                if hasattr(controller, 'get_status'):
                    status['hardware'][name] = controller.get_status()
                else:
                    status['hardware'][name] = {
                        'initialized': hasattr(controller, 'is_initialized') and controller.is_initialized()
                    }
            except Exception as e:
                status['hardware'][name] = {'error': str(e)}

        return status

    def cleanup(self):
        """Clean shutdown of all systems"""
        self.logger.info("Shutting down TreatDispenser Robot...")

        # Stop all activities
        self.stop()

        # Cleanup hardware
        for name, controller in self.hardware.items():
            try:
                if hasattr(controller, 'cleanup'):
                    controller.cleanup()
                    self.logger.info(f"{name} cleaned up")
            except Exception as e:
                self.logger.error(f"{name} cleanup failed: {e}")

        # Cleanup vision
        if self.vision:
            try:
                self.vision.cleanup()
                self.logger.info("Vision system cleaned up")
            except Exception as e:
                self.logger.error(f"Vision cleanup failed: {e}")

        self.logger.info("Robot shutdown complete")

    # Manual control methods (for testing/debugging)
    def manual_move(self, direction: MotorDirection, duration: float = 1.0, speed: int = None):
        """Manual motor control"""
        if self.current_mode != RobotMode.MANUAL:
            self.logger.warning("Manual control only available in MANUAL mode")
            return

        speed = speed or self.config['hardware']['motors']['default_speed']
        self.hardware['motors'].tank_steering(direction, speed, duration)

    def manual_servo(self, action: str, **kwargs):
        """Manual servo control"""
        # Allow servo control in tracking mode for camera adjustment
        if self.current_mode not in [RobotMode.MANUAL, RobotMode.TRACKING, RobotMode.TRAINING]:
            self.logger.warning("Servo control not available in current mode")
            return

        if 'servos' not in self.hardware or not self.hardware['servos']:
            self.logger.warning("Servo controller not available")
            return

        servo_controller = self.hardware['servos']

        try:
            if action == 'center':
                servo_controller.center_camera()
            elif action == 'scan':
                servo_controller.scan_left_right()
            elif action == 'winch':
                servo_controller.rotate_winch(**kwargs)
            elif action == 'pan':
                # Set pan angle
                angle = kwargs.get('angle', 0)
                if hasattr(servo_controller, 'set_camera_pan'):
                    servo_controller.set_camera_pan(angle)
                else:
                    self.logger.warning("Pan control not available")
            elif action == 'tilt':
                # Set tilt angle
                angle = kwargs.get('angle', 0)
                if hasattr(servo_controller, 'set_camera_tilt'):
                    servo_controller.set_camera_tilt(angle)
                else:
                    self.logger.warning("Tilt control not available")
        except Exception as e:
            self.logger.error(f"Servo control error: {e}")

    def manual_audio(self, sound_name: str):
        """Manual audio control"""
        self.hardware['audio'].play_sound(sound_name)

    def manual_leds(self, mode: LEDMode):
        """Manual LED control"""
        self.hardware['leds'].set_mode(mode)