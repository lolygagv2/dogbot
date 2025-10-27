#!/usr/bin/env python3
"""
WIM-Z Bluetooth ESC Controller Integration
Provides gamepad control for motors, treat dispensing, and AI activation
Compatible with standard Bluetooth game controllers
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from enum import Enum

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not available - install with: pip install pygame")

# Core imports
from core.bus import get_bus, MotionEvent, RewardEvent, VisionEvent
from core.state import get_state, SystemMode

logger = logging.getLogger(__name__)

class ControllerButton(Enum):
    """Standard gamepad button mappings"""
    A = 0  # Cross on PS
    B = 1  # Circle on PS
    X = 2  # Square on PS
    Y = 3  # Triangle on PS
    L_BUMPER = 4
    R_BUMPER = 5
    BACK = 6  # Select
    START = 7
    L_STICK = 8
    R_STICK = 9

class BluetoothESCController:
    """
    Bluetooth gamepad controller for WIM-Z robot

    Controls:
    - Left stick: Movement (forward/back, turn)
    - Right stick: Camera pan/tilt
    - A button: Dispense treat
    - B button: Play sound
    - X button: Toggle AI detection
    - Y button: Emergency stop
    - Start: Switch modes
    - L/R Bumpers: Speed control
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Bluetooth ESC controller"""
        self.config = config or {}
        self.bus = get_bus()
        self.state = get_state()

        # Controller state
        self.controller = None
        self.is_connected = False
        self.running = False
        self.control_thread = None

        # Movement parameters
        self.max_speed = self.config.get('max_speed', 100)
        self.turn_speed = self.config.get('turn_speed', 50)
        self.deadzone = self.config.get('deadzone', 0.1)
        self.speed_multiplier = 1.0  # Adjustable with bumpers

        # Camera control
        self.pan_angle = 90
        self.tilt_angle = 90
        self.camera_speed = self.config.get('camera_speed', 5)

        # AI state
        self.ai_enabled = False
        self.last_treat_time = 0
        self.treat_cooldown = 5.0  # seconds

        # Control mappings (customizable)
        self.button_actions = {
            ControllerButton.A: self.dispense_treat,
            ControllerButton.B: self.play_sound,
            ControllerButton.X: self.toggle_ai,
            ControllerButton.Y: self.emergency_stop,
            ControllerButton.START: self.switch_mode,
            ControllerButton.L_BUMPER: lambda: self.adjust_speed(-0.25),
            ControllerButton.R_BUMPER: lambda: self.adjust_speed(0.25),
        }

        logger.info("Bluetooth ESC Controller initialized")

    def initialize(self) -> bool:
        """Initialize pygame and detect controller"""
        if not PYGAME_AVAILABLE:
            logger.error("pygame not available")
            return False

        try:
            pygame.init()
            pygame.joystick.init()

            # Check for controllers
            joystick_count = pygame.joystick.get_count()

            if joystick_count == 0:
                logger.warning("No controllers detected")
                return False

            # Use first available controller
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()

            name = self.controller.get_name()
            logger.info(f"Connected to controller: {name}")
            print(f"🎮 Connected: {name}")
            print(f"   Axes: {self.controller.get_numaxes()}")
            print(f"   Buttons: {self.controller.get_numbuttons()}")

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            return False

    def start(self):
        """Start controller monitoring thread"""
        if not self.is_connected:
            if not self.initialize():
                return False

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        print("\n=== WIM-Z CONTROLLER ACTIVE ===")
        print("Controls:")
        print("  Left Stick: Move/Turn")
        print("  Right Stick: Camera")
        print("  A: Treat  B: Sound")
        print("  X: AI Toggle  Y: STOP")
        print("  Bumpers: Speed adjust")
        print("================================\n")

        return True

    def stop(self):
        """Stop controller monitoring"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        # Stop motors
        self.emergency_stop()

        logger.info("Controller stopped")

    def _control_loop(self):
        """Main control loop"""
        clock = pygame.time.Clock()

        while self.running:
            try:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        self._handle_button(event.button)
                    elif event.type == pygame.JOYDEVICEREMOVED:
                        logger.warning("Controller disconnected")
                        self.is_connected = False
                        break

                # Read analog sticks
                if self.controller:
                    # Left stick - movement
                    left_x = self.controller.get_axis(0)  # Turn
                    left_y = -self.controller.get_axis(1)  # Forward/Back (inverted)

                    # Right stick - camera
                    right_x = self.controller.get_axis(3)  # Pan
                    right_y = -self.controller.get_axis(4)  # Tilt (inverted)

                    # Apply deadzone
                    if abs(left_x) < self.deadzone:
                        left_x = 0
                    if abs(left_y) < self.deadzone:
                        left_y = 0
                    if abs(right_x) < self.deadzone:
                        right_x = 0
                    if abs(right_y) < self.deadzone:
                        right_y = 0

                    # Send movement commands
                    if left_x != 0 or left_y != 0:
                        self._send_movement(left_x, left_y)

                    # Send camera commands
                    if right_x != 0 or right_y != 0:
                        self._control_camera(right_x, right_y)

                # Limit update rate
                clock.tick(30)  # 30 Hz

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(0.1)

    def _handle_button(self, button_id: int):
        """Handle button press"""
        try:
            button = ControllerButton(button_id)
            if button in self.button_actions:
                self.button_actions[button]()
        except ValueError:
            logger.debug(f"Unknown button: {button_id}")

    def _send_movement(self, turn: float, forward: float):
        """Send movement commands to motors"""
        # Calculate motor speeds
        left_speed = forward + turn
        right_speed = forward - turn

        # Apply speed multiplier
        left_speed *= self.speed_multiplier * self.max_speed
        right_speed *= self.speed_multiplier * self.max_speed

        # Clamp to max speed
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

        # Publish motion event
        self.bus.publish(MotionEvent.MOVE, {
            'left_speed': int(left_speed),
            'right_speed': int(right_speed),
            'source': 'bluetooth_esc'
        })

    def _control_camera(self, pan: float, tilt: float):
        """Control camera servos"""
        # Update angles
        self.pan_angle += pan * self.camera_speed
        self.tilt_angle += tilt * self.camera_speed

        # Clamp angles
        self.pan_angle = max(0, min(180, self.pan_angle))
        self.tilt_angle = max(0, min(180, self.tilt_angle))

        # Publish camera event
        self.bus.publish(VisionEvent.CAMERA_MOVE, {
            'pan': int(self.pan_angle),
            'tilt': int(self.tilt_angle),
            'source': 'bluetooth_esc'
        })

    def dispense_treat(self):
        """Dispense treat with cooldown"""
        current_time = time.time()
        if current_time - self.last_treat_time < self.treat_cooldown:
            remaining = self.treat_cooldown - (current_time - self.last_treat_time)
            print(f"⏳ Treat cooldown: {remaining:.1f}s")
            return

        self.last_treat_time = current_time
        print("🍖 Dispensing treat!")

        self.bus.publish(RewardEvent.DISPENSE_TREAT, {
            'source': 'bluetooth_esc',
            'manual': True
        })

    def play_sound(self):
        """Play sound effect"""
        print("🔊 Playing sound!")
        self.bus.publish('audio.play_sound', {
            'sound': 'good_dog',
            'source': 'bluetooth_esc'
        })

    def toggle_ai(self):
        """Toggle AI detection on/off"""
        self.ai_enabled = not self.ai_enabled
        mode = SystemMode.AI_ACTIVE if self.ai_enabled else SystemMode.MANUAL

        print(f"🤖 AI Detection: {'ON' if self.ai_enabled else 'OFF'}")

        self.state.set_mode(mode)
        self.bus.publish(VisionEvent.TOGGLE_DETECTION, {
            'enabled': self.ai_enabled,
            'source': 'bluetooth_esc'
        })

    def emergency_stop(self):
        """Emergency stop all motors"""
        print("🛑 EMERGENCY STOP!")

        self.bus.publish(MotionEvent.STOP, {
            'emergency': True,
            'source': 'bluetooth_esc'
        })

        # Reset camera to center
        self.pan_angle = 90
        self.tilt_angle = 90
        self.bus.publish(VisionEvent.CAMERA_HOME, {
            'source': 'bluetooth_esc'
        })

    def switch_mode(self):
        """Switch between system modes"""
        modes = [SystemMode.IDLE, SystemMode.MANUAL, SystemMode.AI_ACTIVE, SystemMode.MISSION]
        current = self.state.get_mode()

        try:
            current_idx = modes.index(current)
            next_idx = (current_idx + 1) % len(modes)
            next_mode = modes[next_idx]
        except ValueError:
            next_mode = SystemMode.MANUAL

        print(f"📱 Mode: {next_mode.value}")
        self.state.set_mode(next_mode)

    def adjust_speed(self, delta: float):
        """Adjust speed multiplier"""
        self.speed_multiplier += delta
        self.speed_multiplier = max(0.25, min(2.0, self.speed_multiplier))

        print(f"⚡ Speed: {int(self.speed_multiplier * 100)}%")

# Singleton instance
_controller_instance = None

def get_bluetooth_controller(config: Dict[str, Any] = None) -> BluetoothESCController:
    """Get singleton Bluetooth controller instance"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = BluetoothESCController(config)
    return _controller_instance

def main():
    """Test Bluetooth controller"""
    import signal

    controller = get_bluetooth_controller({
        'max_speed': 100,
        'turn_speed': 50,
        'deadzone': 0.15
    })

    if not controller.start():
        print("Failed to start controller")
        return

    def signal_handler(sig, frame):
        print("\nShutting down...")
        controller.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Controller active. Press Ctrl+C to exit.")

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()