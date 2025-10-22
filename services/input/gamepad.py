#!/usr/bin/env python3
"""
Bluetooth Gamepad Input Service for TreatBot
Supports Xbox, PlayStation, and generic Bluetooth controllers
"""

import pygame
import time
import threading
from typing import Dict, Callable, Optional
from dataclasses import dataclass

@dataclass
class GamepadState:
    """Current gamepad input state"""
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False
    button_start: bool = False
    button_select: bool = False
    left_bumper: bool = False
    right_bumper: bool = False
    left_trigger: float = 0.0
    right_trigger: float = 0.0

class GamepadService:
    """Bluetooth gamepad input service"""

    def __init__(self):
        self.joystick = None
        self.running = False
        self.thread = None
        self.state = GamepadState()

        # Callbacks for different input types
        self.movement_callback: Optional[Callable] = None
        self.button_callback: Optional[Callable] = None

        # Movement deadzone
        self.deadzone = 0.15

        # Initialize pygame
        pygame.init()
        pygame.joystick.init()

    def scan_controllers(self):
        """Scan for available Bluetooth controllers"""
        pygame.joystick.quit()
        pygame.joystick.init()

        controller_count = pygame.joystick.get_count()
        controllers = []

        for i in range(controller_count):
            joystick = pygame.joystick.Joystick(i)
            controllers.append({
                'id': i,
                'name': joystick.get_name(),
                'guid': joystick.get_guid()
            })

        return controllers

    def connect_controller(self, controller_id: int = 0) -> bool:
        """Connect to a specific controller"""
        try:
            if pygame.joystick.get_count() == 0:
                print("❌ No controllers detected")
                return False

            if controller_id >= pygame.joystick.get_count():
                print(f"❌ Controller {controller_id} not found")
                return False

            self.joystick = pygame.joystick.Joystick(controller_id)
            self.joystick.init()

            print(f"✅ Connected to: {self.joystick.get_name()}")
            print(f"   Axes: {self.joystick.get_numaxes()}")
            print(f"   Buttons: {self.joystick.get_numbuttons()}")
            print(f"   Hats: {self.joystick.get_numhats()}")

            return True

        except Exception as e:
            print(f"❌ Controller connection failed: {e}")
            return False

    def set_movement_callback(self, callback: Callable):
        """Set callback for movement input (left stick, dpad)"""
        self.movement_callback = callback

    def set_button_callback(self, callback: Callable):
        """Set callback for button input"""
        self.button_callback = callback

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to analog stick values"""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _normalize_stick_input(self, x: float, y: float) -> tuple:
        """Convert stick input to movement direction and speed"""
        x = self._apply_deadzone(x)
        y = self._apply_deadzone(-y)  # Invert Y axis (up is negative in pygame)

        # Calculate magnitude for speed
        magnitude = (x*x + y*y) ** 0.5
        speed = min(int(magnitude * 100), 100)  # Convert to 0-100%

        # Determine primary direction
        if magnitude < 0.3:
            return 'stop', 0
        elif abs(x) > abs(y):
            return 'right' if x > 0 else 'left', speed
        else:
            return 'forward' if y > 0 else 'backward', speed

    def _update_state(self):
        """Update gamepad state from pygame events"""
        if not self.joystick:
            return

        try:
            # Update analog sticks
            if self.joystick.get_numaxes() >= 2:
                self.state.left_stick_x = self.joystick.get_axis(0)
                self.state.left_stick_y = self.joystick.get_axis(1)

            if self.joystick.get_numaxes() >= 4:
                self.state.right_stick_x = self.joystick.get_axis(2)
                self.state.right_stick_y = self.joystick.get_axis(3)

            # Update triggers
            if self.joystick.get_numaxes() >= 6:
                self.state.left_trigger = max(0, self.joystick.get_axis(4))
                self.state.right_trigger = max(0, self.joystick.get_axis(5))

            # Update D-pad (hat)
            if self.joystick.get_numhats() > 0:
                hat = self.joystick.get_hat(0)
                self.state.dpad_left = hat[0] < 0
                self.state.dpad_right = hat[0] > 0
                self.state.dpad_down = hat[1] < 0
                self.state.dpad_up = hat[1] > 0

            # Update buttons (Xbox/PS controller mapping)
            if self.joystick.get_numbuttons() >= 10:
                self.state.button_a = self.joystick.get_button(0)      # A/Cross
                self.state.button_b = self.joystick.get_button(1)      # B/Circle
                self.state.button_x = self.joystick.get_button(2)      # X/Square
                self.state.button_y = self.joystick.get_button(3)      # Y/Triangle
                self.state.left_bumper = self.joystick.get_button(4)   # LB/L1
                self.state.right_bumper = self.joystick.get_button(5)  # RB/R1
                self.state.button_select = self.joystick.get_button(6) # Back/Select
                self.state.button_start = self.joystick.get_button(7)  # Start/Options

        except Exception as e:
            print(f"State update error: {e}")

    def _input_loop(self):
        """Main input processing loop"""
        print("🎮 Gamepad input loop started")

        while self.running:
            try:
                # Process pygame events
                pygame.event.pump()

                # Update gamepad state
                self._update_state()

                # Handle movement input (left stick or D-pad)
                direction, speed = 'stop', 0

                # Check left stick first
                stick_dir, stick_speed = self._normalize_stick_input(
                    self.state.left_stick_x, self.state.left_stick_y
                )

                if stick_speed > 0:
                    direction, speed = stick_dir, stick_speed
                else:
                    # Fall back to D-pad
                    if self.state.dpad_up:
                        direction, speed = 'forward', 60
                    elif self.state.dpad_down:
                        direction, speed = 'backward', 60
                    elif self.state.dpad_left:
                        direction, speed = 'left', 60
                    elif self.state.dpad_right:
                        direction, speed = 'right', 60

                # Send movement commands
                if self.movement_callback:
                    self.movement_callback(direction, speed)

                # Handle button presses
                if self.button_callback:
                    if self.state.button_a:  # Treat dispenser
                        self.button_callback('treat')
                    if self.state.button_b:  # Emergency stop
                        self.button_callback('emergency_stop')

                time.sleep(0.05)  # 20Hz update rate

            except Exception as e:
                print(f"Input loop error: {e}")
                time.sleep(0.1)

        print("🎮 Gamepad input loop stopped")

    def start(self) -> bool:
        """Start the gamepad input service"""
        if self.running:
            print("⚠️ Gamepad service already running")
            return True

        if not self.joystick:
            print("❌ No controller connected")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._input_loop, daemon=True)
        self.thread.start()

        print("✅ Gamepad service started")
        return True

    def stop(self):
        """Stop the gamepad input service"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)
            print("✅ Gamepad service stopped")

    def cleanup(self):
        """Clean up pygame resources"""
        self.stop()
        if self.joystick:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()

# Service instance getter
_gamepad_service = None

def get_gamepad_service() -> GamepadService:
    """Get the global gamepad service instance"""
    global _gamepad_service
    if _gamepad_service is None:
        _gamepad_service = GamepadService()
    return _gamepad_service