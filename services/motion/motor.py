#!/usr/bin/env python3
"""
Motor Control Service - Unified wrapper for manual vehicle control
Integrates with existing working motor controllers for remote control driving
"""

import time
import threading
from typing import Dict, Any, Optional
from enum import Enum

# Import existing working motor controllers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from core.hardware.motor_controller import MotorController as CoreMotorController, MotorDirection
    CORE_MOTOR_AVAILABLE = True
except ImportError:
    CORE_MOTOR_AVAILABLE = False
    print("[WARNING] Core motor controller not available")

try:
    from motor_led_camera_control import MotorController as AltMotorController
    ALT_MOTOR_AVAILABLE = True
except ImportError:
    ALT_MOTOR_AVAILABLE = False
    print("[WARNING] Alternative motor controller not available")

try:
    from core.hardware.motor_controller_gpioset import MotorControllerGpioset, MotorDirection as GpiosetDirection
    GPIOSET_MOTOR_AVAILABLE = True
except ImportError:
    GPIOSET_MOTOR_AVAILABLE = False
    print("[WARNING] Gpioset motor controller not available")

from core.bus import get_bus, MotionEvent
from core.state import get_state

class MovementMode(Enum):
    MANUAL = "manual"
    AUTO = "auto"
    DISABLED = "disabled"

class MotorService:
    """
    Unified motor control service for manual vehicle control
    Wraps existing working motor controllers and provides unified interface
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()

        # Motor controller (try core first, fallback to alt)
        self.motor_controller = None
        self.controller_type = None

        # Movement state
        self.movement_mode = MovementMode.DISABLED
        self.current_speed = 0
        self.current_direction = None
        self.is_moving = False

        # Auto-stop safety
        self.auto_stop_timer = None
        self.max_movement_duration = 10.0  # Safety: auto-stop after 10 seconds

        # Manual control tracking
        self.manual_control_active = False
        self.last_command_time = 0

        # Status tracking
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize motor service with available controller"""
        try:
            # Try core motor controller first (lgpio-based)
            if CORE_MOTOR_AVAILABLE:
                self.motor_controller = CoreMotorController()
                if self.motor_controller.is_initialized():
                    self.controller_type = "core_lgpio"
                    print("✅ Motor service using core lgpio controller")
                else:
                    self.motor_controller = None

            # Fallback to alternative controller (RPi.GPIO-based)
            if not self.motor_controller and ALT_MOTOR_AVAILABLE:
                self.motor_controller = AltMotorController()
                if self.motor_controller.initialize():
                    self.controller_type = "alt_rpi_gpio"
                    print("✅ Motor service using alternative RPi.GPIO controller")
                else:
                    self.motor_controller = None

            # Final fallback to gpioset controller (most reliable)
            if not self.motor_controller and GPIOSET_MOTOR_AVAILABLE:
                self.motor_controller = MotorControllerGpioset()
                if self.motor_controller.initialize():
                    self.controller_type = "gpioset"
                    print("✅ Motor service using gpioset controller (safe fallback)")
                else:
                    self.motor_controller = None

            if not self.motor_controller:
                print("❌ No motor controllers available")
                return False

            # Set initial mode
            self.movement_mode = MovementMode.MANUAL

            # Subscribe to motion events from Bluetooth controller
            self.bus.subscribe('motion', self._handle_motion_event)

            # Subscribe to emergency events
            self.bus.subscribe('emergency.*', self._handle_emergency)

            self.initialized = True
            print(f"✅ Motor service initialized (controller: {self.controller_type})")
            return True

        except Exception as e:
            print(f"❌ Motor service initialization failed: {e}")
            return False

    def set_movement_mode(self, mode: MovementMode) -> bool:
        """Set movement mode (manual/auto/disabled)"""
        try:
            old_mode = self.movement_mode
            self.movement_mode = mode

            # Stop movement when switching modes
            if old_mode != mode:
                self.emergency_stop()

            # Update system state
            self.state.hardware.motor_enabled = (mode != MovementMode.DISABLED)

            # Publish event
            from core.bus import MotionEvent
            self.bus.publish(MotionEvent('mode_changed', {
                'old_mode': old_mode.value,
                'new_mode': mode.value
            }))

            print(f"🚗 Movement mode changed: {old_mode.value} → {mode.value}")
            return True

        except Exception as e:
            print(f"❌ Failed to set movement mode: {e}")
            return False

    def manual_drive(self, direction: str, speed: Optional[int] = None, duration: Optional[float] = None) -> bool:
        """
        Manual driving control (RC car style)

        Args:
            direction: 'forward', 'backward', 'left', 'right', 'stop'
            speed: Speed percentage (0-100), defaults to system setting
            duration: Max duration in seconds, defaults to safety limit
        """
        if not self.initialized or self.movement_mode != MovementMode.MANUAL:
            print(f"❌ Manual drive not available (mode: {self.movement_mode.value})")
            return False

        try:
            # Set defaults
            if speed is None:
                speed = 50  # Default moderate speed
            if duration is None:
                duration = self.max_movement_duration

            speed = max(0, min(100, speed))  # Clamp to valid range

            # Rate limiting: Don't send commands too frequently (prevents PWM overflow)
            now = time.time()
            min_command_interval = 0.05  # 50ms minimum between commands (20Hz max)
            if hasattr(self, '_last_command_time') and (now - self._last_command_time) < min_command_interval:
                return True  # Silently succeed to avoid spamming
            self._last_command_time = now

            # Clear any existing auto-stop timer
            if self.auto_stop_timer:
                self.auto_stop_timer.cancel()

            # Convert direction string to enum for core controller
            if self.controller_type == "core_lgpio":
                direction_map = {
                    'forward': MotorDirection.FORWARD,
                    'backward': MotorDirection.BACKWARD,
                    'left': MotorDirection.LEFT,
                    'right': MotorDirection.RIGHT,
                    'stop': MotorDirection.STOP
                }

                if direction not in direction_map:
                    print(f"❌ Invalid direction: {direction}")
                    return False

                motor_direction = direction_map[direction]
                success = self.motor_controller.tank_steering(motor_direction, speed, None, "reduce_interference")

            elif self.controller_type == "alt_rpi_gpio":
                # Use alternative controller methods
                if direction == 'forward':
                    self.motor_controller.set_motor_speed(speed, speed)
                elif direction == 'backward':
                    self.motor_controller.set_motor_speed(-speed, -speed)
                elif direction == 'left':
                    self.motor_controller.set_motor_speed(-speed, speed)
                elif direction == 'right':
                    self.motor_controller.set_motor_speed(speed, -speed)
                elif direction == 'stop':
                    self.motor_controller.stop()
                else:
                    print(f"❌ Invalid direction: {direction}")
                    return False
                success = True

            elif self.controller_type == "gpioset":
                # Use gpioset controller (most reliable fallback)
                direction_map = {
                    'forward': GpiosetDirection.FORWARD,
                    'backward': GpiosetDirection.BACKWARD,
                    'left': GpiosetDirection.LEFT,
                    'right': GpiosetDirection.RIGHT,
                    'stop': GpiosetDirection.STOP
                }

                if direction not in direction_map:
                    print(f"❌ Invalid direction: {direction}")
                    return False

                motor_direction = direction_map[direction]
                success = self.motor_controller.tank_steering(motor_direction)

            else:
                print("❌ No motor controller available")
                return False

            if success:
                # Update state
                self.current_direction = direction
                self.current_speed = speed if direction != 'stop' else 0
                self.is_moving = (direction != 'stop')
                self.manual_control_active = True
                self.last_command_time = time.time()

                # Set auto-stop timer for safety
                if direction != 'stop':
                    self.auto_stop_timer = threading.Timer(duration, self._auto_stop_callback)
                    self.auto_stop_timer.start()

                # Publish movement event
                self.bus.publish(MotionEvent('movement', {
                    'direction': direction,
                    'speed': speed,
                    'duration': duration,
                    'manual': True
                }))

                print(f"🚗 Manual drive: {direction} at {speed}% for {duration}s")
                return True
            else:
                print(f"❌ Motor controller failed to execute: {direction}")
                return False

        except Exception as e:
            print(f"❌ Manual drive error: {e}")
            self.emergency_stop()
            return False

    def keyboard_control(self, key: str) -> bool:
        """
        Process keyboard input for manual control

        Args:
            key: Keyboard key ('w', 'a', 's', 'd', 'space', etc.)
        """
        if self.movement_mode != MovementMode.MANUAL:
            return False

        # Movement speed for keyboard control
        speed = 60
        duration = 0.5  # Short bursts for responsive control

        key_map = {
            'w': 'forward',
            'W': 'forward',
            'up': 'forward',
            's': 'backward',
            'S': 'backward',
            'down': 'backward',
            'a': 'left',
            'A': 'left',
            'left': 'left',
            'd': 'right',
            'D': 'right',
            'right': 'right',
            'space': 'stop',
            ' ': 'stop',
            'stop': 'stop'
        }

        if key in key_map:
            direction = key_map[key]
            return self.manual_drive(direction, speed, duration)

        return False

    def emergency_stop(self):
        """Emergency stop - immediately halt all motors"""
        try:
            # Cancel auto-stop timer
            if self.auto_stop_timer:
                self.auto_stop_timer.cancel()
                self.auto_stop_timer = None

            # Stop motors using appropriate method
            if self.controller_type == "core_lgpio":
                self.motor_controller.emergency_stop()
            elif self.controller_type == "alt_rpi_gpio":
                self.motor_controller.stop()

            # Update state
            self.current_direction = 'stop'
            self.current_speed = 0
            self.is_moving = False

            # Publish emergency stop event
            self.bus.publish(MotionEvent('emergency_stop', {
                'reason': 'manual_emergency_stop'
            }))

            print("🛑 Emergency stop activated")

        except Exception as e:
            print(f"❌ Emergency stop error: {e}")

    def _auto_stop_callback(self):
        """Auto-stop callback for safety timeout"""
        print("⏰ Auto-stop triggered (safety timeout)")
        self.emergency_stop()

    def _handle_emergency(self, event_type: str, data: Dict[str, Any]):
        """Handle emergency events from event bus"""
        print(f"🚨 Emergency event received: {event_type}")
        self.emergency_stop()
        self.movement_mode = MovementMode.DISABLED

    def get_status(self) -> Dict[str, Any]:
        """Get current motor service status"""
        return {
            'initialized': self.initialized,
            'controller_type': self.controller_type,
            'movement_mode': self.movement_mode.value,
            'current_direction': self.current_direction,
            'current_speed': self.current_speed,
            'is_moving': self.is_moving,
            'manual_control_active': self.manual_control_active,
            'last_command_time': self.last_command_time,
            'auto_stop_active': self.auto_stop_timer is not None,
            'motor_controller_status': self.motor_controller.get_status() if hasattr(self.motor_controller, 'get_status') else None
        }

    def _handle_motion_event(self, event):
        """Handle motion events from Bluetooth controller"""
        try:
            if event.subtype == 'MOVE' and self.movement_mode == MovementMode.MANUAL:
                data = event.data
                left_speed = data.get('left_speed', 0)
                right_speed = data.get('right_speed', 0)

                # Control motors directly with differential drive
                if self.controller_type == "gpioset":
                    # For gpioset controller, use differential drive
                    if left_speed == 0 and right_speed == 0:
                        self.motor_controller.stop()
                    else:
                        # Convert to direction and speed
                        if left_speed > 0 and right_speed > 0:
                            # Forward
                            self.motor_controller.move_forward(max(abs(left_speed), abs(right_speed)))
                        elif left_speed < 0 and right_speed < 0:
                            # Backward
                            self.motor_controller.move_backward(max(abs(left_speed), abs(right_speed)))
                        elif left_speed > right_speed:
                            # Turn right
                            self.motor_controller.turn_right(abs(left_speed - right_speed))
                        else:
                            # Turn left
                            self.motor_controller.turn_left(abs(right_speed - left_speed))
                else:
                    # For other controllers, implement differential drive
                    self._differential_drive(left_speed, right_speed)

                print(f"🎮 Motor command: L={left_speed}, R={right_speed}")

        except Exception as e:
            print(f"❌ Motion event handling error: {e}")

    def _handle_emergency(self, event):
        """Handle emergency stop events"""
        self.emergency_stop()
        print("🛑 Emergency stop activated from event")

    def _differential_drive(self, left_speed: int, right_speed: int):
        """Implement differential drive for tank steering"""
        # This would need implementation based on specific motor controller
        # For now, convert to simple forward/turn commands
        avg_speed = (left_speed + right_speed) / 2
        turn = (left_speed - right_speed) / 2

        if abs(avg_speed) > 5:
            if avg_speed > 0:
                self.motor_controller.move_forward(abs(avg_speed))
            else:
                self.motor_controller.move_backward(abs(avg_speed))
        elif abs(turn) > 5:
            if turn > 0:
                self.motor_controller.turn_left(abs(turn))
            else:
                self.motor_controller.turn_right(abs(turn))
        else:
            self.motor_controller.stop()

    def cleanup(self):
        """Clean up motor service"""
        print("🧹 Cleaning up motor service...")

        try:
            # Emergency stop
            self.emergency_stop()

            # Cancel timers
            if self.auto_stop_timer:
                self.auto_stop_timer.cancel()

            # Cleanup controller
            if self.motor_controller and hasattr(self.motor_controller, 'cleanup'):
                self.motor_controller.cleanup()

            self.initialized = False
            print("✅ Motor service cleanup complete")

        except Exception as e:
            print(f"❌ Motor service cleanup error: {e}")

# Global service instance
_motor_service = None

def get_motor_service() -> MotorService:
    """Get global motor service instance"""
    global _motor_service
    if _motor_service is None:
        _motor_service = MotorService()
    return _motor_service

# Test function
def test_motor_service():
    """Test motor service functionality"""
    print("🧪 Testing Motor Service...")

    service = get_motor_service()

    if not service.initialize():
        print("❌ Motor service initialization failed")
        return

    print("✅ Motor service initialized")

    # Test manual mode
    service.set_movement_mode(MovementMode.MANUAL)

    # Test movements
    movements = [
        ('forward', 30, 1.0),
        ('stop', 0, 0.5),
        ('backward', 30, 1.0),
        ('stop', 0, 0.5),
        ('left', 40, 0.8),
        ('stop', 0, 0.5),
        ('right', 40, 0.8),
        ('stop', 0, 0.5)
    ]

    for direction, speed, duration in movements:
        print(f"Testing: {direction} at {speed}% for {duration}s")
        service.manual_drive(direction, speed, duration)
        time.sleep(duration + 0.5)

    # Test keyboard control
    print("Testing keyboard controls...")
    for key in ['w', 's', 'a', 'd', 'space']:
        print(f"Testing key: {key}")
        service.keyboard_control(key)
        time.sleep(1.0)

    # Cleanup
    service.cleanup()
    print("✅ Motor service test complete")

if __name__ == "__main__":
    test_motor_service()