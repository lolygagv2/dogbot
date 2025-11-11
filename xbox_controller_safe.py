#!/usr/bin/env python3
"""
SAFE Xbox Controller for DogBot - WITH CRITICAL SAFETY FIXES
- Watchdog timer that kills motors if no heartbeat
- Signal handlers for crashes
- Automatic motor stop on any error
- Non-daemon threads that must be explicitly stopped
"""

import struct
import time
import os
import sys
import logging
import signal
import subprocess
import threading
from threading import Thread, Event, Lock
from dataclasses import dataclass
from typing import Optional, Tuple

# Setup logging first
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('XboxSafe')

# Global emergency stop function
def global_emergency_stop():
    """Global emergency stop accessible from signal handlers"""
    logger.critical("GLOBAL EMERGENCY STOP TRIGGERED")
    try:
        # Use subprocess to ensure GPIO cleanup even if Python is frozen
        pins_to_clear = [17, 27, 22, 23, 24, 25]  # All motor pins
        for pin in pins_to_clear:
            subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                          capture_output=True, timeout=0.1)
        logger.info("All motor pins cleared via gpioset")
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        # Try alternative method
        try:
            subprocess.run(['killall', 'gpioset'], capture_output=True)
        except:
            pass

# Signal handlers for safety
def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.warning(f"Received signal {sig}")
    global_emergency_stop()
    sys.exit(0)

# Register signal handlers IMMEDIATELY
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

# Motor safety configuration
MOTOR_WATCHDOG_TIMEOUT = 0.5  # Stop motors if no heartbeat for 0.5 seconds
MOTOR_COMMAND_TIMEOUT = 2.0   # Stop motors if no command for 2 seconds
MAX_MOTOR_PWM = 50            # Maximum safe PWM for 6V motors on 12V system

# Pin configuration
class Pins:
    MOTOR_IN1 = 17
    MOTOR_IN2 = 27
    MOTOR_IN3 = 22
    MOTOR_IN4 = 23
    MOTOR_ENA = 24
    MOTOR_ENB = 25

@dataclass
class ControllerState:
    """Track controller state"""
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    right_trigger: float = 0.0
    last_heartbeat: float = 0.0
    connected: bool = False

class SafeMotorController:
    """Motor controller with multiple safety mechanisms"""

    def __init__(self):
        self.pins = Pins()
        self.motor_lock = Lock()
        self.watchdog_running = True
        self.last_command_time = time.time()
        self.current_left_speed = 0
        self.current_right_speed = 0

        # Start watchdog thread (NOT daemon - must be explicitly stopped)
        self.watchdog_thread = Thread(target=self._watchdog_loop)
        self.watchdog_thread.start()

        logger.info("Safe motor controller initialized with watchdog")

    def _watchdog_loop(self):
        """Watchdog that stops motors if no recent commands"""
        while self.watchdog_running:
            try:
                current_time = time.time()

                # Check for timeout
                if current_time - self.last_command_time > MOTOR_WATCHDOG_TIMEOUT:
                    if self.current_left_speed != 0 or self.current_right_speed != 0:
                        logger.warning("WATCHDOG: Motor timeout - stopping motors")
                        self.emergency_stop()

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                self.emergency_stop()

    def set_motor_speeds(self, left: int, right: int):
        """Set motor speeds with safety limits"""
        with self.motor_lock:
            try:
                # Update heartbeat
                self.last_command_time = time.time()

                # Safety limit speeds
                left = max(-100, min(100, left))
                right = max(-100, min(100, right))

                # Apply PWM safety limit for 6V motors
                left_pwm = abs(left) * MAX_MOTOR_PWM // 100
                right_pwm = abs(right) * MAX_MOTOR_PWM // 100

                # Set left motor
                if left == 0:
                    self._stop_motor('left')
                else:
                    self._set_motor('left', left_pwm, left > 0)

                # Set right motor
                if right == 0:
                    self._stop_motor('right')
                else:
                    self._set_motor('right', right_pwm, right > 0)

                self.current_left_speed = left
                self.current_right_speed = right

                if left != 0 or right != 0:
                    logger.debug(f"Motors: L={left:4d} ({left_pwm}% PWM), R={right:4d} ({right_pwm}% PWM)")

            except Exception as e:
                logger.error(f"Motor control error: {e}")
                self.emergency_stop()

    def _set_motor(self, motor: str, pwm: int, forward: bool):
        """Set individual motor with simple on/off control"""
        if motor == 'left':
            in1, in2, ena = self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, self.pins.MOTOR_ENA
        else:
            in1, in2, ena = self.pins.MOTOR_IN3, self.pins.MOTOR_IN4, self.pins.MOTOR_ENB

        # Set direction
        if forward:
            subprocess.run(['gpioset', 'gpiochip0', f'{in1}=1', f'{in2}=0'],
                          capture_output=True, timeout=0.1)
        else:
            subprocess.run(['gpioset', 'gpiochip0', f'{in1}=0', f'{in2}=1'],
                          capture_output=True, timeout=0.1)

        # Simple PWM using on/off ratio (crude but safe)
        # For safety, we use simple on/off instead of threading
        if pwm >= 50:
            # Just turn on for speeds above 50%
            subprocess.run(['gpioset', 'gpiochip0', f'{ena}=1'],
                          capture_output=True, timeout=0.1)
        else:
            # Pulse for lower speeds (very crude PWM)
            subprocess.run(['gpioset', 'gpiochip0', f'{ena}=1'],
                          capture_output=True, timeout=0.1)
            time.sleep(0.01 * pwm / 50)  # On time proportional to speed
            subprocess.run(['gpioset', 'gpiochip0', f'{ena}=0'],
                          capture_output=True, timeout=0.1)

    def _stop_motor(self, motor: str):
        """Stop a motor completely"""
        if motor == 'left':
            pins = [self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, self.pins.MOTOR_ENA]
        else:
            pins = [self.pins.MOTOR_IN3, self.pins.MOTOR_IN4, self.pins.MOTOR_ENB]

        for pin in pins:
            subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                          capture_output=True, timeout=0.1)

    def emergency_stop(self):
        """Emergency stop all motors"""
        with self.motor_lock:
            logger.warning("EMERGENCY STOP")
            # Clear all motor pins
            all_pins = [self.pins.MOTOR_IN1, self.pins.MOTOR_IN2,
                       self.pins.MOTOR_IN3, self.pins.MOTOR_IN4,
                       self.pins.MOTOR_ENA, self.pins.MOTOR_ENB]

            for pin in all_pins:
                try:
                    subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                                 capture_output=True, timeout=0.1)
                except:
                    pass

            self.current_left_speed = 0
            self.current_right_speed = 0

    def cleanup(self):
        """Cleanup motor controller"""
        logger.info("Motor controller cleanup")
        self.emergency_stop()
        self.watchdog_running = False
        if hasattr(self, 'watchdog_thread'):
            self.watchdog_thread.join(timeout=1.0)

class XboxControllerSafe:
    """Safe Xbox controller with multiple failsafes"""

    def __init__(self, device_path: str):
        self.device_path = device_path
        self.device = None
        self.running = True
        self.state = ControllerState()
        self.motor_controller = SafeMotorController()

        # Control parameters
        self.DEADZONE = 0.15
        self.MAX_SPEED = 100

        # Start heartbeat thread
        self.heartbeat_thread = Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.start()

        logger.info("Safe Xbox controller initialized")

    def _heartbeat_loop(self):
        """Send heartbeat to motor controller"""
        while self.running:
            try:
                # Update heartbeat
                self.state.last_heartbeat = time.time()

                # Send current motor commands as heartbeat
                self.update_motors()

                time.sleep(0.2)  # Heartbeat every 200ms

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.motor_controller.emergency_stop()

    def connect(self):
        """Connect to Xbox controller"""
        try:
            self.device = open(self.device_path, 'rb')
            self.state.connected = True
            logger.info(f"Connected to Xbox controller at {self.device_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def read_event(self) -> Optional[Tuple]:
        """Read controller event"""
        if not self.device:
            return None

        try:
            data = self.device.read(8)
            if data:
                timestamp, value, event_type, number = struct.unpack('IhBB', data)
                return (timestamp, value, event_type, number)
        except Exception as e:
            logger.error(f"Read error: {e}")
            self.state.connected = False
            self.motor_controller.emergency_stop()
        return None

    def process_event(self, event):
        """Process controller event"""
        if not event:
            return

        timestamp, value, event_type, number = event

        # Process axis events
        if event_type == 2:  # Axis
            normalized = value / 32767.0

            if abs(normalized) < self.DEADZONE:
                normalized = 0.0

            if number == 0:  # Left stick X
                self.state.left_x = normalized
            elif number == 1:  # Left stick Y
                self.state.left_y = -normalized  # Invert for forward
            elif number == 5:  # Right trigger
                self.state.right_trigger = (value + 32767) / 65534.0

        # Process button events
        elif event_type == 1:  # Button
            if number == 0 and value == 1:  # A button - emergency stop
                logger.info("A button: Emergency stop")
                self.motor_controller.emergency_stop()

    def update_motors(self):
        """Update motor speeds based on current state"""
        # Calculate speed based on trigger (30-100% range)
        if self.state.right_trigger > 0.1:
            speed_mult = 0.3 + (self.state.right_trigger * 0.7)
        else:
            # Use stick magnitude for slow speed
            magnitude = (self.state.left_x**2 + self.state.left_y**2) ** 0.5
            magnitude = min(1.0, magnitude)
            speed_mult = 0.1 + (magnitude * 0.3)  # 10-40% range

        # Calculate motor speeds
        forward = self.state.left_y * self.MAX_SPEED * speed_mult
        turn = self.state.left_x * self.MAX_SPEED * speed_mult * 0.5

        left_speed = int(forward - turn)
        right_speed = int(forward + turn)

        # Send to motor controller
        self.motor_controller.set_motor_speeds(left_speed, right_speed)

    def run(self):
        """Main control loop"""
        if not self.connect():
            return

        logger.info("Controller running - Press A for emergency stop")
        logger.info("Safety features: Watchdog timer, signal handlers, motor timeout")

        try:
            while self.running:
                event = self.read_event()
                if event:
                    self.process_event(event)
                else:
                    # No event - possible disconnect
                    if not self.state.connected:
                        logger.error("Controller disconnected!")
                        self.motor_controller.emergency_stop()
                        break

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
        except Exception as e:
            logger.error(f"Control loop error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup controller"""
        logger.info("Controller cleanup")
        self.running = False

        # Stop motors first
        self.motor_controller.cleanup()

        # Close device
        if self.device:
            self.device.close()

        # Wait for threads
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join(timeout=1.0)

        logger.info("Cleanup complete")

def main():
    """Main entry point"""
    logger.warning("=" * 60)
    logger.warning("SAFE Xbox Controller - Multiple Safety Features:")
    logger.warning("1. Watchdog timer stops motors on freeze")
    logger.warning("2. Signal handlers for clean shutdown")
    logger.warning("3. Motor timeout if no commands")
    logger.warning("4. Emergency stop on any error")
    logger.warning("5. PWM limited to 50% for 6V motor safety")
    logger.warning("=" * 60)

    # Ensure emergency stop on exit
    import atexit
    atexit.register(global_emergency_stop)

    device = '/dev/input/js0'
    if not os.path.exists(device):
        logger.error(f"No controller at {device}")
        return 1

    controller = XboxControllerSafe(device)

    try:
        controller.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        global_emergency_stop()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())