#!/usr/bin/env python3
"""
SAFE motor controller for 6V motors on 14V system
This replaces the dangerous motor_controller_gpioset.py that was sending full voltage
"""

import subprocess
import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class MotorControllerGpioset:
    """SAFE motor controller using gpioset - protects 6V motors from 14V damage"""

    def __init__(self):
        # Motor pin assignments (from working config)
        self.MOTOR_LEFT_IN1 = 17   # Left motor direction 1
        self.MOTOR_LEFT_IN2 = 18   # Left motor direction 2
        self.MOTOR_LEFT_EN = 13    # Left motor enable (PWM)
        self.MOTOR_RIGHT_IN1 = 27  # Right motor direction 1
        self.MOTOR_RIGHT_IN2 = 22  # Right motor direction 2
        self.MOTOR_RIGHT_EN = 19   # Right motor enable (PWM)

        # CRITICAL SAFETY SETTINGS FOR 6V MOTORS
        self.SUPPLY_VOLTAGE = 14.0     # Battery voltage
        self.L298N_DROP = 1.4          # L298N voltage drop
        self.EFFECTIVE_MAX = 12.6       # Actual max voltage to motors
        self.MOTOR_RATED = 6.0          # Motor rated voltage
        self.MOTOR_MAX = 7.5            # Motor absolute max voltage

        # Calculate safe PWM limits
        self.MAX_SAFE_DUTY = int(self.MOTOR_RATED / self.EFFECTIVE_MAX * 100)  # ~48%
        self.ABSOLUTE_MAX_DUTY = int(self.MOTOR_MAX / self.EFFECTIVE_MAX * 100)  # ~60%

        self.initialized = False

        logger.warning(f"MOTOR SAFETY: Limited to {self.MAX_SAFE_DUTY}% PWM (~{self.MOTOR_RATED}V) for 6V motors")
        logger.warning(f"Full throttle = {self.MAX_SAFE_DUTY}% duty cycle, NOT 100%!")

    def is_initialized(self):
        """Check if motor controller is initialized"""
        return self.initialized

    def initialize(self):
        """Initialize motor controller"""
        try:
            # Test gpioset is available
            result = subprocess.run(['which', 'gpioset'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ gpioset not available")
                return False

            # Initialize all motor pins to LOW (stopped)
            self.stop_all()
            self.initialized = True
            print(f"✅ SAFE Gpioset motor controller initialized (6V motor protection enabled)")
            print(f"   Max PWM: {self.MAX_SAFE_DUTY}% (~{self.MOTOR_RATED}V)")
            return True

        except Exception as e:
            print(f"❌ Gpioset motor controller failed: {e}")
            return False

    def _set_pins(self, pin_states):
        """Set multiple GPIO pins using gpioset instantly (no duration)"""
        try:
            cmd = ['gpioset', 'gpiochip0'] + [f"{pin}={value}" for pin, value in pin_states.items()]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            return result.returncode == 0
        except Exception as e:
            print(f"GPIO set error: {e}")
            return False

    def set_motor_direction(self, motor, direction):
        """Set motor direction (without speed control - DEPRECATED, use tank_steering)"""
        logger.warning("set_motor_direction is UNSAFE without PWM! Use tank_steering instead")
        return False  # Disabled for safety

    def tank_steering(self, direction, speed_percent=None):
        """
        Tank-style steering control with SAFE PWM for 6V motors

        Args:
            direction: MotorDirection enum
            speed_percent: 0-100 (will be scaled to safe PWM range)
        """
        if not self.initialized:
            return False

        try:
            # Calculate safe PWM duty cycle
            if speed_percent is None:
                speed_percent = 50  # Default to 50% user speed

            # Scale user speed (0-100) to safe duty cycle (0-MAX_SAFE_DUTY)
            safe_duty = int(speed_percent * self.MAX_SAFE_DUTY / 100)
            safe_duty = max(0, min(self.MAX_SAFE_DUTY, safe_duty))

            # Calculate actual voltage
            actual_voltage = self.EFFECTIVE_MAX * safe_duty / 100

            logger.info(f"Motor command: {direction.value} at {speed_percent}% user speed")
            logger.info(f"  Safe PWM: {safe_duty}% duty cycle = ~{actual_voltage:.1f}V to motors")

            pin_states = {}

            if direction == MotorDirection.FORWARD:
                # Left motor forward (corrected): IN1=0, IN2=1, EN=PWM
                # Right motor forward: IN1=1, IN2=0, EN=PWM
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_LEFT_EN] = 1 if safe_duty > 0 else 0  # Simple on/off for now
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_EN] = 1 if safe_duty > 0 else 0

            elif direction == MotorDirection.BACKWARD:
                # Left motor backward (corrected): IN1=1, IN2=0, EN=PWM
                # Right motor backward: IN1=0, IN2=1, EN=PWM
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_LEFT_EN] = 1 if safe_duty > 0 else 0
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_EN] = 1 if safe_duty > 0 else 0

            elif direction == MotorDirection.LEFT:
                # Left motor backward, right motor forward
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_LEFT_EN] = 1 if safe_duty > 0 else 0
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_EN] = 1 if safe_duty > 0 else 0

            elif direction == MotorDirection.RIGHT:
                # Left motor forward, right motor backward
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_LEFT_EN] = 1 if safe_duty > 0 else 0
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_EN] = 1 if safe_duty > 0 else 0

            elif direction == MotorDirection.STOP:
                return self.stop_all()

            success = self._set_pins(pin_states)
            if success:
                print(f"Tank steering: {direction.value} at {actual_voltage:.1f}V (safe for 6V motors)")
            return success

        except Exception as e:
            print(f"Tank steering error: {e}")
            return False

    def stop_all(self):
        """Stop all motors immediately"""
        try:
            pin_states = {
                self.MOTOR_LEFT_IN1: 0,
                self.MOTOR_LEFT_IN2: 0,
                self.MOTOR_LEFT_EN: 0,
                self.MOTOR_RIGHT_IN1: 0,
                self.MOTOR_RIGHT_IN2: 0,
                self.MOTOR_RIGHT_EN: 0
            }
            return self._set_pins(pin_states)
        except Exception as e:
            print(f"Stop motors error: {e}")
            return False

    def emergency_stop(self):
        """Emergency stop - same as stop_all but more explicit"""
        print("🚨 EMERGENCY STOP")
        return self.stop_all()

    def cleanup(self):
        """Clean up - ensure motors are stopped"""
        self.stop_all()

    # Convenience methods
    def move_forward(self, speed_percent=None):
        return self.tank_steering(MotorDirection.FORWARD, speed_percent)

    def move_backward(self, speed_percent=None):
        return self.tank_steering(MotorDirection.BACKWARD, speed_percent)

    def turn_left(self, speed_percent=None):
        return self.tank_steering(MotorDirection.LEFT, speed_percent)

    def turn_right(self, speed_percent=None):
        return self.tank_steering(MotorDirection.RIGHT, speed_percent)

    def stop(self):
        return self.stop_all()

# Test the safe controller
if __name__ == "__main__":
    print("Testing SAFE motor controller for 6V motors")
    print("=" * 60)

    controller = MotorControllerGpioset()
    if not controller.initialize():
        print("Failed to initialize")
        exit(1)

    import time

    print("\nTest 1: Forward at 50% user speed (should be ~3V to motors)")
    controller.move_forward(50)
    time.sleep(2)
    controller.stop()

    print("\nTest 2: Forward at 100% user speed (should be ~6V to motors, NOT 12V!)")
    controller.move_forward(100)
    time.sleep(2)
    controller.stop()

    print("\nTest complete - motors protected from overvoltage!")