#!/usr/bin/env python3
"""
Motor controller using gpioset commands - reliable fallback when Python GPIO fails
"""

import subprocess
import time
from enum import Enum

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class MotorControllerGpioset:
    """Motor controller using gpioset commands for reliability"""

    def __init__(self):
        # Motor pin assignments (from working config)
        self.MOTOR_LEFT_IN1 = 17   # Left motor direction 1
        self.MOTOR_LEFT_IN2 = 18   # Left motor direction 2
        self.MOTOR_LEFT_EN = 13    # Left motor enable (PWM)
        self.MOTOR_RIGHT_IN1 = 27  # Right motor direction 1
        self.MOTOR_RIGHT_IN2 = 22  # Right motor direction 2
        self.MOTOR_RIGHT_EN = 19   # Right motor enable (PWM)

        self.initialized = False

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
            print("✅ Gpioset motor controller initialized")
            return True

        except Exception as e:
            print(f"❌ Gpioset motor controller failed: {e}")
            return False

    def _set_pins(self, pin_states):
        """Set multiple GPIO pins using gpioset instantly (no duration)"""
        try:
            # Build gpioset command for instant pin setting
            cmd = ['gpioset', 'gpiochip0'] + [f"{pin}={value}" for pin, value in pin_states.items()]

            # Run command immediately and wait for completion
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            return result.returncode == 0
        except Exception as e:
            print(f"GPIO set error: {e}")
            return False

    def set_motor_direction(self, motor, direction):
        """Set motor direction (without PWM speed control for now)"""
        if not self.initialized:
            return False

        try:
            if motor in ['A', 'left']:
                in1, in2 = self.MOTOR_LEFT_IN1, self.MOTOR_LEFT_IN2
                # Apply direction correction for left motor
                if direction == 'forward':
                    direction = 'backward'
                elif direction == 'backward':
                    direction = 'forward'
            elif motor in ['B', 'right']:
                in1, in2 = self.MOTOR_RIGHT_IN1, self.MOTOR_RIGHT_IN2
            else:
                return False

            # Set direction pins
            pin_states = {}
            if direction == 'stop':
                pin_states[in1] = 0
                pin_states[in2] = 0
            elif direction == 'forward':
                pin_states[in1] = 1
                pin_states[in2] = 0
            elif direction == 'backward':
                pin_states[in1] = 0
                pin_states[in2] = 1

            return self._set_pins(pin_states)

        except Exception as e:
            print(f"Motor direction error: {e}")
            return False

    def tank_steering(self, direction):
        """Tank-style steering control with instant commands"""
        if not self.initialized:
            return False

        try:
            pin_states = {}

            if direction == MotorDirection.FORWARD:
                # Left motor forward (corrected): IN1=0, IN2=1, EN=1
                # Right motor forward: IN1=1, IN2=0, EN=1
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_LEFT_EN] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_EN] = 1

            elif direction == MotorDirection.BACKWARD:
                # Left motor backward (corrected): IN1=1, IN2=0, EN=1
                # Right motor backward: IN1=0, IN2=1, EN=1
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_LEFT_EN] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_EN] = 1

            elif direction == MotorDirection.LEFT:
                # Left motor backward, right motor forward
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_LEFT_EN] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_EN] = 1

            elif direction == MotorDirection.RIGHT:
                # Left motor forward, right motor backward
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_LEFT_EN] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_EN] = 1

            elif direction == MotorDirection.STOP:
                return self.stop_all()

            success = self._set_pins(pin_states)
            if success:
                print(f"Tank steering: {direction.value}")
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