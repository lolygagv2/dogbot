#!/usr/bin/env python3
"""
Motor controller using gpioset commands with PWM for 6V motors on 14V system
Safely controls DFRobot Devastator motors (6V rated, 2-7.5V operating)
"""

import subprocess
import time
import threading
from enum import Enum

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class MotorControllerGpiosetPWM:
    """Motor controller using gpioset with PWM for voltage control"""

    def __init__(self):
        # Motor pin assignments (from working config)
        self.MOTOR_LEFT_IN1 = 17   # Left motor direction 1
        self.MOTOR_LEFT_IN2 = 18   # Left motor direction 2
        self.MOTOR_LEFT_EN = 13    # Left motor enable (PWM)
        self.MOTOR_RIGHT_IN1 = 27  # Right motor direction 1
        self.MOTOR_RIGHT_IN2 = 22  # Right motor direction 2
        self.MOTOR_RIGHT_EN = 19   # Right motor enable (PWM)

        # Motor specs from hardware/motor_specs.py
        self.RATED_VOLTAGE = 6.0  # V
        self.MAX_VOLTAGE = 7.5    # V
        self.SUPPLY_VOLTAGE = 14.0  # V to L298N
        self.L298N_DROP = 1.4     # V typical drop
        self.EFFECTIVE_MAX = self.SUPPLY_VOLTAGE - self.L298N_DROP  # 12.6V

        # Safe PWM duty cycles for 6V motors
        self.DEFAULT_DUTY_CYCLE = 0.40  # ~5V effective (safe for 6V motor)
        self.MAX_DUTY_CYCLE = 0.60      # ~7.5V effective (absolute max)
        self.MIN_DUTY_CYCLE = 0.20      # ~2.5V effective (minimum useful)

        # PWM parameters
        self.pwm_frequency = 1000  # Hz (1ms period)
        self.pwm_period = 1.0 / self.pwm_frequency  # seconds

        # PWM control threads
        self.left_pwm_thread = None
        self.right_pwm_thread = None
        self.pwm_running = False

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
            print("✅ Gpioset PWM motor controller initialized (6V motor safe)")
            print(f"   Default voltage: {self.DEFAULT_DUTY_CYCLE * self.EFFECTIVE_MAX:.1f}V")
            return True

        except Exception as e:
            print(f"❌ Gpioset PWM motor controller failed: {e}")
            return False

    def _set_pins(self, pin_states):
        """Set multiple GPIO pins using gpioset instantly"""
        try:
            cmd = ['gpioset', 'gpiochip0'] + [f"{pin}={value}" for pin, value in pin_states.items()]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            return result.returncode == 0
        except Exception as e:
            return False

    def _pwm_control(self, enable_pin, duty_cycle, stop_flag):
        """PWM control thread for a single enable pin"""
        on_time = self.pwm_period * duty_cycle
        off_time = self.pwm_period * (1.0 - duty_cycle)

        while not stop_flag.is_set():
            # ON phase
            subprocess.run(['gpioset', 'gpiochip0', f'{enable_pin}=1'],
                         capture_output=True, timeout=0.1)
            time.sleep(on_time)

            # OFF phase
            subprocess.run(['gpioset', 'gpiochip0', f'{enable_pin}=0'],
                         capture_output=True, timeout=0.1)
            time.sleep(off_time)

        # Ensure pin is off when stopping
        subprocess.run(['gpioset', 'gpiochip0', f'{enable_pin}=0'],
                     capture_output=True, timeout=0.1)

    def tank_steering(self, direction, speed_percent=None):
        """
        Tank-style steering control with PWM speed control

        Args:
            direction: MotorDirection enum
            speed_percent: 0-100 speed percentage (maps to safe duty cycle)
        """
        if not self.initialized:
            return False

        try:
            # Stop any existing PWM threads
            self.stop_pwm()

            # Calculate duty cycle from speed percentage
            if speed_percent is None:
                duty_cycle = self.DEFAULT_DUTY_CYCLE
            else:
                # Map 0-100% to MIN_DUTY_CYCLE to MAX_DUTY_CYCLE
                speed_factor = max(0, min(100, speed_percent)) / 100.0
                duty_cycle = self.MIN_DUTY_CYCLE + (self.MAX_DUTY_CYCLE - self.MIN_DUTY_CYCLE) * speed_factor

            effective_voltage = duty_cycle * self.EFFECTIVE_MAX
            print(f"Motor control: {direction.value} at {duty_cycle*100:.0f}% PWM (~{effective_voltage:.1f}V)")

            # Set direction pins
            pin_states = {}

            if direction == MotorDirection.FORWARD:
                # Left motor forward (corrected): IN1=0, IN2=1
                # Right motor forward: IN1=1, IN2=0
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0

                # Start PWM on both motors
                self.start_dual_pwm(duty_cycle, duty_cycle)

            elif direction == MotorDirection.BACKWARD:
                # Left motor backward (corrected): IN1=1, IN2=0
                # Right motor backward: IN1=0, IN2=1
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1

                # Start PWM on both motors
                self.start_dual_pwm(duty_cycle, duty_cycle)

            elif direction == MotorDirection.LEFT:
                # Left motor backward, right motor forward
                pin_states[self.MOTOR_LEFT_IN1] = 1
                pin_states[self.MOTOR_LEFT_IN2] = 0
                pin_states[self.MOTOR_RIGHT_IN1] = 1
                pin_states[self.MOTOR_RIGHT_IN2] = 0

                # Start PWM on both motors (can adjust speeds for smoother turns)
                self.start_dual_pwm(duty_cycle * 0.7, duty_cycle)

            elif direction == MotorDirection.RIGHT:
                # Left motor forward, right motor backward
                pin_states[self.MOTOR_LEFT_IN1] = 0
                pin_states[self.MOTOR_LEFT_IN2] = 1
                pin_states[self.MOTOR_RIGHT_IN1] = 0
                pin_states[self.MOTOR_RIGHT_IN2] = 1

                # Start PWM on both motors (can adjust speeds for smoother turns)
                self.start_dual_pwm(duty_cycle, duty_cycle * 0.7)

            elif direction == MotorDirection.STOP:
                return self.stop_all()

            # Set direction pins
            success = self._set_pins(pin_states)
            return success

        except Exception as e:
            print(f"Tank steering error: {e}")
            return False

    def start_dual_pwm(self, left_duty, right_duty):
        """Start PWM control threads for both motors"""
        self.stop_pwm()  # Stop any existing threads

        self.left_stop_flag = threading.Event()
        self.right_stop_flag = threading.Event()

        self.left_pwm_thread = threading.Thread(
            target=self._pwm_control,
            args=(self.MOTOR_LEFT_EN, left_duty, self.left_stop_flag)
        )
        self.right_pwm_thread = threading.Thread(
            target=self._pwm_control,
            args=(self.MOTOR_RIGHT_EN, right_duty, self.right_stop_flag)
        )

        self.left_pwm_thread.daemon = True
        self.right_pwm_thread.daemon = True

        self.left_pwm_thread.start()
        self.right_pwm_thread.start()
        self.pwm_running = True

    def stop_pwm(self):
        """Stop PWM control threads"""
        if self.pwm_running:
            if hasattr(self, 'left_stop_flag'):
                self.left_stop_flag.set()
            if hasattr(self, 'right_stop_flag'):
                self.right_stop_flag.set()

            if self.left_pwm_thread:
                self.left_pwm_thread.join(timeout=0.5)
            if self.right_pwm_thread:
                self.right_pwm_thread.join(timeout=0.5)

            self.pwm_running = False

    def stop_all(self):
        """Stop all motors immediately"""
        try:
            # Stop PWM threads
            self.stop_pwm()

            # Set all pins to LOW
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

    # Convenience methods for simple control
    def move_forward(self, speed_percent=None):
        """Move forward at specified speed"""
        return self.tank_steering(MotorDirection.FORWARD, speed_percent)

    def move_backward(self, speed_percent=None):
        """Move backward at specified speed"""
        return self.tank_steering(MotorDirection.BACKWARD, speed_percent)

    def turn_left(self, speed_percent=None):
        """Turn left at specified speed"""
        return self.tank_steering(MotorDirection.LEFT, speed_percent)

    def turn_right(self, speed_percent=None):
        """Turn right at specified speed"""
        return self.tank_steering(MotorDirection.RIGHT, speed_percent)

    def stop(self):
        """Stop all motors"""
        return self.stop_all()

# Test function
def test_pwm_controller():
    """Test the PWM motor controller"""
    print("Testing PWM Motor Controller for 6V motors")
    print("=" * 50)

    controller = MotorControllerGpiosetPWM()
    if not controller.initialize():
        print("Failed to initialize controller")
        return

    # Test at safe speeds
    tests = [
        ("Forward slow", MotorDirection.FORWARD, 30),
        ("Forward medium", MotorDirection.FORWARD, 50),
        ("Stop", MotorDirection.STOP, 0),
        ("Backward medium", MotorDirection.BACKWARD, 50),
        ("Stop", MotorDirection.STOP, 0),
        ("Left turn", MotorDirection.LEFT, 40),
        ("Stop", MotorDirection.STOP, 0),
        ("Right turn", MotorDirection.RIGHT, 40),
        ("Stop", MotorDirection.STOP, 0),
    ]

    for name, direction, speed in tests:
        print(f"\nTest: {name}")
        controller.tank_steering(direction, speed)
        time.sleep(2)

    controller.cleanup()
    print("\nTest complete!")

if __name__ == "__main__":
    test_pwm_controller()