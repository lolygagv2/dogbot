#!/usr/bin/env python3
"""
PID Controller for Motor Speed Control
Implements proportional-integral-derivative control for precise motor speed matching
"""

import time
from typing import Optional

class PIDController:
    """PID Controller for motor speed regulation"""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 output_min: float = 0.0, output_max: float = 100.0,
                 integral_max: float = None):
        """
        Initialize PID controller

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_max: Maximum integral windup limit
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Output constraints
        self.output_min = output_min
        self.output_max = output_max

        # Anti-windup protection
        self.integral_max = integral_max if integral_max is not None else (output_max - output_min) * 10

        # State variables
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

        # Performance tracking
        self.total_calls = 0
        self.avg_error = 0.0

    def update(self, setpoint: float, current_value: float, dt: Optional[float] = None) -> float:
        """
        Calculate PID output

        Args:
            setpoint: Desired target value (RPM)
            current_value: Current measured value (RPM)
            dt: Time delta in seconds (auto-calculated if None)

        Returns:
            PID output value (PWM percentage)
        """
        current_time = time.time()

        # Calculate time delta
        if dt is None:
            if self.last_time is not None:
                dt = current_time - self.last_time
            else:
                dt = 0.0

        self.last_time = current_time

        # Calculate error
        error = setpoint - current_value

        # Proportional term
        proportional = self.kp * error

        # Integral term with anti-windup
        if dt > 0:
            self.integral += error * dt
            # Clamp integral to prevent windup
            self.integral = max(-self.integral_max, min(self.integral_max, self.integral))

        integral_term = self.ki * self.integral

        # Derivative term
        derivative_term = 0.0
        if dt > 0:
            derivative = (error - self.previous_error) / dt
            derivative_term = self.kd * derivative

        # Calculate output
        output = proportional + integral_term + derivative_term

        # Clamp output to valid range
        output = max(self.output_min, min(self.output_max, output))

        # Update state
        self.previous_error = error

        # Performance tracking
        self.total_calls += 1
        self.avg_error = ((self.avg_error * (self.total_calls - 1)) + abs(error)) / self.total_calls

        return output

    def reset(self):
        """Reset PID controller state"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.total_calls = 0
        self.avg_error = 0.0

    def tune(self, kp: float = None, ki: float = None, kd: float = None):
        """Update PID gains while running"""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

    def get_gains(self) -> tuple:
        """Get current PID gains"""
        return (self.kp, self.ki, self.kd)

    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'total_calls': self.total_calls,
            'average_error': self.avg_error,
            'current_integral': self.integral,
            'gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        }

class MotorPIDController:
    """Dual PID controller for differential motor control"""

    def __init__(self,
                 left_gains: tuple = (1.2, 0.5, 0.1),
                 right_gains: tuple = (1.2, 0.5, 0.1),
                 pwm_min: float = 30.0,
                 pwm_max: float = 80.0):
        """
        Initialize dual motor PID controllers

        Args:
            left_gains: (kp, ki, kd) for left motor
            right_gains: (kp, ki, kd) for right motor
            pwm_min: Minimum PWM percentage
            pwm_max: Maximum PWM percentage
        """
        self.left_pid = PIDController(
            kp=left_gains[0], ki=left_gains[1], kd=left_gains[2],
            output_min=pwm_min, output_max=pwm_max
        )

        self.right_pid = PIDController(
            kp=right_gains[0], ki=right_gains[1], kd=right_gains[2],
            output_min=pwm_min, output_max=pwm_max
        )

        # Control settings
        self.update_rate = 100  # 100Hz control loop
        self.last_update_time = time.time()

    def update_motors(self, target_left_rpm: float, target_right_rpm: float,
                     current_left_rpm: float, current_right_rpm: float) -> tuple:
        """
        Update both motor PID controllers

        Returns:
            (left_pwm, right_pwm) tuple
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        left_pwm = self.left_pid.update(target_left_rpm, current_left_rpm, dt)
        right_pwm = self.right_pid.update(target_right_rpm, current_right_rpm, dt)

        return (left_pwm, right_pwm)

    def reset_controllers(self):
        """Reset both PID controllers"""
        self.left_pid.reset()
        self.right_pid.reset()

    def tune_left(self, kp: float = None, ki: float = None, kd: float = None):
        """Tune left motor PID gains"""
        self.left_pid.tune(kp, ki, kd)

    def tune_right(self, kp: float = None, ki: float = None, kd: float = None):
        """Tune right motor PID gains"""
        self.right_pid.tune(kp, ki, kd)

    def get_stats(self) -> dict:
        """Get performance statistics for both controllers"""
        return {
            'left': self.left_pid.get_stats(),
            'right': self.right_pid.get_stats(),
            'control_rate': self.update_rate
        }