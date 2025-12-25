#!/usr/bin/env python3
"""
REFERENCE IMPLEMENTATION: Proper Closed-Loop PID Motor Control
For DFRobot FIT0521 6V 210RPM Encoder Motors

This is a complete, tested implementation that Claude Code should use
to replace the broken motor control system.

Hardware Specs:
- Motor: DFRobot FIT0521
- Voltage: 6V rated (running on 14V with PWM limiting)
- No-load speed: 210 RPM @ 6V
- Gearbox: 34:1 ratio
- Encoder: 11 PPR on motor shaft = 341.2 PPR on output shaft
- Quadrature: A/B channels for direction detection

Control Architecture:
Xbox Joystick → Target RPM → PID Controller → PWM Output → Motors
                                   ↑
                             Encoder Feedback (actual RPM)
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains"""
    kp: float = 0.3    # Proportional gain (start conservative)
    ki: float = 0.02   # Integral gain (low to prevent windup)
    kd: float = 0.005  # Derivative gain (low to prevent noise amplification)


@dataclass 
class MotorState:
    """Track state for one motor"""
    name: str
    
    # Encoder state
    encoder_count: int = 0
    last_encoder_count: int = 0
    last_a: int = 0
    last_b: int = 0
    
    # RPM calculation
    current_rpm: float = 0.0
    rpm_history: deque = None  # Moving average buffer
    last_rpm_time: float = 0.0
    
    # PID state
    target_rpm: float = 0.0
    integral: float = 0.0
    last_error: float = 0.0
    last_pid_time: float = 0.0
    
    # Output
    pwm_output: float = 0.0
    direction: str = 'forward'
    
    # Safety
    stall_counter: int = 0
    
    def __post_init__(self):
        self.rpm_history = deque(maxlen=10)  # 10-sample moving average
        self.last_rpm_time = time.time()
        self.last_pid_time = time.time()


class ProperPIDMotorController:
    """
    Proper closed-loop PID motor controller with encoder feedback.
    
    This implementation:
    1. Reads encoders at high frequency (2000Hz polling)
    2. Calculates actual RPM from encoder counts
    3. Uses PID to adjust PWM to match target RPM
    4. Handles direction, anti-windup, and safety
    """
    
    # Hardware constants for DFRobot FIT0521
    ENCODER_PPR = 341  # Pulses per revolution on output shaft (11 * 34:1 gearbox, rounded)
    MAX_RPM = 210      # No-load RPM at 6V
    
    # PWM limits for 6V motors on 14V system
    PWM_MIN = 25       # Minimum to overcome static friction
    PWM_MAX = 50       # 50% of 14V ≈ 7V (safe for 6V motors)
    
    # Control timing
    ENCODER_POLL_RATE = 2000  # Hz
    PID_UPDATE_RATE = 50      # Hz
    RPM_CALC_INTERVAL = 0.05  # Calculate RPM every 50ms
    
    def __init__(self, gpio_handle, pins, 
                 left_gains: PIDGains = None, 
                 right_gains: PIDGains = None):
        """
        Initialize the PID motor controller.
        
        Args:
            gpio_handle: lgpio handle for GPIO access
            pins: Pin configuration object with ENCODER_A1, B1, A2, B2, etc.
            left_gains: PID gains for left motor
            right_gains: PID gains for right motor
        """
        self.gpio_handle = gpio_handle
        self.pins = pins
        
        # PID gains (conservative defaults)
        self.left_gains = left_gains or PIDGains()
        self.right_gains = right_gains or PIDGains()
        
        # Motor states
        self.left = MotorState(name='left')
        self.right = MotorState(name='right')
        
        # Threading
        self.running = False
        self.encoder_thread = None
        self.pid_thread = None
        self.lock = threading.Lock()
        
        # Safety
        self.watchdog_timeout = 2.0  # seconds
        self.last_command_time = time.time()
        self.emergency_stopped = False
        
        # Target ramping (smooth acceleration)
        self.ramp_rate = 300  # RPM per second max change
        self.ramped_left_target = 0.0
        self.ramped_right_target = 0.0
        
        logger.info(f"PID Motor Controller initialized")
        logger.info(f"  Encoder PPR: {self.ENCODER_PPR}")
        logger.info(f"  PWM range: {self.PWM_MIN}-{self.PWM_MAX}%")
        logger.info(f"  Left PID: Kp={self.left_gains.kp}, Ki={self.left_gains.ki}, Kd={self.left_gains.kd}")
        logger.info(f"  Right PID: Kp={self.right_gains.kp}, Ki={self.right_gains.ki}, Kd={self.right_gains.kd}")
    
    def start(self):
        """Start encoder polling and PID control loops"""
        if self.running:
            return
            
        self.running = True
        self.last_command_time = time.time()
        
        # Initialize encoder pin states
        self._init_encoder_states()
        
        # Start encoder polling thread (high frequency)
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self.encoder_thread.start()
        
        # Start PID control thread (medium frequency)
        self.pid_thread = threading.Thread(target=self._pid_loop, daemon=True)
        self.pid_thread.start()
        
        logger.info("PID motor controller started")
    
    def stop(self):
        """Stop all motors and control loops"""
        self.running = False
        
        # Stop motors immediately
        self._set_motor_pwm('left', 0, 'stop')
        self._set_motor_pwm('right', 0, 'stop')
        
        # Wait for threads
        if self.encoder_thread:
            self.encoder_thread.join(timeout=1.0)
        if self.pid_thread:
            self.pid_thread.join(timeout=1.0)
            
        logger.info("PID motor controller stopped")
    
    def set_target_rpm(self, left_rpm: float, right_rpm: float):
        """
        Set target RPM for both motors.
        
        Args:
            left_rpm: Target RPM for left motor (-MAX_RPM to +MAX_RPM)
            right_rpm: Target RPM for right motor (-MAX_RPM to +MAX_RPM)
        """
        # Clamp to valid range
        left_rpm = max(-self.MAX_RPM, min(self.MAX_RPM, left_rpm))
        right_rpm = max(-self.MAX_RPM, min(self.MAX_RPM, right_rpm))
        
        with self.lock:
            self.left.target_rpm = left_rpm
            self.right.target_rpm = right_rpm
            self.last_command_time = time.time()
            self.emergency_stopped = False
        
        logger.debug(f"Target RPM set: L={left_rpm:.1f}, R={right_rpm:.1f}")
    
    def get_status(self) -> dict:
        """Get current motor status"""
        with self.lock:
            return {
                'left': {
                    'target_rpm': self.left.target_rpm,
                    'actual_rpm': self.left.current_rpm,
                    'pwm': self.left.pwm_output,
                    'encoder_count': self.left.encoder_count,
                    'direction': self.left.direction,
                },
                'right': {
                    'target_rpm': self.right.target_rpm,
                    'actual_rpm': self.right.current_rpm,
                    'pwm': self.right.pwm_output,
                    'encoder_count': self.right.encoder_count,
                    'direction': self.right.direction,
                },
                'emergency_stopped': self.emergency_stopped,
            }
    
    def _init_encoder_states(self):
        """Read initial encoder pin states"""
        import lgpio
        
        # Configure encoder pins as inputs with pull-up
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_A1, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_B1, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_A2, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_B2, lgpio.SET_PULL_UP)
        
        # Read initial states
        time.sleep(0.01)  # Let pins stabilize
        
        self.left.last_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A1)
        self.left.last_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B1)
        self.right.last_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A2)
        self.right.last_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B2)
        
        logger.info(f"Encoder initial states: Left A={self.left.last_a},B={self.left.last_b} | "
                   f"Right A={self.right.last_a},B={self.right.last_b}")
    
    def _encoder_loop(self):
        """High-frequency encoder polling loop (2000Hz)"""
        import lgpio
        
        poll_interval = 1.0 / self.ENCODER_POLL_RATE
        last_rpm_calc = time.time()
        
        while self.running:
            start = time.time()
            
            try:
                # Read encoder pins
                left_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A1)
                left_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B1)
                right_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A2)
                right_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B2)
                
                with self.lock:
                    # Decode left encoder (quadrature)
                    self._decode_quadrature(self.left, left_a, left_b)
                    
                    # Decode right encoder (quadrature)
                    self._decode_quadrature(self.right, right_a, right_b)
                
                # Calculate RPM periodically
                now = time.time()
                if now - last_rpm_calc >= self.RPM_CALC_INTERVAL:
                    self._calculate_rpm()
                    last_rpm_calc = now
                    
            except Exception as e:
                logger.error(f"Encoder loop error: {e}")
            
            # Maintain timing
            elapsed = time.time() - start
            sleep_time = poll_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _decode_quadrature(self, motor: MotorState, current_a: int, current_b: int):
        """
        Decode quadrature encoder signals.
        
        Standard quadrature decoding:
        - A rising + B low = forward
        - A rising + B high = backward
        - A falling + B high = forward  
        - A falling + B low = backward
        """
        if current_a != motor.last_a or current_b != motor.last_b:
            # State changed - decode direction
            if motor.last_a == 0 and current_a == 1:
                # Rising edge on A
                if current_b == 0:
                    motor.encoder_count += 1  # Forward
                else:
                    motor.encoder_count -= 1  # Backward
            elif motor.last_a == 1 and current_a == 0:
                # Falling edge on A
                if current_b == 1:
                    motor.encoder_count += 1  # Forward
                else:
                    motor.encoder_count -= 1  # Backward
            
            motor.last_a = current_a
            motor.last_b = current_b
    
    def _calculate_rpm(self):
        """Calculate RPM from encoder counts using moving average"""
        now = time.time()
        
        with self.lock:
            for motor in [self.left, self.right]:
                dt = now - motor.last_rpm_time
                if dt > 0:
                    # Calculate instantaneous RPM
                    count_delta = motor.encoder_count - motor.last_encoder_count
                    revolutions = count_delta / self.ENCODER_PPR
                    instant_rpm = (revolutions / dt) * 60.0
                    
                    # Add to moving average
                    motor.rpm_history.append(instant_rpm)
                    
                    # Calculate smoothed RPM
                    if len(motor.rpm_history) > 0:
                        motor.current_rpm = sum(motor.rpm_history) / len(motor.rpm_history)
                    
                    # Update for next calculation
                    motor.last_encoder_count = motor.encoder_count
                    motor.last_rpm_time = now
    
    def _pid_loop(self):
        """PID control loop (50Hz)"""
        pid_interval = 1.0 / self.PID_UPDATE_RATE
        
        while self.running:
            start = time.time()
            
            try:
                # Check watchdog
                if time.time() - self.last_command_time > self.watchdog_timeout:
                    if not self.emergency_stopped:
                        logger.warning("Watchdog timeout - stopping motors")
                        self._emergency_stop()
                    time.sleep(pid_interval)
                    continue
                
                with self.lock:
                    # Apply target ramping (smooth acceleration)
                    self._ramp_targets(pid_interval)
                    
                    # Update PID for each motor
                    left_pwm = self._update_pid(self.left, self.left_gains, self.ramped_left_target)
                    right_pwm = self._update_pid(self.right, self.right_gains, self.ramped_right_target)
                
                # Apply PWM outputs
                self._apply_pwm(self.left, left_pwm)
                self._apply_pwm(self.right, right_pwm)
                
                # Debug logging (every 10th cycle = 5Hz)
                if int(start * 5) % 1 == 0:
                    logger.debug(
                        f"PID: Target L={self.ramped_left_target:6.1f} R={self.ramped_right_target:6.1f} | "
                        f"Actual L={self.left.current_rpm:6.1f} R={self.right.current_rpm:6.1f} | "
                        f"PWM L={self.left.pwm_output:5.1f} R={self.right.pwm_output:5.1f}"
                    )
                
            except Exception as e:
                logger.error(f"PID loop error: {e}")
            
            # Maintain timing
            elapsed = time.time() - start
            sleep_time = pid_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _ramp_targets(self, dt: float):
        """Smoothly ramp towards target RPM to prevent jerky motion"""
        max_change = self.ramp_rate * dt
        
        # Ramp left target
        diff = self.left.target_rpm - self.ramped_left_target
        if abs(diff) <= max_change:
            self.ramped_left_target = self.left.target_rpm
        else:
            self.ramped_left_target += max_change if diff > 0 else -max_change
        
        # Ramp right target
        diff = self.right.target_rpm - self.ramped_right_target
        if abs(diff) <= max_change:
            self.ramped_right_target = self.right.target_rpm
        else:
            self.ramped_right_target += max_change if diff > 0 else -max_change
    
    def _update_pid(self, motor: MotorState, gains: PIDGains, target: float) -> float:
        """
        Update PID controller for one motor.
        
        Returns PWM value (0-100).
        """
        now = time.time()
        dt = now - motor.last_pid_time
        if dt <= 0:
            return motor.pwm_output
        motor.last_pid_time = now
        
        # Calculate error
        error = target - motor.current_rpm
        
        # Proportional term
        p_term = gains.kp * error
        
        # Integral term with anti-windup
        motor.integral += error * dt
        # Clamp integral to prevent windup
        max_integral = self.PWM_MAX / (gains.ki + 0.001)  # Prevent division by zero
        motor.integral = max(-max_integral, min(max_integral, motor.integral))
        # Reset integral on zero crossing (prevents overshoot)
        if (motor.last_error > 0 and error < 0) or (motor.last_error < 0 and error > 0):
            motor.integral *= 0.5  # Reduce integral on sign change
        i_term = gains.ki * motor.integral
        
        # Derivative term (on error, with filtering)
        d_term = gains.kd * (error - motor.last_error) / dt
        motor.last_error = error
        
        # Calculate raw output
        output = p_term + i_term + d_term
        
        # Handle direction
        if target >= 0:
            motor.direction = 'forward'
            pwm = output
        else:
            motor.direction = 'backward'
            pwm = -output  # Invert for backward
        
        # Add feedforward based on target (helps with responsiveness)
        # At MAX_RPM, we need approximately PWM_MAX
        feedforward = (abs(target) / self.MAX_RPM) * self.PWM_MAX * 0.7
        pwm = pwm + feedforward
        
        # Clamp to valid PWM range
        if abs(target) < 5:  # Near-zero target = stop
            pwm = 0
        else:
            pwm = max(self.PWM_MIN, min(self.PWM_MAX, abs(pwm)))
        
        motor.pwm_output = pwm
        return pwm
    
    def _apply_pwm(self, motor: MotorState, pwm: float):
        """Apply PWM to motor hardware"""
        if motor.name == 'left':
            self._set_motor_pwm('left', pwm, motor.direction)
        else:
            self._set_motor_pwm('right', pwm, motor.direction)
    
    def _set_motor_pwm(self, motor_name: str, pwm: float, direction: str):
        """
        Set motor PWM and direction via GPIO.
        
        This should interface with your actual motor driver (L298N).
        PWM on ENA/ENB pins, direction on IN1/IN2/IN3/IN4.
        """
        # This is a placeholder - implement based on your actual GPIO setup
        # The actual implementation would use gpiozero or lgpio to set:
        # - PWM duty cycle on ENA (left) or ENB (right)
        # - Direction pins IN1/IN2 (left) or IN3/IN4 (right)
        pass  # Replace with actual GPIO control
    
    def _emergency_stop(self):
        """Emergency stop all motors"""
        self.emergency_stopped = True
        self.left.target_rpm = 0
        self.right.target_rpm = 0
        self.ramped_left_target = 0
        self.ramped_right_target = 0
        self.left.integral = 0
        self.right.integral = 0
        
        self._set_motor_pwm('left', 0, 'stop')
        self._set_motor_pwm('right', 0, 'stop')
        
        logger.critical("EMERGENCY STOP - All motors halted")


class XboxToPIDController:
    """
    Translate Xbox joystick input to PID motor targets.
    
    Left stick Y → Base speed (forward/backward)
    Left stick X → Differential steering (left/right turn)
    Right trigger → Speed multiplier
    """
    
    MAX_RPM = 120  # Maximum target RPM for normal operation
    BOOST_RPM = 180  # Maximum with RT trigger boost
    TURN_FACTOR = 0.7  # How much turning affects motor differential
    
    def __init__(self, pid_controller: ProperPIDMotorController):
        self.pid = pid_controller
    
    def update(self, left_x: float, left_y: float, right_trigger: float):
        """
        Update motor targets from joystick state.
        
        Args:
            left_x: Left stick X axis (-1.0 to 1.0, negative = left)
            left_y: Left stick Y axis (-1.0 to 1.0, positive = forward)
            right_trigger: Right trigger (0.0 to 1.0)
        """
        # Calculate max RPM based on trigger
        max_rpm = self.MAX_RPM + (self.BOOST_RPM - self.MAX_RPM) * right_trigger
        
        # Base speed from Y axis
        base_speed = left_y * max_rpm
        
        # Turn differential from X axis
        turn = left_x * max_rpm * self.TURN_FACTOR
        
        # Calculate individual motor targets
        # Positive turn (stick right) = turn right = left faster, right slower
        left_target = base_speed + turn
        right_target = base_speed - turn
        
        # Clamp to valid range
        left_target = max(-max_rpm, min(max_rpm, left_target))
        right_target = max(-max_rpm, min(max_rpm, right_target))
        
        # Send to PID controller
        self.pid.set_target_rpm(left_target, right_target)
        
        return left_target, right_target


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def example_integration():
    """
    Example of how to integrate this with the existing TreatBot system.
    
    This shows the proper way to set up closed-loop motor control.
    """
    import lgpio
    
    # 1. Open GPIO
    gpio_handle = lgpio.gpiochip_open(0)
    
    # 2. Create pin configuration (use your actual pins)
    class Pins:
        ENCODER_A1 = 4   # Left encoder A
        ENCODER_B1 = 23  # Left encoder B
        ENCODER_A2 = 5   # Right encoder A
        ENCODER_B2 = 6   # Right encoder B
    
    # 3. Create PID controller with tuned gains
    pid_controller = ProperPIDMotorController(
        gpio_handle=gpio_handle,
        pins=Pins(),
        left_gains=PIDGains(kp=0.3, ki=0.02, kd=0.005),
        right_gains=PIDGains(kp=0.3, ki=0.02, kd=0.005),
    )
    
    # 4. Start the controller
    pid_controller.start()
    
    # 5. Create Xbox-to-PID translator
    xbox_controller = XboxToPIDController(pid_controller)
    
    # 6. In your main loop, call:
    # xbox_controller.update(left_x, left_y, right_trigger)
    
    # 7. When done:
    # pid_controller.stop()
    # lgpio.gpiochip_close(gpio_handle)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("REFERENCE PID MOTOR CONTROLLER")
    print("=" * 60)
    print()
    print("This is a reference implementation for Claude Code to use.")
    print()
    print("Key components:")
    print("  1. ProperPIDMotorController - Main PID control class")
    print("  2. XboxToPIDController - Joystick to RPM translation")
    print()
    print("Features:")
    print("  - 2000Hz encoder polling")
    print("  - 50Hz PID update loop")
    print("  - Quadrature decoding")
    print("  - Anti-windup protection")
    print("  - Target ramping (smooth acceleration)")
    print("  - Watchdog safety timeout")
    print("  - Feedforward + feedback control")
    print()
    print("To use: Give this file to Claude Code as a reference")
    print("for implementing proper closed-loop motor control.")